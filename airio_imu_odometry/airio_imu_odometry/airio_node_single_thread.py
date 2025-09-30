import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from collections import deque

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

from airio_imu_odometry.airimu_wrapper import AirIMUCorrector, ImuData
from airio_imu_odometry.airio_wrapper import AirIOWrapper, RawImuSample
from airio_imu_odometry.velocity_integrator import VelocityIntegrator
from airio_imu_odometry.tools import _so3_from_xyzw

import numpy as np
import threading
import math
import pypose as pp
import torch
import time

import os
import matplotlib.pyplot as plt

class AirIoImuOdomNode(Node):
    def __init__(self):
        super().__init__('airio_imu_odometry_single_thread')

        # --- Parameters ---
        self.declare_parameter("airimu_root", "")
        self.declare_parameter("airimu_ckpt", "")
        self.declare_parameter("airimu_conf", "")
        self.declare_parameter("airio_root", "")
        self.declare_parameter("airio_ckpt", "")
        self.declare_parameter("airio_conf", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("airimu_seqlen", 200)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("timming_logging_mode",False)
        self.declare_parameter("timming_logging_outputpath",".")

        airimu_root = self.get_parameter("airimu_root").get_parameter_value().string_value
        airimu_ckpt = self.get_parameter("airimu_ckpt").get_parameter_value().string_value
        airimu_conf = self.get_parameter("airimu_conf").get_parameter_value().string_value
        airio_root  = self.get_parameter("airio_root").get_parameter_value().string_value
        airio_ckpt  = self.get_parameter("airio_ckpt").get_parameter_value().string_value
        airio_conf  = self.get_parameter("airio_conf").get_parameter_value().string_value
        device      = self.get_parameter("device").get_parameter_value().string_value
        seqlen      = int(self.get_parameter("airimu_seqlen").get_parameter_value().integer_value)
        self.TL_out_path = self.get_parameter("timming_logging_outputpath").get_parameter_value().string_value
        self.TL_mode     = bool(self.get_parameter("timming_logging_mode").get_parameter_value().bool_value)
        self.pub_hz = float(self.get_parameter("publish_rate").get_parameter_value().double_value)

        # --- Init gating ---
        self.initialized = False
        self.init_lock = threading.Lock()
        self.prev_odom = None
        self.init_state = {
            "pos": None,     # [x, y, z]
            "rot": None,     # quaternion [x, y, z, w]
            "vel": None,     # [vx, vy, vz]
            "stamp": None,
        }

        # --- Modules ---
        self.corrector = AirIMUCorrector(
            airimu_root=airimu_root,
            ckpt_path=airimu_ckpt,
            conf_path=airimu_conf,
            device=device,
            seqlen=seqlen,
        )

        # AirIO: velocity network wrapper 
        self.airio = AirIOWrapper(
            airio_root=airio_root,
            ckpt_path=airio_ckpt,
            conf_path=airio_conf,
            device=device,
        )
        # --- Pypose INTEGRATOR ---
        self.pp_integrator = None
        self.last_integrated_stamp = None
        self.gravity = 9.81007
        self.pp_dev = None

        # --- Subscribers & Publishers ---
        self.create_subscription(Imu, '/imu/data_raw', self.imu_callback, 1000)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10) 
        self.odom_pub = self.create_publisher(Odometry, '/odom_airio', 10)
        self.filtered_pub = self.create_publisher(Imu, '/airimu_imu_data', 10)
        self.imu_sec = None
        self.imu_nanosec = None

        # --- Timer ---
        period = 1.0 / max(1.0, self.pub_hz)
        self.timer = self.create_timer(period, self.on_timer)

        if self.corrector.ready:
            self.get_logger().info("AIR-IMU ready.")
        else:
            self.get_logger().warn("AIR-IMU in pass-through mode.")

        if self.airio.ready:
            self.get_logger().info("AIR-IO ready (velocity net).")
        else:
            self.get_logger().warn("AIR-IO in pass-through mode (velocity=0).")

        self.get_logger().info("Waiting for /odom to initialize...")

        # --- Velocity_Integrator ---
        self.net_vel_is_body = True

        self.vel_integ = None  

        # --- Timming_Logging ---
        self.airimu_step_t_deque = deque(maxlen = 5000)
        self.airimu_rot_step_t_deque = deque(maxlen = 5000)
        self.airio_network_step_t_deque = deque(maxlen = 5000)
        self.velocity_integrator_step_t_deque = deque(maxlen = 5000)
        self.total_t_deque = deque(maxlen = 5000)
    
    def _diff_velocity(self, prev: Odometry, curr: Odometry):
        p0, p1 = prev.pose.pose.position, curr.pose.pose.position
        t0 = prev.header.stamp.sec + prev.header.stamp.nanosec * 1e-9
        t1 = curr.header.stamp.sec + curr.header.stamp.nanosec * 1e-9
        dt = t1 - t0
        if dt <= 0.0:
            raise ValueError(f"non-positive dt: {dt}")
        vx = (p1.x - p0.x) / dt
        vy = (p1.y - p0.y) / dt
        vz = (p1.z - p0.z) / dt
        return [vx, vy, vz], t1
    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_callback(self, msg: Odometry):
        # 최초 1회만 초기화
        if self.initialized:
            return
        with self.init_lock:
            if self.initialized:
                return
        
        if self.prev_odom is None:
            self.prev_odom = msg
            self.get_logger().info("First /odom buffered (no init yet). Waiting next /odom for diff-velocity.")
            return

        try:
            vel, stamp = self._diff_velocity(self.prev_odom, msg)
        except Exception as e:
            # dt<=0 등 예외 시, 초기화 보류(다음 프레임에서 재시도)
            self.prev_odom = msg  # 그래도 최신값으로 갱신
            self.get_logger().warn(f"diff-velocity failed; deferring init. reason={e}")
            return

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        self.init_state = {
            "pos": [p.x, p.y, p.z],
            "rot": [q.x, q.y, q.z, q.w],
            "vel": [v.x, v.y, v.z],
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        }

        # (선택) 내부 래퍼/적분기에 초기 상태를 넘길 훅이 있다면 여기서 호출
        try:
            if hasattr(self.airio, "set_init_state"):
                self.airio.set_init_state(self.init_state)
            if hasattr(self.corrector, "set_init_state"):
                self.corrector.set_init_state(self.init_state)
        except Exception as e:
            self.get_logger().warn(f"set_init_state hook failed: {e}")
        
        # --- Integrator 초기화 ---
        try:
            pos0 = torch.tensor(self.init_state["pos"], dtype=torch.float64)
            vel0 = torch.tensor(self.init_state["vel"], dtype=torch.float64)
            qx, qy, qz, qw = self.init_state["rot"]  # ROS: [x,y,z,w]
            rot0 = _so3_from_xyzw(qx, qy, qz, qw, device=self.pp_dev)

            device_str = self.get_parameter("device").get_parameter_value().string_value
            self.pp_dev = torch.device(device_str)
            self.pp_integrator = pp.module.IMUPreintegrator(
                pos0, rot0, vel0, gravity=self.gravity, reset=False
            ).to(self.pp_dev).double()

            self.last_integrated_stamp = None  # 적분 시각 초기화
            
        except Exception as e:
            self.get_logger().error(f"IMUPreintegrator init failed: {e}")
            return
        # -----------

        # --- Velocity_Integrator ---
        try:
            init_pos = torch.tensor(self.init_state["pos"], dtype=torch.float64)
            self.vel_integ = VelocityIntegrator(
                init_pos, frame=('body' if self.net_vel_is_body else 'world'),
                method='trapezoid', device='cpu'  # device 파라미터는 네 환경에 맞춰
            ).double()
            self.last_integrated_stamp = None

        except Exception as e:
            self.get_logger().error(f"VelocityIntegrator init failed: {e}")
            return
        self.get_logger().info("Integrator ready.")
        # --- --- --- --- --- ---
        
        self.initialized = True

        self.get_logger().info(
            f"/odom received. Initialized with "
            f"pos={self.init_state['pos']}, "
            f"vel={self.init_state['vel']}"
        )


    def imu_callback(self, msg: Imu):
        if not self.initialized:
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_sec = msg.header.stamp.sec
        self.imu_nanosec = msg.header.stamp.nanosec

        imu_in = ImuData(
            wx=msg.angular_velocity.x, wy=msg.angular_velocity.y, wz=msg.angular_velocity.z,
            ax=msg.linear_acceleration.x, ay=msg.linear_acceleration.y, az=msg.linear_acceleration.z,
            qx=msg.orientation.x, qy=msg.orientation.y, qz=msg.orientation.z, qw=msg.orientation.w,
            stamp=stamp
        )
        self.corrector.add_sample(imu_in)
        self.airio.add_sample(imu_in)
        

    def on_timer(self):
        if not self.initialized:
            return
        # AIR-IMU 보정 결과 꺼내기
        t0 = time.time()
        self.corrector.set_init_state(self.init_state)
        imu_out = self.corrector.correct_latest()
        if imu_out is None:
            return
        airimu_step_t = time.time()-t0
        
        # 보정된 IMU data로 airimu_rot 만들기
        # 같은 샘플을 중복 적분하지 않도록 가드
        t1 = time.time()        
        if self.last_integrated_stamp is not None and imu_out.stamp <= self.last_integrated_stamp + 1e-12:
            return
        # dt 계산
        if self.last_integrated_stamp is None:
            self.last_integrated_stamp = imu_out.stamp
            return  # 다음 틱부터 적분 시작

        dt = max(1e-6, imu_out.stamp - self.last_integrated_stamp)
        self.last_integrated_stamp = imu_out.stamp

        # integrator 적용
        try:
            dev = self.pp_dev
            dt_t  = torch.tensor([[[dt]]], dtype=torch.float64, device=dev)  # (1,1,1)
            gyr_t = torch.tensor([[[imu_out.wx, imu_out.wy, imu_out.wz]]], dtype=torch.float64, device=dev)  # (1,1,3)
            acc_t = torch.tensor([[[imu_out.ax, imu_out.ay, imu_out.az]]], dtype=torch.float64, device=dev)  # (1,1,3)
            rot_step = _so3_from_xyzw(imu_out.qx, imu_out.qy, imu_out.qz, imu_out.qw, device=dev)
            
            state = self.pp_integrator(init_state=None, dt=dt_t, gyro=gyr_t, acc=acc_t, rot=None)

            cur_pos = state['pos'][..., -1, :].detach().cpu().numpy().ravel()
            cur_vel = state['vel'][..., -1, :].detach().cpu().numpy().ravel()
            cur_rot = state['rot'][..., -1, :].detach().cpu().numpy().ravel() #--> airimu_rot
        except Exception as e:
            self.get_logger().warn(f"IMUPreintegrator step failed: {e}")
            return
        airimu_rot_step_t = time.time() - t1

        # AIR-IO Network
        t2 = time.time() 
        net_vel = self.airio.predict_velocity(cur_rot)
        airio_network_step_t = time.time() - t2

        # Velocity기반 적분
        t3 = time.time()
        v_body = np.asarray(net_vel, dtype=float)   
        
        try:
            ego_pos = self.vel_integ.step(dt, v_body, orient=cur_rot )
            ego_pos = ego_pos.detach().cpu().numpy()

        except Exception as e:
            self.get_logger().warn(f"Velocity integration failed: {e}")
            return
        velocity_integrator_step_t = time.time() - t3
        total_t = time.time() - t0

        if self.TL_mode:
            if total_t > 0.1: 
                pass
            else:
                self.airimu_step_t_deque.append(airimu_step_t)
                self.airimu_rot_step_t_deque.append(airimu_rot_step_t)
                self.airio_network_step_t_deque.append(airio_network_step_t)
                self.velocity_integrator_step_t_deque.append(velocity_integrator_step_t)
                self.total_t_deque.append(total_t)
    
                self.get_logger().info(f"AIR-IMU step 소요={airimu_step_t:.6f}s")
                self.get_logger().info(f"AIRIMU Rot step 소요={airimu_rot_step_t:.6f}s")
                self.get_logger().info(f"AIR-IO Network step 소요={airio_network_step_t:.6f}s")
                self.get_logger().info(f"VelocityIntegrator step 소요={velocity_integrator_step_t:.6f}s")
                self.get_logger().info(f"on_timer 전체 처리시간={total_t:.6f}s")

        # 보정된 IMU re-publish
        imu_msg = Imu()
        imu_msg.header.stamp.sec = self.imu_sec
        imu_msg.header.stamp.nanosec = self.imu_nanosec
        imu_msg.header.frame_id = "base_link"
        imu_msg.angular_velocity.x = imu_out.wx
        imu_msg.angular_velocity.y = imu_out.wy
        imu_msg.angular_velocity.z = imu_out.wz
        imu_msg.linear_acceleration.x = imu_out.ax
        imu_msg.linear_acceleration.y = imu_out.ay
        imu_msg.linear_acceleration.z = imu_out.az
        imu_msg.orientation.x = imu_out.qx
        imu_msg.orientation.y = imu_out.qy
        imu_msg.orientation.z = imu_out.qz
        imu_msg.orientation.w = imu_out.qw
        self.filtered_pub.publish(imu_msg)

        # Odometry publish
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(ego_pos[0])
        odom.pose.pose.position.y = float(ego_pos[1])
        odom.pose.pose.position.z = float(ego_pos[2])
        odom.pose.pose.orientation.x = float(cur_rot[0])
        odom.pose.pose.orientation.y = float(cur_rot[1])
        odom.pose.pose.orientation.z = float(cur_rot[2])
        odom.pose.pose.orientation.w = float(cur_rot[3])

        odom.twist.twist.linear.x = float(v_body[0])
        odom.twist.twist.linear.y = float(v_body[1])
        odom.twist.twist.linear.z = float(v_body[2])

        self.odom_pub.publish(odom)

    def save_timings(self):
        timings = {
            "AIR-IMU": list(self.airimu_step_t_deque),
            "AIR-IMU RotStep": list(self.airimu_rot_step_t_deque),
            "AIR-IO Network": list(self.airio_network_step_t_deque),
            "VelocityIntegrator": list(self.velocity_integrator_step_t_deque),
            "Total": list(self.total_t_deque),
        }
    
        # self.TL_out_path 가 파일 이름이면 → 디렉터리와 prefix 분리
        outdir = os.path.dirname(self.TL_out_path)
        prefix = os.path.splitext(os.path.basename(self.TL_out_path))[0]
    
        # 디렉터리 없으면 생성
        os.makedirs(outdir, exist_ok=True)
    
        for name, values in timings.items():
            plt.figure(figsize=(8, 4))
            ms_values = [v * 1000 for v in values]
            plt.plot(ms_values, marker='o', markersize=2, linewidth=0.7, label=name)
            if values:
                avg = sum(values) / len(values) * 1000
                plt.axhline(avg, color='red', linestyle='--', label=f"avg={avg:.6f}ms")
                plt.axhline(min(values), color='green', linestyle=':', label=f"min={min(values)* 1000:.6f}ms")
                plt.axhline(max(values), color='orange', linestyle=':', label=f"max={max(values* 1000):.6f}ms")
            plt.ylabel("millisecond")
            plt.xlabel("#")
            plt.title(f"{name}")
            plt.legend(loc="upper right")
            plt.grid(True)
    
            # outdir/prefix_name.png 형태로 저장
            safe_name = name.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(outdir, f"{prefix}_{safe_name}.png")
    
            plt.tight_layout()
            plt.savefig(out_path, dpi=130)
            plt.close()
            print(f"[INFO] Timing plot saved to: {out_path}")

def main(args=None):
    rclpy.init(args=args)
    node = AirIoImuOdomNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        if node.TL_mode:
            node.save_timings()
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # launch 사용 시 rclpy.shutdown() 중복 호출 방지
        try:
            rclpy.shutdown()
        except Exception:
            pass
