import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from collections import deque

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped  # === MOD ===
import tf2_ros  # === MOD ===

from airio_imu_odometry.airimu_wrapper import AirIMUCorrector, ImuData
from airio_imu_odometry.airio_wrapper import AirIOWrapper
from airio_imu_odometry.velocity_integrator import VelocityIntegrator
from airio_imu_odometry.tools import _so3_from_xyzw

import numpy as np
import threading
import torch
import time
import os
import matplotlib.pyplot as plt
import pypose as pp


# === MOD: SE3/Quat helpers ===
def _q_norm(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q if n < 1e-12 else q / n

def _q_mul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return _q_norm([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def _q_conj(q):
    x, y, z, w = q; return [-x, -y, -z, w]

def _q_slerp(q0, q1, t):
    q0 = _q_norm(q0); q1 = _q_norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -np.asarray(q1); dot = -dot
    if dot > 0.9995:
        return _q_norm(q0 + t*(np.asarray(q1)-q0))
    th0 = np.arccos(np.clip(dot, -1.0, 1.0)); th = th0 * t
    s0 = np.sin(th0 - th) / np.sin(th0); s1 = np.sin(th) / np.sin(th0)
    return _q_norm(s0*np.asarray(q0) + s1*np.asarray(q1))

def _q_angle(q):
    q = _q_norm(q); ang = 2*np.arccos(np.clip(q[3], -1.0, 1.0))
    return 2*np.pi - ang if ang > np.pi else ang

def _se3_comp(Ta, Tb):
    (ta, qa), (tb, qb) = Ta, Tb
    px, py, pz = tb
    qx, qy, qz, qw = qa
    uvx = 2*(qy*pz - qz*py); uvy = 2*(qz*px - qx*pz); uvz = 2*(qx*py - qy*px)
    uuvx = 2*(qy*uvz - qz*uvy); uuvy = 2*(qz*uvx - qx*uvz); uuvz = 2*(qx*uvy - qy*uvx)
    rx = px + qw*uvx + uuvx; ry = py + qw*uvy + uuvy; rz = pz + qw*uvz + uuvz
    return ([ta[0]+rx, ta[1]+ry, ta[2]+rz], _q_mul(qa, qb))

def _se3_inv(T):
    t, q = T; qi = _q_conj(q)
    px, py, pz = -np.asarray(t)
    qx, qy, qz, qw = qi
    uvx = 2*(qy*pz - qz*py); uvy = 2*(qz*px - qx*pz); uvz = 2*(qx*py - qy*px)
    uuvx = 2*(qy*uvz - qz*uvy); uuvy = 2*(qz*uvx - qx*uvz); uuvz = 2*(qx*uvy - qy*uvx)
    rx = px + qw*uvx + uuvx; ry = py + qw*uvy + uuvy; rz = pz + qw*uvz + uuvz
    return ([rx, ry, rz], qi)

def _apply_delta_alpha(T_map_odom, dT, alpha):
    tmo, qmo = T_map_odom; td, qd = dT
    q_inc = _q_slerp([0,0,0,1], qd, alpha)
    q_new = _q_mul(q_inc, qmo)
    t_new = [tmo[0]+alpha*td[0], tmo[1]+alpha*td[1], tmo[2]+alpha*td[2]]
    return (t_new, q_new)


class AirIoImuOdomNode(Node):
    def __init__(self):
        super().__init__('airio_imu_odometry')

        # === 콜백 그룹 ===
        self.cbgroup_imu   = MutuallyExclusiveCallbackGroup()
        self.cbgroup_timer = MutuallyExclusiveCallbackGroup()

        # --- Parameters ---
        self.declare_parameter("airimu_root", "")
        self.declare_parameter("airimu_ckpt", "")
        self.declare_parameter("airimu_conf", "")
        self.declare_parameter("airio_root", "")
        self.declare_parameter("airio_ckpt", "")
        self.declare_parameter("airio_conf", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("airimu_seqlen", 200)
        self.declare_parameter("publish_rate", 80.0)           # === MOD: 기본값 상향
        self.declare_parameter("timming_logging_mode", False)
        self.declare_parameter("timming_logging_outputpath", ".")

        # === MOD: 프레임/보정/토픽 ===
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("cartographer_odom_topic", "/odom")  # Cartographer가 주는 절대 포즈
        self.declare_parameter("output_odom_topic", "/odom_fused")        # 최종 퍼블리시(충돌시 변경)
        self.declare_parameter("alpha_map_correction", 0.15)
        self.declare_parameter("gate_trans_m", 1.0)
        self.declare_parameter("gate_yaw_deg", 10.0)
        self.declare_parameter("publish_tf", False)  # 기본 끔(충돌 방지)

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
        self.pub_hz      = float(self.get_parameter("publish_rate").get_parameter_value().double_value)

        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.odom_frame  = self.get_parameter("odom_frame").get_parameter_value().string_value
        self.base_frame  = self.get_parameter("base_frame").get_parameter_value().string_value
        self.carto_topic = self.get_parameter("cartographer_odom_topic").get_parameter_value().string_value
        self.output_odom_topic = self.get_parameter("output_odom_topic").get_parameter_value().string_value
        self.alpha       = float(self.get_parameter("alpha_map_correction").get_parameter_value().double_value)
        self.gate_trans  = float(self.get_parameter("gate_trans_m").get_parameter_value().double_value)
        self.gate_yaw    = float(self.get_parameter("gate_yaw_deg").get_parameter_value().double_value) * np.pi/180.0
        self.pub_tf      = bool(self.get_parameter("publish_tf").get_parameter_value().bool_value)

        # --- Init gating (대기/락 제거) ===
        self.initialized = False
        self.init_lock   = threading.Lock()
        self.sample_lock = threading.Lock()
        self.init_state  = {"pos": [0,0,0], "rot": [0,0,0,1], "vel": [0,0,0], "stamp": None}
        self._got_first_imu = False  # === MOD: 첫 IMU로 초기화

        # --- Modules ---
        self.corrector = AirIMUCorrector(
            airimu_root=airimu_root, ckpt_path=airimu_ckpt, conf_path=airimu_conf,
            device=device, seqlen=seqlen
        )
        self.airio = AirIOWrapper(
            airio_root=airio_root, ckpt_path=airio_ckpt, conf_path=airio_conf, device=device
        )

        # --- Integrators ---
        self.pp_integrator = None
        self.last_integrated_stamp = None
        self.gravity = 9.81007
        self.pp_dev = None
        self.net_vel_is_body = True
        self.vel_integ = None

        # --- Subs & Pubs ---
        self.create_subscription(Imu, '/imu/data_raw', self.imu_callback, 1000,
                                 callback_group=self.cbgroup_imu)
        # === MOD: Cartographer 절대 포즈 구독 (초기화/보정 용도만, 대기 안 함)
        self.create_subscription(Odometry, self.carto_topic, self.carto_odom_callback, 50)

        self.odom_pub     = self.create_publisher(Odometry, self.output_odom_topic, 10)  # 최종 고주기
        self.filtered_pub = self.create_publisher(Imu, '/airimu_imu_data', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.pub_tf else None

        self.imu_sec = None
        self.imu_nanosec = None

        # --- Timer ---
        period = 1.0 / max(1.0, self.pub_hz)
        self.timer = self.create_timer(period, self.on_timer, callback_group=self.cbgroup_timer)

        # --- Logs ---
        self.get_logger().info("Air-IO & IMU pipelines ready. Not waiting for Cartographer /odom.")

        # --- Timings ---
        self.airimu_step_t_deque = deque(maxlen=5000)
        self.airimu_rot_step_t_deque = deque(maxlen=5000)
        self.airio_network_step_t_deque = deque(maxlen=5000)
        self.velocity_integrator_step_t_deque = deque(maxlen=5000)
        self.total_t_deque = deque(maxlen=5000)

        # --- ZUPT ---
        self.zupt_win_sec = 0.3
        self.gyro_thr     = 0.02
        self.acc_thr      = 0.15
        self.deadband_ms  = 5.0
        self.max_dt       = 0.2

        self._imu_hist = deque(maxlen=2000)
        self._last_ego_pos = np.zeros(3, dtype=float)

        # === MOD: Fusion states ===
        self.T_map_odom = ([0.0,0.0,0.0], [0.0,0.0,0.0,1.0])  # map->odom 오프셋
        self._map_odom_initialized = False
        self.airio_pose_buf = deque(maxlen=400)  # (t, pos(3), quat(4), vel(3))

    # === MOD: Air-IO 포즈 보간 ===
    def _interp_airio_pose(self, t_query):
        if len(self.airio_pose_buf) < 2:
            return None
        left, right = None, None
        for i in range(len(self.airio_pose_buf)-1):
            t0, p0, q0, v0 = self.airio_pose_buf[i]
            t1, p1, q1, v1 = self.airio_pose_buf[i+1]
            if t0 <= t_query <= t1:
                left, right = (t0, p0, q0, v0), (t1, p1, q1, v1)
                break
        if left is None:
            # 최신 샘플로 최대 50ms 외삽
            t_last, p_last, q_last, v_last = self.airio_pose_buf[-1]
            if 0.0 <= (t_query - t_last) <= 0.05:
                return (p_last, q_last, v_last)
            return None
        (t0, p0, q0, v0), (t1, p1, q1, v1) = left, right
        r = float((t_query - t0) / max(1e-9, (t1 - t0)))
        p = (1.0 - r)*p0 + r*p1
        q = _q_slerp(q0, q1, r)
        v = (1.0 - r)*v0 + r*v1
        return (p, q, v)

    # === MOD: Cartographer /odom 콜백 (대기/락 없음) ===
    def carto_odom_callback(self, msg: Odometry):
        if not self.initialized:
            return  # 아직 로컬 적분 초기화 전이면 스킵(초기화는 IMU가 담당)

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pc = msg.pose.pose.position
        qc = msg.pose.pose.orientation
        T_map_base_carto = ([pc.x, pc.y, pc.z], [qc.x, qc.y, qc.z, qc.w])

        interp = self._interp_airio_pose(stamp)  # 같은 시각의 odom->base_link (Air-IO)
        if interp is None:
            return
        p_ob, q_ob, _ = interp
        T_odom_base_k = (p_ob.tolist(), q_ob.tolist())

        if not self._map_odom_initialized:
            # 초기 정렬: map->odom = Tmap->base * (Tom->base)^-1
            self.T_map_odom = _se3_comp(T_map_base_carto, _se3_inv(T_odom_base_k))
            self._map_odom_initialized = True
            self.get_logger().info("map->odom aligned from first Cartographer /odom.")
            return

        # 이후는 부드러운 보정
        T_map_base_pred = _se3_comp(self.T_map_odom, T_odom_base_k)
        dT = _se3_comp(T_map_base_carto, _se3_inv(T_map_base_pred))
        trans = np.linalg.norm(np.asarray(dT[0])); yaw = _q_angle(dT[1])
        if trans <= self.gate_trans and yaw <= self.gate_yaw:
            self.T_map_odom = _apply_delta_alpha(self.T_map_odom, dT, self.alpha)

    # === 기존 diff_velocity 그대로 사용(내부 init에만 필요) ===
    def _diff_velocity(self, prev: Odometry, curr: Odometry):
        p0, p1 = prev.pose.pose.position, curr.pose.pose.position
        t0 = prev.header.stamp.sec + prev.header.stamp.nanosec * 1e-9
        t1 = curr.header.stamp.sec + curr.header.stamp.nanosec * 1e-9
        dt = t1 - t0
        if dt <= 0.0:
            raise ValueError(f"non-positive dt: {dt}")
        return [(p1.x - p0.x)/dt, (p1.y - p0.y)/dt, (p1.z - p0.z)/dt], t1

    def _is_stationary(self, now_stamp: float) -> bool:
        if not self._imu_hist:
            return False
        win_start = now_stamp - self.zupt_win_sec
        g_vals, a_vals = [], []
        for t, g, a in self._imu_hist:
            if t >= win_start:
                g_vals.append(g); a_vals.append(a)
        if len(g_vals) < 3:
            return False
        return (float(np.mean(g_vals)) < self.gyro_thr) and (float(np.mean(a_vals)) < self.acc_thr)

    # === MOD: IMU 콜백에서 '초기화' 수행 (Cartographer 대기 X) ===
    def imu_callback(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_sec = msg.header.stamp.sec; self.imu_nanosec = msg.header.stamp.nanosec

        # 최초 1회: 로컬(odom) 기준 초기화
        if not self.initialized:
            with self.init_lock:
                if not self.initialized:
                    try:
                        device_str = self.get_parameter("device").get_parameter_value().string_value
                        self.pp_dev = torch.device(device_str)
                        # 초기 pos/vel = 0, 초기 rot = 현재 IMU 쿼터니언
                        pos0 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
                        vel0 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
                        rot0 = _so3_from_xyzw(msg.orientation.x, msg.orientation.y,
                                              msg.orientation.z, msg.orientation.w, device=self.pp_dev)
                        self.pp_integrator = pp.module.IMUPreintegrator(
                            pos0, rot0, vel0, gravity=self.gravity, reset=False
                        ).to(self.pp_dev).double()
                        self.vel_integ = VelocityIntegrator(
                            pos0, frame=('body' if self.net_vel_is_body else 'world'),
                            method='trapezoid', device='cpu'
                        ).double()
                        self._last_ego_pos = np.zeros(3, dtype=float)
                        self.init_state.update({
                            "pos":[0,0,0],
                            "rot":[msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                            "vel":[0,0,0],
                            "stamp": stamp
                        })
                        # 네트/GNSS 모듈 초기 상태 전달(선택)
                        try:
                            if hasattr(self.airio, "set_init_state"): self.airio.set_init_state(self.init_state)
                            if hasattr(self.corrector, "set_init_state"): self.corrector.set_init_state(self.init_state)
                        except Exception as e:
                            self.get_logger().warn(f"set_init_state hook failed: {e}")

                        self.initialized = True
                        self.get_logger().info("Initialized from first IMU sample (local odom).")
                    except Exception as e:
                        self.get_logger().error(f"IMU-based init failed: {e}")
                        return

        imu_in = ImuData(
            wx=msg.angular_velocity.x, wy=msg.angular_velocity.y, wz=msg.angular_velocity.z,
            ax=msg.linear_acceleration.x, ay=msg.linear_acceleration.y, az=msg.linear_acceleration.z,
            qx=msg.orientation.x, qy=msg.orientation.y, qz=msg.orientation.z, qw=msg.orientation.w,
            stamp=stamp
        )
        with self.sample_lock:
            self.corrector.add_sample(imu_in)
            self.airio.add_sample(imu_in)

        gx, gy, gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gyro_norm = float(np.linalg.norm([gx, gy, gz]))
        acc_norm  = float(np.linalg.norm([ax, ay, az]) - self.gravity)
        self._imu_hist.append((stamp, gyro_norm, abs(acc_norm)))

    # === 타이머: 고주기 적분 + 퍼블리시 ===
    def on_timer(self):
        if not self.initialized:
            return

        t0 = time.time()
        imu_out = self.corrector.correct_latest()
        if imu_out is None:
            return

        if self.last_integrated_stamp is not None and imu_out.stamp <= self.last_integrated_stamp + 1e-12:
            return
        if self.last_integrated_stamp is None:
            self.last_integrated_stamp = imu_out.stamp
            return

        dt = max(1e-6, imu_out.stamp - self.last_integrated_stamp)
        if dt > self.max_dt:
            self.get_logger().warn(f"Abnormal dt={dt:.3f}s skipped.")
            self.last_integrated_stamp = imu_out.stamp
            return
        self.last_integrated_stamp = imu_out.stamp

        # IMU 적분 → pose, vel, rot
        try:
            dev  = self.pp_dev
            dt_t = torch.tensor([[[dt]]], dtype=torch.float64, device=dev)
            gyr  = torch.tensor([[[imu_out.wx, imu_out.wy, imu_out.wz]]], dtype=torch.float64, device=dev)
            acc  = torch.tensor([[[imu_out.ax, imu_out.ay, imu_out.az]]], dtype=torch.float64, device=dev)
            state = self.pp_integrator(init_state=None, dt=dt_t, gyro=gyr, acc=acc, rot=None)

            cur_pos = state['pos'][..., -1, :].detach().cpu().numpy().ravel()
            cur_vel = state['vel'][..., -1, :].detach().cpu().numpy().ravel()
            cur_rot = state['rot'][..., -1, :].detach().cpu().numpy().ravel()
            nrm = np.linalg.norm(cur_rot); 
            if nrm > 1e-9: cur_rot = (cur_rot / nrm).astype(float)
        except Exception as e:
            self.get_logger().warn(f"IMUPreintegrator step failed: {e}")
            return

        # 정지 판정 → 속도 0
        stationary = self._is_stationary(imu_out.stamp)
        if stationary:
            net_vel = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            net_vel = np.asarray(self.airio.predict_velocity(cur_rot), dtype=float)

        if np.linalg.norm(net_vel) * 1000.0 < self.deadband_ms:
            net_vel[:] = 0.0

        try:
            if stationary or np.allclose(net_vel, 0.0, atol=1e-9):
                ego_pos = self._last_ego_pos.copy()
            else:
                ego_pos = self.vel_integ.step(dt, net_vel, orient=cur_rot).detach().cpu().numpy()
                self._last_ego_pos = ego_pos.copy()
        except Exception as e:
            self.get_logger().warn(f"Velocity integration failed: {e}")
            return

        # 보간용 버퍼 저장
        self.airio_pose_buf.append((
            self.last_integrated_stamp,
            np.asarray(ego_pos, dtype=float),
            np.asarray(cur_rot, dtype=float),
            np.asarray(net_vel, dtype=float)
        ))

        # 최종 오돔 퍼블리시 (항상 로컬 odom 프레임)
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id  = self.base_frame
        odom.pose.pose.position.x = float(ego_pos[0])
        odom.pose.pose.position.y = float(ego_pos[1])
        odom.pose.pose.position.z = float(ego_pos[2])
        odom.pose.pose.orientation.x = float(cur_rot[0])
        odom.pose.pose.orientation.y = float(cur_rot[1])
        odom.pose.pose.orientation.z = float(cur_rot[2])
        odom.pose.pose.orientation.w = float(cur_rot[3])
        odom.twist.twist.linear.x = float(net_vel[0])
        odom.twist.twist.linear.y = float(net_vel[1])
        odom.twist.twist.linear.z = float(net_vel[2])
        self.odom_pub.publish(odom)

        # (옵션) TF broadcast
        if self.pub_tf and self.tf_broadcaster is not None:
            now = self.get_clock().now().to_msg()

            tf_ob = TransformStamped()
            tf_ob.header.stamp = now
            tf_ob.header.frame_id = self.odom_frame
            tf_ob.child_frame_id  = self.base_frame
            tf_ob.transform.translation.x = float(ego_pos[0])
            tf_ob.transform.translation.y = float(ego_pos[1])
            tf_ob.transform.translation.z = float(ego_pos[2])
            tf_ob.transform.rotation.x = float(cur_rot[0])
            tf_ob.transform.rotation.y = float(cur_rot[1])
            tf_ob.transform.rotation.z = float(cur_rot[2])
            tf_ob.transform.rotation.w = float(cur_rot[3])

            tmo, qmo = self.T_map_odom
            tf_mo = TransformStamped()
            tf_mo.header.stamp = now
            tf_mo.header.frame_id = self.world_frame
            tf_mo.child_frame_id  = self.odom_frame
            tf_mo.transform.translation.x = float(tmo[0])
            tf_mo.transform.translation.y = float(tmo[1])
            tf_mo.transform.translation.z = float(tmo[2])
            tf_mo.transform.rotation.x = float(qmo[0])
            tf_mo.transform.rotation.y = float(qmo[1])
            tf_mo.transform.rotation.z = float(qmo[2])
            tf_mo.transform.rotation.w = float(qmo[3])

            self.tf_broadcaster.sendTransform(tf_mo)
            self.tf_broadcaster.sendTransform(tf_ob)

        # 타이밍 기록(옵션)
        total_t = time.time() - t0
        if self.TL_mode and total_t <= 0.1:
            self.total_t_deque.append(total_t)

    def save_timings(self):
        timings = {
            "Total": list(self.total_t_deque),
        }
        outdir = os.path.dirname(self.TL_out_path) or "."
        prefix = os.path.splitext(os.path.basename(self.TL_out_path))[0] or "timings"
        os.makedirs(outdir, exist_ok=True)
        for name, values in timings.items():
            plt.figure(figsize=(8, 4))
            ms = [v * 1000 for v in values]
            plt.plot(ms, marker='o', markersize=2, linewidth=0.7, label=name)
            if values:
                avg = sum(values) / len(values) * 1000.0
                plt.axhline(avg, linestyle='--', label=f"avg={avg:.3f}ms")
                plt.axhline(min(values) * 1000.0, linestyle=':', label=f"min={min(values)*1000:.3f}ms")
                plt.axhline(max(values) * 1000.0, linestyle=':', label=f"max={max(values)*1000:.3f}ms")
            plt.ylabel("millisecond"); plt.xlabel("#"); plt.title(name)
            plt.legend(loc="upper right"); plt.grid(True)
            safe = name.replace(" ", "_").replace("/", "_")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{prefix}_{safe}.png"), dpi=130); plt.close()


def main(args=None):
    rclpy.init(args=args)
    node = AirIoImuOdomNode()
    executor = MultiThreadedExecutor(num_threads=2)  # imu와 timer 병렬 처리
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        if node.TL_mode:
            node.save_timings()
    finally:
        executor.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
