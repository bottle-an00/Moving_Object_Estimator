#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

class ObstacleDetector : public rclcpp::Node {
public:
  ObstacleDetector()
  : Node("obstacle_detector"),
    kalman_initialized_(false),
    has_prev_position_(false),
    prev_heading_(0.0)
  {
    // 파라미터 선언 및 취득
    this->declare_parameter<double>("dbscan_eps", 0.3);
    this->declare_parameter<int>("dbscan_min_points", 3);
    this->declare_parameter<bool>("use_weighted_median", false);
    this->declare_parameter<double>("kalman_process_noise", 0.1);
    this->declare_parameter<double>("kalman_measurement_noise", 0.1);
    this->declare_parameter<bool>("use_kalman_filter", true);
    this->declare_parameter<double>("obstacle_timeout", 1.0);
    this->declare_parameter<int>("min_candidates_to_process", 3);

    this->get_parameter("dbscan_eps", dbscan_eps_);
    this->get_parameter("dbscan_min_points", dbscan_min_points_);
    this->get_parameter("use_weighted_median", use_weighted_median_);
    this->get_parameter("kalman_process_noise", kalman_process_noise_);
    this->get_parameter("kalman_measurement_noise", kalman_measurement_noise_);
    this->get_parameter("use_kalman_filter", use_kalman_filter_);
    this->get_parameter("obstacle_timeout", obstacle_timeout_);
    this->get_parameter("min_candidates_to_process", min_candidates_to_process_);

    candidate_points_.reserve(static_cast<size_t>(min_candidates_to_process_) * 2);

    // 퍼블리셔 생성
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/opponent_odom", 20);
    obstacle_detected_pub_ = this->create_publisher<std_msgs::msg::Bool>("/obstacle_detected", 20);

    // 구독자 생성 (후보점 수신)
    candidate_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/obstacle_candidates", 20,
      [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
        candidate_points_.push_back(*msg);
        if (candidate_points_.size() >= static_cast<size_t>(min_candidates_to_process_)) {
          processAndUpdateMeasurement();
        }
      });

    // opponent_odom 메시지를 빠르게 퍼블리시하기 위한 타이머 (예: 20Hz)
    odom_timer_ = this->create_wall_timer(50ms, [this]() {
      rclcpp::Time now = this->now();
      if (kalman_initialized_) {
        // 예측 단계: dt 동안 상태 예측
        double dt = (now - last_kf_time_).seconds();
        kf_state_[0] += kf_state_[2] * dt;
        kf_state_[1] += kf_state_[3] * dt;
        // 공분산 업데이트 (필요한 경우)
        kf_P_[0][0] += kalman_process_noise_;
        kf_P_[1][1] += kalman_process_noise_;
        kf_P_[2][2] += kalman_process_noise_;
        kf_P_[3][3] += kalman_process_noise_;
        last_kf_time_ = now;
        publishOdomWithKalmanState(now);
      }
    });
  }

private:
  // 후보 점들의 대표 timestamp를 중앙값 방식으로 산출
  rclcpp::Time computeAverageTimestamp(const std::vector<size_t>& indices) {
    std::vector<uint64_t> times;
    times.reserve(indices.size());
    for (size_t idx : indices) {
      const auto &stamp = candidate_points_[idx].header.stamp;
      uint64_t ns = static_cast<uint64_t>(stamp.sec) * 1000000000ull + stamp.nanosec;
      times.push_back(ns);
    }
    size_t n = times.size();
    auto mid = times.begin() + n / 2;
    std::nth_element(times.begin(), mid, times.end());
    uint64_t median_ns = *mid;
    builtin_interfaces::msg::Time rep_time;
    rep_time.sec = static_cast<int32_t>(median_ns / 1000000000ull);
    rep_time.nanosec = static_cast<uint32_t>(median_ns % 1000000000ull);
    return rclcpp::Time(rep_time);
  }

  // DBSCAN 클러스터링 (변경 없음)
  std::vector<std::vector<size_t>> performDBSCAN() {
    size_t N = candidate_points_.size();
    std::vector<int> cluster_ids(N, -1);  // -1: 미할당, -2: 노이즈

    auto distance = [this](size_t i, size_t j) -> double {
      double dx = candidate_points_[i].point.x - candidate_points_[j].point.x;
      double dy = candidate_points_[i].point.y - candidate_points_[j].point.y;
      return std::sqrt(dx * dx + dy * dy);
    };

    auto regionQuery = [this, N, &distance](size_t i) -> std::vector<size_t> {
      std::vector<size_t> neighbors;
      for (size_t j = 0; j < N; j++) {
        if (distance(i, j) <= dbscan_eps_) {
          neighbors.push_back(j);
        }
      }
      return neighbors;
    };

    int cluster_id = 0;
    for (size_t i = 0; i < N; i++) {
      if (cluster_ids[i] != -1)
        continue;
      auto neighbors = regionQuery(i);
      if (neighbors.size() < static_cast<size_t>(dbscan_min_points_)) {
        cluster_ids[i] = -2;
        continue;
      }
      cluster_ids[i] = cluster_id;
      std::vector<size_t> seed_set = std::move(neighbors);
      for (size_t idx = 0; idx < seed_set.size(); idx++) {
        size_t j = seed_set[idx];
        if (cluster_ids[j] == -2)
          cluster_ids[j] = cluster_id;
        if (cluster_ids[j] != -1)
          continue;
        cluster_ids[j] = cluster_id;
        auto neighbors_j = regionQuery(j);
        if (neighbors_j.size() >= static_cast<size_t>(dbscan_min_points_)) {
          seed_set.insert(seed_set.end(), neighbors_j.begin(), neighbors_j.end());
        }
      }
      cluster_id++;
    }

    std::vector<std::vector<size_t>> clusters(cluster_id);
    for (size_t i = 0; i < N; i++) {
      if (cluster_ids[i] >= 0)
        clusters[cluster_ids[i]].push_back(i);
    }
    return clusters;
  }

  // 대표점 산출 (가중 평균 혹은 가중 중앙값)
  std::pair<double, double> computeRepresentativePoint(const std::vector<size_t>& cluster) {
    double sum_x = 0.0, sum_y = 0.0;
    for (size_t idx : cluster) {
      sum_x += candidate_points_[idx].point.x;
      sum_y += candidate_points_[idx].point.y;
    }
    double center_x = sum_x / cluster.size();
    double center_y = sum_y / cluster.size();
    const double epsilon = 1e-3;
    double rep_x = 0.0, rep_y = 0.0;

    if (!use_weighted_median_) {
      double weighted_sum_x = 0.0, weighted_sum_y = 0.0, total_weight = 0.0;
      for (size_t idx : cluster) {
        double dx = candidate_points_[idx].point.x - center_x;
        double dy = candidate_points_[idx].point.y - center_y;
        double d = std::sqrt(dx * dx + dy * dy);
        double weight = 1.0 / (d + epsilon);
        weighted_sum_x += candidate_points_[idx].point.x * weight;
        weighted_sum_y += candidate_points_[idx].point.y * weight;
        total_weight += weight;
      }
      rep_x = weighted_sum_x / total_weight;
      rep_y = weighted_sum_y / total_weight;
    } else {
      struct WeightedVal {
        double val;
        double weight;
      };
      std::vector<WeightedVal> wx, wy;
      double total_weight = 0.0;
      for (size_t idx : cluster) {
        double dx = candidate_points_[idx].point.x - center_x;
        double dy = candidate_points_[idx].point.y - center_y;
        double d = std::sqrt(dx * dx + dy * dy);
        double weight = 1.0 / (d + epsilon);
        wx.push_back({candidate_points_[idx].point.x, weight});
        wy.push_back({candidate_points_[idx].point.y, weight});
        total_weight += weight;
      }
      auto cmp = [](const WeightedVal &a, const WeightedVal &b) { return a.val < b.val; };
      std::sort(wx.begin(), wx.end(), cmp);
      std::sort(wy.begin(), wy.end(), cmp);
      double cum = 0.0, median_x = wx.front().val;
      for (const auto &w : wx) {
        cum += w.weight;
        if (cum >= total_weight / 2.0) {
          median_x = w.val;
          break;
        }
      }
      cum = 0.0;
      double median_y = wy.front().val;
      for (const auto &w : wy) {
        cum += w.weight;
        if (cum >= total_weight / 2.0) {
          median_y = w.val;
          break;
        }
      }
      rep_x = median_x;
      rep_y = median_y;
    }
    return {rep_x, rep_y};
  }

  // 후보점 처리 및 측정 업데이트 (오직 칼만 필터의 측정 보정만 수행)
  void processAndUpdateMeasurement() {
    rclcpp::Time now = this->now();
    double meas_x = 0.0, meas_y = 0.0;
    bool measurement_available = false;
    rclcpp::Time candidate_stamp = now;

    if (!candidate_points_.empty()) {
      auto clusters = performDBSCAN();
      int best_cluster = -1;
      size_t best_cluster_size = 0;
      for (size_t i = 0; i < clusters.size(); i++) {
        if (clusters[i].size() > best_cluster_size) {
          best_cluster_size = clusters[i].size();
          best_cluster = static_cast<int>(i);
        }
      }
      if (best_cluster != -1 && best_cluster_size > 0) {
        measurement_available = true;
        last_measurement_time_ = now;
        candidate_stamp = computeAverageTimestamp(clusters[best_cluster]);
        std::tie(meas_x, meas_y) = computeRepresentativePoint(clusters[best_cluster]);
      }
      candidate_points_.clear();
    }
    
    // 칼만 필터 사용 시 측정 보정만 업데이트 (예측은 타이머에서 수행)
    if (use_kalman_filter_) {
      if (!kalman_initialized_ && measurement_available) {
        // 첫 측정 시 초기화
        kf_state_[0] = meas_x;
        kf_state_[1] = meas_y;
        kf_state_[2] = 0.0;
        kf_state_[3] = 0.0;
        // 초기 공분산 설정
        kf_P_[0][0] = 1.0; kf_P_[0][1] = 0.0; kf_P_[0][2] = 0.0; kf_P_[0][3] = 0.0;
        kf_P_[1][0] = 0.0; kf_P_[1][1] = 1.0; kf_P_[1][2] = 0.0; kf_P_[1][3] = 0.0;
        kf_P_[2][0] = 0.0; kf_P_[2][1] = 0.0; kf_P_[2][2] = 1.0; kf_P_[2][3] = 0.0;
        kf_P_[3][0] = 0.0; kf_P_[3][1] = 0.0; kf_P_[3][2] = 0.0; kf_P_[3][3] = 1.0;
        kalman_initialized_ = true;
        last_kf_time_ = now;
      }
      
      if (measurement_available && kalman_initialized_) {
        double P00 = kf_P_[0][0];
        double P11 = kf_P_[1][1];
        double S0 = P00 + kalman_measurement_noise_;
        double S1 = P11 + kalman_measurement_noise_;
        double K0 = P00 / S0;
        double K1 = P11 / S1;
        double y0 = meas_x - kf_state_[0];
        double y1 = meas_y - kf_state_[1];
        kf_state_[0] += K0 * y0;
        kf_state_[1] += K1 * y1;
        kf_P_[0][0] = (1 - K0) * P00;
        kf_P_[1][1] = (1 - K1) * P11;
        last_kf_time_ = now;
      }
    } else {
      // 칼만 필터 미사용 시 단순 업데이트
      if (measurement_available) {
        kf_state_[0] = meas_x;
        kf_state_[1] = meas_y;
        kf_state_[2] = 0.0;
        kf_state_[3] = 0.0;
        kalman_initialized_ = true;
        last_kf_time_ = now;
      }
    }
    
    // 측정 업데이트 시 obstacle_detected 플래그 갱신
    std_msgs::msg::Bool detected_msg;
    detected_msg.data = (kalman_initialized_ && (now - last_measurement_time_).seconds() < obstacle_timeout_);
    obstacle_detected_pub_->publish(detected_msg);
  }

  // 칼만 필터 상태를 기반으로 odom 메시지 퍼블리시 (헤딩 스무딩 적용)
  void publishOdomWithKalmanState(const rclcpp::Time & stamp) {
    double heading_pos = prev_heading_;
    if (has_prev_position_) {
      double dx = kf_state_[0] - prev_x_;
      double dy = kf_state_[1] - prev_y_;
      if (std::sqrt(dx * dx + dy * dy) > 1e-3) {
        heading_pos = std::atan2(dy, dx);
      }
    }
    double speed = std::sqrt(kf_state_[2] * kf_state_[2] + kf_state_[3] * kf_state_[3]);
    double heading_vel = (speed > 1e-3) ? std::atan2(kf_state_[3], kf_state_[2]) : heading_pos;
    double raw_heading = (heading_pos + heading_vel) / 2.0;

    // 지수 이동평균을 통한 헤딩 스무딩 (wrap-around 보정 포함)
    double alpha = 0.5;
    double delta = raw_heading - prev_heading_;
    while (delta > M_PI)  delta -= 2.0 * M_PI;
    while (delta < -M_PI) delta += 2.0 * M_PI;
    double smoothed_heading = prev_heading_ + alpha * delta;

    prev_heading_ = smoothed_heading;
    prev_x_ = kf_state_[0];
    prev_y_ = kf_state_[1];
    has_prev_position_ = true;

    double sin_yaw = std::sin(smoothed_heading * 0.5);
    double cos_yaw = std::cos(smoothed_heading * 0.5);

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = "map";
    odom_msg.pose.pose.position.x = kf_state_[0];
    odom_msg.pose.pose.position.y = kf_state_[1];
    odom_msg.pose.pose.position.z = 0.0;
    odom_msg.twist.twist.linear.x = kf_state_[2];
    odom_msg.twist.twist.linear.y = kf_state_[3];
    odom_msg.twist.twist.linear.z = 0.0;
    odom_msg.pose.pose.orientation.x = 0.0;
    odom_msg.pose.pose.orientation.y = 0.0;
    odom_msg.pose.pose.orientation.z = sin_yaw;
    odom_msg.pose.pose.orientation.w = cos_yaw;
    odom_msg.twist.twist.angular.x = 0.0;
    odom_msg.twist.twist.angular.y = 0.0;
    odom_msg.twist.twist.angular.z = 0.0;

    odom_pub_->publish(odom_msg);
  }

  // 멤버 변수들
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr candidate_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr obstacle_detected_pub_;
  rclcpp::TimerBase::SharedPtr odom_timer_;

  std::vector<geometry_msgs::msg::PointStamped> candidate_points_;

  double dbscan_eps_;
  int dbscan_min_points_;
  bool use_weighted_median_;
  int min_candidates_to_process_;

  bool kalman_initialized_;
  // 상태: [x, y, vx, vy]
  double kf_state_[4];
  // 공분산 행렬 (4x4)
  double kf_P_[4][4];
  double kalman_process_noise_;
  double kalman_measurement_noise_;
  rclcpp::Time last_kf_time_;

  bool use_kalman_filter_;

  // 이전 위치 및 헤딩 (orientation 계산용)
  double prev_x_;
  double prev_y_;
  double prev_heading_;
  bool has_prev_position_;

  rclcpp::Time last_measurement_time_;
  double obstacle_timeout_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleDetector>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
