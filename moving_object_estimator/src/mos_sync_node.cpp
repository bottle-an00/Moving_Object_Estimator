#include "moving_object_estimator/mos_sync_node.hpp"

SyncNode::SyncNode() : Node("sync_node"),
  map_manager_(wall_deque_size_),
  detector_(knn_k_, wall_dist_thresh_, voxel_leaf_)
{
    // 파라미터 선언/로드 (생략 가능)
    this->declare_parameter("wall_topic", wall_topic_);
    this->declare_parameter("wall_deque_size", wall_deque_size_);
    this->declare_parameter("voxel_leaf", voxel_leaf_);
    this->declare_parameter("knn_k", knn_k_);
    this->declare_parameter("wall_dist_thresh", wall_dist_thresh_);

    this->get_parameter("wall_topic", wall_topic_);
    this->get_parameter("wall_deque_size", wall_deque_size_);
    this->get_parameter("voxel_leaf", voxel_leaf_);
    this->get_parameter("knn_k", knn_k_);
    this->get_parameter("wall_dist_thresh", wall_dist_thresh_);

    map_manager_.set_max_deque_size(wall_deque_size_);
    detector_.set_voxel_leaf(voxel_leaf_);
    detector_.set_knn_k(knn_k_);
    detector_.set_wall_dist_thresh(wall_dist_thresh_);

    rclcpp::QoS obstacle_qos = rclcpp::SensorDataQoS();
    rclcpp::QoS odom_qos =(50);

    obstacle_sub_.subscribe(this,"/opponent_odom", obstacle_qos.get_rmw_qos_profile());
    odom_sub_.subscribe(this,"/odom", odom_qos.get_rmw_qos_profile());

    double slop_sec = this->declare_parameter<double>("sync_slop_sec", 0.025); // 25ms <- max(obstacle: 40hz/2 , odom: 20hz/2)
    sync_ = std::make_shared<message_filters::Synchronizer<ApproxPolicy>>(ApproxPolicy(20), obstacle_sub_, odom_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop_sec));
    sync_->registerCallback(std::bind(&SyncNode::obstacle_odom_callback, this, std::placeholders::_1,std::placeholders::_2));

    only_obstacle_sub_ = this->create_subscription<Odometry>
               ("/opponent_odom",rclcpp::SensorDataQoS(),[this](Odometry::ConstSharedPtr msg) {this->obstacle_callback(msg);});
    only_odom_sub_ = this->create_subscription<Odometry>
               ("/odom",rclcpp::QoS(50),[this](Odometry::ConstSharedPtr msg) {this->odom_callback(msg);});

    wall_sub_ = this->create_subscription<PointCloud2>
               ("/wall_points",rclcpp::QoS(50),[this](PointCloud2::ConstSharedPtr msg) {this->wall_callback(msg);});

    accumulated_wall_cloud_pub_ = this->create_publisher<PointCloud2>("/accumulated_wall_map", rclcpp::QoS(50));
    dynamic_obs_pub_ = this->create_publisher<Odometry>("/dynamic_obstacle", rclcpp::QoS(50));
}

static inline tf2::Transform make_T_w_b(const geometry_msgs::msg::Pose& pose)
{
  tf2::Quaternion q;
  tf2::fromMsg(pose.orientation, q);
  tf2::Vector3 t(pose.position.x, pose.position.y, pose.position.z);
  return tf2::Transform(q, t);
}

void SyncNode::obstacle_odom_callback(const Odometry::ConstSharedPtr& obstacle,
                                  const Odometry::ConstSharedPtr& odom)
{
    matched_msgs_count_++;

    dt_ = abs(rclcpp::Time(obstacle->header.stamp).seconds() - rclcpp::Time(odom->header.stamp).seconds());
    sum_dt_ += dt_;
    min_dt_ = std::min(min_dt_,dt_);
    max_dt_ = std::max(max_dt_,dt_);

    vec_dt_->push_back(dt_);

    RCLCPP_INFO(this->get_logger(), "\033[32mSynced pair[sec]: obstacle\033[0m %.3f \033[32m odom\033[0m %.3f \n\033[32m dt\033[0m %.5f\033[32m mean dt\033[0m %.5f \033[32m max dt\033[0m %.5f \033[32m min dt \033[0m %.5f",
                rclcpp::Time(obstacle->header.stamp).seconds(),
                rclcpp::Time(odom->header.stamp).seconds(),
                dt_, sum_dt_/matched_msgs_count_ , max_dt_, min_dt_);

    // 프레임 체크
    if (obstacle->header.frame_id.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Obstacle Odometry frame_id empty; expected 'map' or same as odom.");
    } else if (odom->header.frame_id != obstacle->header.frame_id) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "Frames mismatch: obstacle '%s' vs odom header '%s'. Proceeding.",
            obstacle->header.frame_id.c_str(), odom->header.frame_id.c_str());
    }

    // //map 생성 및 detector rebuild
    // auto snap = map_manager_.snapshot();
    // detector_.rebuild(snap);

    // //만약 인식한 장애물이 wall이라면 pass
    // if(detector_.isWall(obstacle->pose.pose.position)){
    //     std::cout << "It's wall!!!!\n";
    //     return;
    // }

    // dynamic_obs_pub_->publish(*obstacle);

    /*만약, local 좌표계에서 publish를 원한다면 좌표 변환 수행*/
    // const tf2::Transform T_w_b = make_T_w_b(odom->pose.pose);
    // const tf2::Transform T_b_w = T_w_b.inverse();

    // const auto& pw = obstacle->pose.pose.position;
    // const tf2::Vector3 p_w(pw.x, pw.y, pw.z);
    // const tf2::Vector3 p_b = T_b_w * p_w;

    // const auto& q_w = obstacle->pose.pose.orientation;
    // tf2::Quaternion q_obs;
    // tf2::fromMsg(q_w, q_obs);

    // tf2::Quaternion q_map_base;
    // tf2::fromMsg(odom->pose.pose.orientation, q_map_base);
    // tf2::Quaternion q_base_map = q_map_base.inverse();

    // tf2::Quaternion q_obs_base = q_base_map * q_obs;
}

void SyncNode::obstacle_callback(const Odometry::ConstSharedPtr& obstacle)
{
    obstacle_msg_count_++;

    //map 생성 및 detector rebuild
    auto snap = map_manager_.snapshot();
    auto snap_obj = map_manager_.snapshot_obj();
    detector_.rebuild(snap);
    detector_.rebuild_obj(snap_obj);

    //만약 인식한 장애물이 wall이라면 pass
    if(detector_.isWall(obstacle->pose.pose.position)){return;}

    //만약 인식한 장애물이 static이라면 pass
    if(detector_.isStatic(obstacle->pose.pose.position)){
        map_manager_.addObj(obstacle->pose.pose.position);
        // //만약 기존의 static vector에 저장되었다면 pass
        detector_.rebuild_obj(detector_.get_static_vec());
        if(!detector_.isStatic(obstacle->pose.pose.position)){detector_.addStatic(obstacle->pose.pose.position);}
        return;
    }
    dynamic_obs_pub_->publish(*obstacle);
    map_manager_.addObj(obstacle->pose.pose.position);
}

void SyncNode::odom_callback(const Odometry::ConstSharedPtr& odom)
{
    odom_msg_count_++;
}
void SyncNode::wall_callback(const PointCloud2::ConstSharedPtr& msg)
{
    map_manager_.addCloud(msg);

    auto snaps = map_manager_.snapshot();
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    for(auto cloud : snaps){
        *map_cloud += *cloud;
    }

    sensor_msgs::msg::PointCloud2 ros_cloud;
    pcl::toROSMsg(*map_cloud, ros_cloud);
    ros_cloud.header.frame_id = "map";     // frame 지정 필수
    ros_cloud.header.stamp = msg->header.stamp;
    accumulated_wall_cloud_pub_->publish(ros_cloud);
}

void SyncNode::show_result()
{
    std::vector<int> x(vec_dt_->size());
    std::iota(x.begin(), x.end(), 0);

    std::vector<double> y = *vec_dt_;

    plt::figure();
    plt::named_plot("dt values", x, y, "b-");
    plt::xlabel("Message index");
    plt::ylabel("dt (obstacle - odom) [s]");
    plt::legend();

    plt::ylim(min_dt_ - 1.0, max_dt_ + 1.0);

    std::stringstream ss;
    RCLCPP_INFO(this->get_logger(), "\n\033[32m [RESULT] [0m \033[32m mean dt\033[0m %.5f \033[32m max dt\033[0m %.5f \033[32m min dt \033[0m %.5f obstacle msg count : %d, odom msg count : %d, bind msg count %d ",
                sum_dt_/matched_msgs_count_ , max_dt_, min_dt_,obstacle_msg_count_,odom_msg_count_,matched_msgs_count_);
    plt::show();
}
