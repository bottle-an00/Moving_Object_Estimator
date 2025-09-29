#include "moving_object_estimator/sync_node.hpp"

SyncNode::SyncNode() : Node("sync_node")
{
    rclcpp::QoS scan_qos = rclcpp::SensorDataQoS();
    rclcpp::QoS odom_qos =(50);

    scan_sub_.subscribe(this,"/scan", scan_qos.get_rmw_qos_profile());
    odom_sub_.subscribe(this,"/odom", odom_qos.get_rmw_qos_profile());

    double slop_sec = this->declare_parameter<double>("sync_slop_sec", 0.025); // 25ms <- max(scan: 40hz/2 , odom: 20hz/2) 
    sync_ = std::make_shared<message_filters::Synchronizer<ApproxPolicy>>(ApproxPolicy(20), scan_sub_, odom_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop_sec)); 
    sync_->registerCallback(std::bind(&SyncNode::scan_odom_callback, this, std::placeholders::_1,std::placeholders::_2));

    only_scan_sub_ = this->create_subscription<LaserScan>
               ("/scan",rclcpp::SensorDataQoS(),std::bind(&SyncNode::scan_callback, this, std::placeholders::_1));
    only_odom_sub_ = this->create_subscription<Odometry>
               ("/odom",rclcpp::QoS(50),std::bind(&SyncNode::odom_callback, this, std::placeholders::_1));
}

void SyncNode::show_result()
{
    std::vector<int> x(vec_dt_->size());
    std::iota(x.begin(), x.end(), 0);  

    std::vector<double> y = *vec_dt_;

    plt::figure();
    plt::named_plot("Δt values", x, y, "b-");  
    plt::xlabel("Message index");
    plt::ylabel("Δt (scan - odom) [s]");
    plt::legend();

    plt::ylim(min_dt_ - 1.0, max_dt_ + 1.0);

    std::stringstream ss;
    RCLCPP_INFO(this->get_logger(), "\n\033[32m [RESULT] [0m \033[32m mean dt\033[0m %.5f \033[32m max dt\033[0m %.5f \033[32m min dt \033[0m %.5f scan msg count : %d, odom msg count : %d, bind msg count %d ",
                sum_dt_/matched_msgs_count_ , max_dt_, min_dt_,scan_msg_count_,odom_msg_count_,matched_msgs_count_);
    plt::show();
}

void SyncNode::scan_odom_callback(const LaserScan::ConstSharedPtr& scan,
                                  const Odometry::ConstSharedPtr& odom)
{
    matched_msgs_count_++;

    dt_ = abs(rclcpp::Time(scan->header.stamp).seconds() - rclcpp::Time(odom->header.stamp).seconds());
    sum_dt_ += dt_; 
    min_dt_ = std::min(min_dt_,dt_);
    max_dt_ = std::max(max_dt_,dt_);

    vec_dt_->push_back(dt_);

    RCLCPP_INFO(this->get_logger(), "\033[32mSynced pair[sec]: scan\033[0m %.3f \033[32m odom\033[0m %.3f \n\033[32m dt\033[0m %.5f\033[32m mean dt\033[0m %.5f \033[32m max dt\033[0m %.5f \033[32m min dt \033[0m %.5f",
                rclcpp::Time(scan->header.stamp).seconds(),
                rclcpp::Time(odom->header.stamp).seconds(),
                dt_, sum_dt_/matched_msgs_count_ , max_dt_, min_dt_);
    
}

void SyncNode::scan_callback(const LaserScan::ConstSharedPtr& scan)
{
    scan_msg_count_++;
}

void SyncNode::odom_callback(const Odometry::ConstSharedPtr& odom)
{
    odom_msg_count_++;
}