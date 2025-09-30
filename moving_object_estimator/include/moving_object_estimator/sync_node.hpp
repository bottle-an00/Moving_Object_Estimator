#pragma once

#include<algorithm>
#include<climits>
#include<vector>

#include<rclcpp/rclcpp.hpp>

#include<sensor_msgs/msg/laser_scan.hpp>
#include<nav_msgs/msg/odometry.hpp>

#include<message_filters/subscriber.h>
#include<message_filters/synchronizer.h>
#include<message_filters/sync_policies/approximate_time.h>

#include "matplotlibcpp.h"

using sensor_msgs::msg::LaserScan;
using nav_msgs::msg::Odometry;
using ApproxPolicy = message_filters::sync_policies::ApproximateTime<LaserScan, Odometry>;

namespace plt = matplotlibcpp;

class SyncNode : public rclcpp::Node{
    public: 
        SyncNode();

        void show_result();
    private:
        int scan_msg_count_ = 0;
        int odom_msg_count_ = 0;
        int matched_msgs_count_ = 0;

        std::vector<double>* vec_dt_ = new std::vector<double>();
    
        double dt_{}, min_dt_{std::numeric_limits<double>::max()},max_dt_{},sum_dt_{};

        void scan_odom_callback(const LaserScan::ConstSharedPtr& scan,
                                const Odometry::ConstSharedPtr& odom);
        
        /*For checking dropped messages count*/
        void scan_callback(const LaserScan::ConstSharedPtr& scan);
        void odom_callback(const Odometry::ConstSharedPtr& odom);
        
        message_filters::Subscriber<LaserScan> scan_sub_;
        message_filters::Subscriber<Odometry> odom_sub_;
        std::shared_ptr<message_filters::Synchronizer<ApproxPolicy>> sync_;

        /*For checking dropped messages count*/
        rclcpp::Subscription<LaserScan>::SharedPtr only_scan_sub_;
        rclcpp::Subscription<Odometry>::SharedPtr only_odom_sub_;
};