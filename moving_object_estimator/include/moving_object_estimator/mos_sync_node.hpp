#pragma once

#include <algorithm>
#include <climits>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "matplotlibcpp.h"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "moving_object_estimator/map_manager.hpp"
#include "moving_object_estimator/dynamic_obstacle_detector.hpp"

using nav_msgs::msg::Odometry;
using sensor_msgs::msg::PointCloud2;
using ApproxPolicy = message_filters::sync_policies::ApproximateTime<Odometry, Odometry>;

namespace plt = matplotlibcpp;

class SyncNode : public rclcpp::Node{
    public: 
        SyncNode();

        void show_result();
    private:
        int obstacle_msg_count_ = 0;
        int odom_msg_count_ = 0;
        int matched_msgs_count_ = 0;

        std::vector<double>* vec_dt_ = new std::vector<double>();
    
        double dt_{}, min_dt_{std::numeric_limits<double>::max()},max_dt_{},sum_dt_{};

        void obstacle_odom_callback(const Odometry::ConstSharedPtr& obstacle,
                                const Odometry::ConstSharedPtr& odom);
        
        void obstacle_callback(const Odometry::ConstSharedPtr& obstacle);
        void odom_callback(const Odometry::ConstSharedPtr& odom);
        void wall_callback(const PointCloud2::ConstSharedPtr& msg);
        
        message_filters::Subscriber<Odometry> obstacle_sub_;
        message_filters::Subscriber<Odometry> odom_sub_;
        std::shared_ptr<message_filters::Synchronizer<ApproxPolicy>> sync_;
        
        rclcpp::Subscription<Odometry>::SharedPtr only_obstacle_sub_;
        rclcpp::Subscription<Odometry>::SharedPtr only_odom_sub_;
        rclcpp::Subscription<PointCloud2>::SharedPtr wall_sub_;
        
        rclcpp::Publisher<Odometry>::SharedPtr dynamic_obs_pub_;
        rclcpp::Publisher<PointCloud2>::SharedPtr accumulated_wall_cloud_pub_;

        MapManager map_manager_;
        DynamicObjectDetector detector_;

        std::string wall_topic_{"/wall_points"};
        int wall_deque_size_{10};
        double voxel_leaf_{0.03};
        int knn_k_{5};
        double wall_dist_thresh_{0.05}; // [m]
};