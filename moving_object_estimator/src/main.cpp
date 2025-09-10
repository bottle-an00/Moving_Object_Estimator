#include "moving_object_estimator/sync_node.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc,argv);
    auto node = std::make_shared<SyncNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    node->show_result();
    return 0;
}