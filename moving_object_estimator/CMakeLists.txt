cmake_minimum_required(VERSION 3.8)
project(moving_object_estimator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(message_filters REQUIRED)

find_package(Python3 COMPONENTS Development NumPy REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(PCL REQUIRED)                 
find_package(pcl_conversions REQUIRED) 

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_executable(mos_sync_node
  # src/sync_node.cpp
  src/mos_sync_node.cpp
  src/map_manager.cpp
  src/dynamic_obstacle_detector.cpp
  src/main.cpp
)

ament_target_dependencies(mos_sync_node
  rclcpp
  sensor_msgs
  nav_msgs
  message_filters
  tf2
  geometry_msgs
  tf2_geometry_msgs
  tf2_ros
  pcl_conversions
)

target_include_directories(mos_sync_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(mos_sync_node ${Python3_LIBRARIES})

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS mos_sync_node
  DESTINATION lib/${PROJECT_NAME}
)
ament_export_include_directories(include ${PCL_INCLUDE_DIRS})
ament_export_dependencies(PCL pcl_conversions)

ament_package()