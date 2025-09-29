cmake_minimum_required(VERSION 3.8)
project(moving_object_estimator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)


include_directories(
  include
)

add_executable(sync_node
  src/sync_node.cpp
  src/main.cpp
)

ament_target_dependencies(sync_node
  rclcpp
  sensor_msgs
  nav_msgs
  message_filters
)

target_include_directories(sync_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(sync_node ${Python3_LIBRARIES})

install(TARGETS sync_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()