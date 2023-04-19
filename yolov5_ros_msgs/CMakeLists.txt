cmake_minimum_required(VERSION 3.0.2)
project(yolov5_ros_msgs)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_message_files(
        DIRECTORY msg
        FILES
        BoundingBox.msg
        BoundingBoxes.msg
)

add_service_files(
 FILES
 meter_response.srv
 counter_response_crop.srv
 gauge_response_crop.srv
 ssdisplay_response_crop.srv
)

generate_messages(
        DEPENDENCIES
        std_msgs
)

catkin_package(

)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
