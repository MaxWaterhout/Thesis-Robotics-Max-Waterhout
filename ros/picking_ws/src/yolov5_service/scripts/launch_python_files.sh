#!/bin/bash
WORKING_DIR="$HOME/Documents/ros_workspaces/zed_ws"

#gnome-terminal --tab -- python /home/max/Documents/ros_workspaces/zed_ws/src/pose_estimator/scripts/predict_service.py &
gnome-terminal --tab  -- python /home/max/Documents/ros_workspaces/zed_ws/src/yolov5_service/scripts/service_yolo.py &
gnome-terminal --tab -- python /home/max/Documents/ros_workspaces/zed_ws/src/yolov5_service/scripts/client_yolo.py &
#gnome-terminal --tab -- bash -c "cd ~/Documents/ros_workspaces/zed_ws; echo rostopic pub --once /yolo_prediction_trigger std_msgs/Empty "{}"; exec bash"

