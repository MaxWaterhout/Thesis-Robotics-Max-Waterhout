#!/bin/bash
WORKING_DIR="$HOME/Documents/ros_workspaces/aruco_marker_ws"

gnome-terminal --tab  -- python /home/max/Documents/ros_workspaces/aruco_marker_ws/src/pose_pkg/scripts/inference_predict.py &
gnome-terminal --tab  -- python /home/max/Documents/ros_workspaces/aruco_marker_ws/src/pose_pkg/scripts/prediction_service.py 

#gnome-terminal --tab -- python /home/max/Documents/ros_workspaces/zed_ws/src/pose_estimator/scripts/predict_service.py --weights "/home/max/Documents/GitHub/thesis/weights/queen/phi_0_linemod_best_ADD-S.h5" --object "queen" &
#gnome-terminal --tab -- python /home/max/Documents/ros_workspaces/zed_ws/src/pose_estimator/scripts/predict_service.py --weights "/home/max/Documents/GitHub/thesis/weights/rook/phi_0_linemod_best_ADD-S.h5" --object "rook" &
#gnome-terminal --tab -- python /home/max/Documents/ros_workspaces/zed_ws/src/pose_estimator/scripts/predict_service.py --weights "/home/max/Documents/GitHub/thesis/weights/king/phi_0_linemod_best_ADD-S.h5" --object "king" &
#gnome-terminal --tab -- python /home/max/Documents/ros_workspaces/zed_ws/src/pose_estimator/scripts/predict_service.py --weights "/home/max/Documents/GitHub/thesis/weights/bishop/phi_0_linemod_best_ADD-S.h5" --object "bishop" &
#gnome-terminal --tab -- python /home/max/Documents/ros_workspaces/zed_ws/src/pose_estimator/scripts/predict_service.py --weights "/home/max/Documents/GitHub/thesis/weights/pawn/phi_0_linemod_best_ADD-S.h5" --object "pawn" &
#gnome-terminal --tab -- bash -c "cd ~/Documents/ros_workspaces/zed_ws; echo rostopic pub --once /yolo_prediction_trigger std_msgs/Empty "{}"; exec bash"

