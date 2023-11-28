import rospy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import pandas as pd
import time
from panda_msgs.srv import InputYoloPrediction, InputYoloPredictionResponse
from std_msgs.msg import Empty, String



class service_yolo:
    def __init__(self):
        self.model = self.load_model()
        self.bridge = CvBridge()
        self.service = rospy.Service("prediction_yolo", InputYoloPrediction, self.prediction)
        self.class_names = {0: 'rook',1: 'queen', 2: 'pawn', 3: 'king', 4: 'horse', 5: 'bishop'}

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/max/Documents/yolov5/weights/best.engine')  # local model
        model.cuda()
        rospy.loginfo('loaded model, waiting for a request call..')
        return model 
    
    def prediction(self,request):
        start = time.time()
        rospy.loginfo('Receiving request ')
        img = self.bridge.imgmsg_to_cv2(request.predicted_image, "bgr8")
        x, y, w, h = 288, 0, 704, 704
        resized_image = img[y:y+h, x:x+w]
        
        result = self.model(resized_image)
        df = result.pandas().xyxy[0]  # Results
        sorted_df = df.sort_values(by='confidence', ascending=False)

        response = InputYoloPredictionResponse()
        response.predicted_image = request.predicted_image
        try:
            response.score =  sorted_df.iloc[0]['confidence']

            name_msg = String()
            name_msg.data = self.class_names[sorted_df.iloc[0]['class']]
            response.name = name_msg
        except:
            response.score = 0
            name_msg = String()
            name_msg.data = ''
            response.name = name_msg

    
        end = time.time()
        rospy.loginfo(f'Sending back request in {end-start} seconds, result is: {response.name}')

        return response

if __name__ == '__main__':
    try:
        rospy.init_node("service_zed_yolov5",anonymous = True)
        yolo = service_yolo()
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


    except rospy.ROSInterruptException:
        pass
