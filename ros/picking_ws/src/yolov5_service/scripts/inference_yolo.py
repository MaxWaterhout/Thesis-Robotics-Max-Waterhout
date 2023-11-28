import rospy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import pandas as pd
import time

class inference_yolo:
    def __init__(self):
        self.model = self.load_model()
        self.predict_pub = rospy.Publisher("image_predictions_yolo", Image, queue_size = 10)
        self.class_colors = self.colors()
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/max/Documents/yolov5/weights/best.engine')  # local model
        model.cuda()
        print('loaded model')
        return model
    
    def colors(self):
        class_colors = {
            0: (255, 0, 0),    # Class 0: Red
            1: (0, 255, 0),    # Class 1: Green
            2: (0, 0, 255),    # Class 2: Blue
            3: (255, 255, 0),    
            4: (0, 255, 255),
            5: (255, 255, 255) 
        }
        return class_colors

    def image_callback(self,data):
        start = time.time()
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        x, y, w, h = 288, 0, 704, 704
        resized_image = img[y:y+h, x:x+w]

        img_with_boxes = resized_image.copy()
        
        result = self.model(resized_image)
        img_with_boxes = self.annotate_img(img_with_boxes, result)

        self.predict_pub.publish(img_with_boxes)
        end = time.time()
    
    def annotate_img(self,img, result):
        df = result.pandas().xyxy[0]  # Results
        for index, row in df.iterrows():
        
            # Load the image using OpenCV
            # Extract bounding box coordinates
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            thickness = 2  # Thickness of the bounding box lines
            print(row['class'], self.class_colors[row['class']])

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.class_colors[row['class']], thickness)
        
        img = self.bridge.cv2_to_imgmsg(img, "bgr8")

        return img

if __name__ == '__main__':
    try:
        rospy.init_node("inference_zed_yolov5",anonymous = True)
        yolo = inference_yolo()
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


    except rospy.ROSInterruptException:
        pass
