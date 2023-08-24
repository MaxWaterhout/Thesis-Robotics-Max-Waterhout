#!/home/max/env38/bin/python

from gc import callbacks
import rospy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from efficientpose_test.srv import InputPrediction,InputPredictionResponse
import numpy as np
from std_msgs.msg import String



class chess_client:
    def __init__(self):

        self.bridge = CvBridge()
        self.K_matrix = None
        self.depth_data = None
        self.image_data = None

        #self.prediction_service = rospy.Service('chess_prediction', predictionImage, self.handle_prediction_request)

        self.camera_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info",CameraInfo,self.camera_callback )
        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback)
        
        self.trigger_sub = rospy.Subscriber("prediction_trigger", String, self.trigger_callback)

    
    def depth_callback(self,data):
        self.depth_data = data
    
    def camera_callback(self, data):
        # This function will be called when a message is received on the topic
        # You can access the camera matrix and other information from the data argument
        if self.K_matrix is None:
            self.K_matrix = np.array([[data.K[0], 0.,data.K[2]], [0., data.K[4], data.K[5]], [0., 0., 1.]], dtype = np.float32)
            print("The k matrix: ", self.K_matrix)

    def image_callback(self, data):
        self.image_data = data

    def trigger_callback(self,msg):
        trigger_name = msg.data
        print(f"prediction for {trigger_name}")
        self.handle_prediction_request(trigger_name)

    def handle_prediction_request(self, name):

        print("prediction request")
        service_name = "predictions_{}".format(name)
        rospy.wait_for_service(service_name)  # Wait for the service to become available
        try:
            prediction_service = rospy.ServiceProxy(service_name, InputPrediction)
            
            # Create a request object
            request = InputPrediction._request_class()
            request.predicted_image = self.image_data
            request.depth_map = self.depth_data
            request.K_matrix = self.K_matrix.flatten()

            # Call the service with the request
            response = prediction_service(request)

            print(response.translations, "translation")

        except rospy.ServiceException as e:
            print("Service call failed:", str(e))


if __name__ == '__main__':
    try:
        rospy.init_node("detection_node_client",anonymous = True)
        detector = chess_client()

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()



    except rospy.ROSInterruptException:
        pass

