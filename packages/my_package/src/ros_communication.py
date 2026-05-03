#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Range
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from cv_bridge import CvBridge
from process import process_all

class ROSCommunication(DTROS):
    def __init__(self, node_name):
        super(ROSCommunication, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._bridge = CvBridge()

        # Latest messages from each topic, initialized to None
        self._image = None
        self._tof = None
        self._left_encoder = None
        self._right_encoder = None

        # Subscribers
        rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed",
                         CompressedImage, self.cb_camera)
        rospy.Subscriber(f"/{self._vehicle_name}/front_center_tof_driver_node/range",
                         Range, self.cb_tof)
        rospy.Subscriber(f"/{self._vehicle_name}/left_wheel_encoder_driver_node/tick",
                         WheelEncoderStamped, self.cb_left_encoder)
        rospy.Subscriber(f"/{self._vehicle_name}/right_wheel_encoder_driver_node/tick",
                         WheelEncoderStamped, self.cb_right_encoder)

        # Publisher
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
    
    # --- Callbacks: just store the latest message ---
    def cb_camera(self, msg):
        self._image = self._bridge.compressed_imgmsg_to_cv2(msg)
        self.process()  # trigger processing on every new camera frame

    def cb_tof(self, msg):
        self._tof = msg.range

    def cb_left_encoder(self, msg):
        self._left_encoder = msg.data

    def cb_right_encoder(self, msg):
        self._right_encoder = msg.data

    def process(self):
        vel_left, vel_right = process_all(self)
        self._publisher.publish(WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right))
        
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

if __name__ == '__main__':
    node = ROSCommunication(node_name='ros_communication')
    rospy.spin()