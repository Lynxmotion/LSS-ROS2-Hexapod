import sys
import signal
import rclpy
import time
import math
import typing
from functools import reduce

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Int16MultiArray, String
from lss_hexapod.msg import Motion

# Default channel settings for R8FM
# 0 - Roll (right)
# 1 - Pitch (right)
# 2 - Throttle (left)
# 3 - Yaw (left)
# 4 - Right 3-way
# 5 - Right button
# 6 - Left 3-way
# 7 - Left trim POT


def constrain(min: float, v: float, max: float):
    if v < min:
        return min
    elif v > max:
        return max
    else:
        return v


class HexapodRadioControl(Node):
    walking_speed: float

    def __init__(self, args):
        super().__init__('hexapod_radio_control')
        self.shutdown = False
        self.walking_speed = 0.0

        best_effort_profile = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)

        # subscribe to radio channels
        self.radio_sub = self.create_subscription(
            Int16MultiArray,
            '/input/ppm',
            self.rc_callback,
            best_effort_profile)

        self.motion_pub = self.create_publisher(
            Motion,
            "/hexapod/motion",
            best_effort_profile)

    def rc_callback(self, msg: Int16MultiArray):
        if len(msg.data) >=8:
            motion = Motion()
            motion.walking_speed = constrain(0.0, (msg.data[2] - 1000) * 8.0 / 1000.0, 8.0)
            if motion.walking_speed < 0.5:
                motion.walking_speed = 0.0
            motion.target_heading = constrain(-15.0, (msg.data[3] - 1500) * 15.0 / 500.0, 15.0)
            if abs(motion.target_heading) < 1.0:
                motion.target_heading = 0.0
            if self.walking_speed is None or abs(self.walking_speed - motion.walking_speed) > 0.1:
                self.walking_speed = motion.walking_speed
                print(f'speed: {motion.walking_speed}')
            self.motion_pub.publish(motion)

    def ctrl_c(self, signum, frame):
        print(f'shutdown requested by {signum}')
        self.shutdown = True

    def run(self):
        try:
            while rclpy.ok() and not self.shutdown:
                rclpy.spin_once(self, timeout_sec=0)
        except KeyboardInterrupt:
            print('shutting down radio control')
        rclpy.shutdown()


def main(args=sys.argv):
    rclpy.init(args=args)
    node = HexapodRadioControl(args)
    signal.signal(signal.SIGINT, node.ctrl_c)
    node.run()

if __name__ == '__main__':
    main()

