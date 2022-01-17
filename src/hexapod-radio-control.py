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
    """ constrain or clamp a number between a min and max value """
    if v < min:
        return min
    elif v > max:
        return max
    else:
        return v


def deadband(v: int, bandwidth: int):
    """ keep a value at 0 until a certain threshold is passed. Useful on noisy or uncalibrated joysticks
        where we want to keep the robot at a neutral position until a certain amount of input is given.
    """
    if bandwidth == 0:
        return v
    elif v < 0:
        return 0 if v > -bandwidth else v + bandwidth
    else:
        return 0 if v < bandwidth else v - bandwidth


def map_throttle(v: int, zero_point: int, min_throttle: int = 0):
    """ map a PPM channel value to a throttle-like output where a certain minimal throttle
        must be seen before activating motion..
    """
    return 0 if v < zero_point else v - zero_point + min_throttle


def map_throttle_reversable(v: int, zero_point: int, with_deadband: int = 0):
    """ map a PPM channel value to a throttle-like output, similar to map_throttle(), but in this case
        the stick is centered and pulling back on mid position indicates a reverse throttle or "go backwards".
    """
    return deadband(v - zero_point, with_deadband)


def map_midstick(v: int, with_deadband: int = 0):
    """ map a PPM channel value to +/- around a midstick position. Useful for roll/pitch/yaw like controls.
    """
    return deadband(v - 1500, with_deadband)


def map_switch(v: int, bands: int):
    """ map a PPM channel to an option switch with the number of bands (toggle position), typical used for
        2 or 3-way switches.
    """
    v = constrain(0, v - 1000, 999)
    bandwidth = 1000 / bands
    return int(v / bandwidth)


def ppm_remap(v: int, min_output: float or int, max_output: float or int, input_offset: int = 1000):
    """ remap the PPM channel range (1000 to 2000) to a scaled floating output value.
        Since other PPM map functions may change the normal 1000 offset to 0 or a +/- value you can
        set the base offset using the 'input_offset' argument.
    """
    if v <= input_offset:
        return min_output
    elif v >= input_offset + 1000:
        return max_output
    elif max_output < min_output:
        # min/max is reversed, so reverse the mapping and call recursively
        # reverse input expression is equivalent to: 1000 - (v - input_offset) + input_offset
        return ppm_remap(2 * input_offset + 1000 - v, max_output, min_output)
    else:
        return min_output + (v - input_offset) / 1000.0 * (max_output - min_output)


def lift(offset: float, v: float):
    """ if value is non-zero, add an offset to lift the value to a minimum value.
        This is useful for providing a minimum amount of power or speed to a movement. Works with
        negative and positive numbers.
    """
    if v < 0.0:
        return v + -offset
    elif v > 0.0:
        return v + offset
    else:
        return 0.0


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
        if len(msg.data) >= 8:
            motion = Motion()

            # 3-way switch to sit, stand and <unassigned)
            motion.behavior = map_switch(msg.data[6], 3)

            # map throttle to walking speed
            # unfortunately no reverse...yet
            motion.walking_speed = lift(0.6, ppm_remap(map_throttle(msg.data[2], 1300), 0.0, 6.0, input_offset=0))
            #motion.walking_speed = constrain(0.0, read_throttle(msg.data[2] - 1000, 1200) * 8.0 / 1000.0, 8.0)

            # map yaw to turning speed (rotates the base)
            motion.heading_mode = Motion.DEG_SEC
            motion.heading = ppm_remap(map_midstick(msg.data[3], 100), -30.0, +30.0, input_offset=-500)
            #motion.heading = constrain(-15.0, (msg.data[3] - 1500) * 15.0 / 500.0, 15.0)

            # map standing height
            motion.standing_height = ppm_remap(msg.data[7], 0.0, 0.1)

            # map roll and pitch to +/- degrees
            motion.roll = ppm_remap(map_midstick(msg.data[0], 100), -40.0, 40.0, input_offset=-500)
            motion.pitch = ppm_remap(map_midstick(msg.data[1], 100), -40.0, 40.0, input_offset=-500)

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
    #x = [1500, 1505, 1495, 1650, 1350, 1100, 1900, 1000, 2000]
    #y = [read_midstick(v, 100) for v in x]
    #y = [map_midstick(v, 100) / 500.0 * 15.0 for v in x]
    #z = [lift(3.0, map_midstick(v, 100) / 500.0 * 15.0) for v in x]
    #roll = [ppm_remap(map_midstick(v, 100), -40.0, 40.0, input_offset=-500) for v in x]
    rclpy.init(args=args)
    node = HexapodRadioControl(args)
    signal.signal(signal.SIGINT, node.ctrl_c)
    node.run()

if __name__ == '__main__':
    main()

