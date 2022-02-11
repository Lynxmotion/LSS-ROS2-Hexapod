import sys
import signal
import rclpy
import time
import math
import datetime
import typing
from functools import reduce
from operator import attrgetter

from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer, GoalResponse, CancelResponse
from rclpy.action.client import ClientGoalHandle, GoalStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Joy
from robot_model_msgs.msg import ModelState, ControlState, Limb, \
    TrajectoryProgress, TrajectoryComplete, SegmentTrajectory
from robot_model_msgs.action import EffectorTrajectory, CoordinatedEffectorTrajectory, LinearEffectorTrajectory
from robot_model_msgs.srv import Reset, ConfigureLimb, SetLimb
from lss_hexapod.msg import Motion
from lss_hexapod.action import Walk, Rotate
from scipy.spatial import ConvexHull

from robot import RobotState
from noisy import NoisyNumber
from ros_trajectory_builder import default_segment_trajectory_msg
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from tf_conv import to_kdl_rotation, to_kdl_vector, to_kdl_frame, to_vector3, to_transform, to_quaternion, to_geo_twist, P, R

from polar import PolarCoord, near_equal, near_zero
from leg import Leg, each_leg, tripod_set
from trajectory import PathTrajectory, LinearTrajectory

DynamicObject = lambda **kwargs: type("Object", (), kwargs)


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


class Hexapod(Node):
    # gait states
    IDLE = 0
    STANDING = 1
    WALKING = 2
    TURNING = 3
    SHIFT_LEG = 4

    # Joystick Axes
    ROLL = 0
    PITCH = 1
    THROTTLE = 2
    YAW = 3
    TRIM_POT = 4

    # Joystick buttons
    RIGHT_3WAY = 0
    LEFT_3WAY = 1
    RIGHT_BUTTON = 2

    # Joystick assignments
    REVERSE = RIGHT_BUTTON


    # if this value is >0 it is the number of spin loops before shutting down
    # give a few spin loops allows pending actions to complete or cancel
    # any value >=1000 means forever
    shutdown: int = 1000

    previewMode = False
    # preview_prefix - make it so we just set this to the prefix we want to base our trajectories on

    # segment names
    base_link = 'base_link'
    odom_link = 'odom'

    state: RobotState

    base_standing_z = 0.06
    base_sitting_z = 0.042

    walking_gait_velocity = 1.2
    stand_velocity = 0.4

    # milliseconds between each desired leg movement
    # we may move sooner but we try to walk to this beat
    beat = datetime.timedelta
    next_beat: datetime = None

    heading = 0
    target_heading = math.inf
    turn_rate: NoisyNumber
    walking_speed: NoisyNumber
    walk_reverse: bool

    base_twist: kdl.Twist
    current_base_twist: kdl.Twist = None

    urdf = None
    robot = None

    # odom => base transform
    odom: kdl.Frame

    # the current pose of the base
    base_pose: kdl.Frame = kdl.Frame()
    CoP: kdl.Vector
    support_margin: float

    # the XYZ location of each hip joint
    # we'll form a polar coordinate system centered on each hip to plan leg movements
    legs: dict = None

    gait: typing.Callable = None
    gait_state = 0
    gait_points: []
    gait_n: int
    neutral_offset = 0.5
    neutral_radius = 0.12

    timestep = 0

    leg_goal: rclpy.task.Future = None
    base_goal_handle: rclpy.task.Future = None

    def __init__(self, args):
        super().__init__('hexapod')

        self.state = RobotState()
        self.support_margin = math.nan

        self.base_twist = kdl.Twist(kdl.Vector(), kdl.Vector())
        self.turn_rate = NoisyNumber(0.0, 0.85, jump=0.5)
        self.turn_rate.on_trigger(0.05, self.turn)

        self.walking_speed = NoisyNumber(0.0, 0.80, jump=0.4)
        self.walking_speed.on_trigger(0.005, self.walk)
        self.walk_reverse = False

        # todo: change this according to walk speed (approx 1200 down to 500, slow to fast respectively)
        # todo: we can also slow beat if activity metric is slow
        self.beat = datetime.timedelta(milliseconds=600)


        self.leg_neighbors = {
            'left-front': ['left-middle', 'right-front'],
            'right-front': ['right-middle', 'left-front'],

            'left-middle': ['left-front', 'left-back'],
            'right-middle': ['right-front', 'right-back'],

            'left-back': ['left-middle', 'right-back'],
            'right-back': ['right-middle', 'left-back']
        }

        self.tripod_set = [
            ['left-middle', 'right-front', 'right-back'],
            ['right-middle', 'left-front', 'left-back']
        ]
        self.tripod_set_supporting = 0

        best_effort_profile = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)

        reliable_profile = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)

        transient_local_reliable_profile = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # pull URDF so we can determine tf linkages
        self.urdf_sub = self.create_subscription(
            String,
            '/robot_description',
            self.urdf_callback,
            transient_local_reliable_profile)

        # subscribe to model state
        self.model_state_sub = self.create_subscription(
            ModelState,
            '/robot_dynamics/model_state',
            self.model_state_callback,
            reliable_profile)

        # subscribe to model control state
        self.control_state_sub = self.create_subscription(
            ControlState,
            '/robot_control/control_state',
            self.control_state_callback,
            reliable_profile)

        self.reset_client = self.create_client(
            Reset,
            "/robot_control/reset")

        self.configure_limb_client = self.create_client(
            ConfigureLimb,
            "/robot_control/configure_limb")

        self.set_limb_client = self.create_client(
            SetLimb,
            "/robot_control/set_limb")

        self.trajectory_client = ActionClient(
            self,
            EffectorTrajectory,
            '/robot_control/trajectory')

        self.coordinated_trajectory_client = ActionClient(
            self,
            CoordinatedEffectorTrajectory,
            '/robot_control/coordinated_trajectory')

        self.linear_trajectory_client = ActionClient(
            self,
            LinearEffectorTrajectory,
            '/robot_control/linear_trajectory')

        #
        # Hexapod Actions for external clients
        #
        self.motion_sub = self.create_subscription(
            Motion,
            '/hexapod/motion',
            self.motion_callback,
            best_effort_profile)

        # subscribe to radio channels
        self.radio_sub = self.create_subscription(
            Joy,
            '/input/ppm',
            self.joy_callback,
            best_effort_profile)

        self.walk_goal = None
        self.walk_action = ActionServer(
            self,
            Walk,
            '~/walk',
            execute_callback=self.walk_action_callback,
            goal_callback=self.handle_goal_callback,
            handle_accepted_callback=self.walk_action_accept_callback,
            cancel_callback=self.walk_action_cancel_callback)

        self.move_timer_ = self.create_timer(
            0.1, self.move_robot)

    def resolve_legs(self, legs: typing.Iterable):
        # resolve an array of leg names to an array leg objects
        return [self.legs[leg] for leg in legs if leg in self.legs]

    def urdf_callback(self, msg):
        self.urdf = msg.data
        self.robot = URDF.from_xml_string(self.urdf)
        # todo: TrajectoryBuilder has an example of kdl_tree_from_urdf_model
        self.legs = dict()
        for leg_prefix in each_leg():
            chain = self.robot.get_chain(self.base_link, leg_prefix + '-hip-span1', links=False)
            hip_tf = kdl.Frame()
            for l in chain:
                joint = self.robot.joint_map[l]
                origin = joint.origin
                joint_tf = kdl.Frame(
                    kdl.Rotation.RPY(roll=origin.rpy[0], pitch=origin.rpy[1], yaw=origin.rpy[2]),
                    kdl.Vector(x=origin.xyz[0], y=origin.xyz[1], z=origin.xyz[2]))
                hip_tf = hip_tf * joint_tf
            hip_yaw = math.atan2(hip_tf.p[1], hip_tf.p[0])
            rot = kdl.Rotation.RPY(0.0, 0.0, hip_yaw)
            leg = self.legs[leg_prefix] = Leg(
                name=leg_prefix,
                origin=kdl.Frame(
                    rot,
                    hip_tf.p
                )
            )
            leg.origin_angle = hip_yaw
            if 'front' in leg.name:
                leg.neutral_angle = self.neutral_offset
            elif 'back' in leg.name:
                leg.neutral_angle = -self.neutral_offset
            if 'right' in leg.name:
                leg.reverse = True

            print(f'  limb: {leg.name} => {hip_tf.p}  yaw: {hip_yaw * 180 / math.pi}')

        print('created legs from URDF')

    def model_state_callback(self, msg: ModelState):
        if not self.legs:
            return

        # simply store model state until next IMU message
        expected_frame_id = 'preview/'+self.odom_link if self.previewMode else self.odom_link
        if msg.header.frame_id == expected_frame_id:
            self.model_state = msg

            # self.world = to_kdl_frame(msg.world)
            self.odom = to_kdl_frame(self.model_state.odom)
            self.base_pose = to_kdl_frame(self.model_state.base_pose)

            self.CoP = to_kdl_vector(self.model_state.support.center_of_pressure)

    def compute_support_margin_of_supporting_set(self):
        # todo: remove me if you can
        # this will run the old support_margin function
        margin = self.compute_support_margin(self.tripod_set[self.tripod_set_supporting])
        self.support_margin = margin

    def compute_support_margin(self, legs: typing.List[str]):

        # get position of support legs relative to odom frame
        # support_legs = [self.base_pose * l.rect for l_name, l in self.legs.items() if l.state == Leg.SUPPORTING]
        # todo: possibly remove conversion to odom frame (we can base it on robot frame)
        support_legs = [self.base_pose * l.rect for l in self.legs.values() if
                        l.rect is not None and l.name in legs]
        if len(support_legs) < 3:
            # no kinematic connection...we're probably falling. Ayeeeeeeeee!
            # no support margin for a non-kinematic attachment to ground
            return 0.0

        # now make triangles between each pair of support legs and the CoP,
        # figure out what the min(hyp) is of the triangles.
        # see: https://hackaday.io/project/21904-hexapod-modelling-path-planning-and-control/log/62326-3-fundamentals-of-hexapod-robot

        # convert the list of legs into pairs
        # for 2-3 legs this is trivial, for more than 3 legs we have an over-constrained kinematic connection
        # with the floor and must calculate the "convext hull" polygon of the legs in order to create our pairs
        elif len(support_legs) == 3:
            # cross match each pair
            lp = [[l.p.x(), l.p.y()] for l in support_legs]
            leg_pairs = [
                (lp[0], lp[1]),
                (lp[1], lp[2]),
                (lp[2], lp[0])
            ]
        else:
            # create array of points (array), call convex, get an array of points
            # back representing a polygon that surrounds all given points
            points = [[l.p.x(), l.p.y()] for l in support_legs]
            hull = ConvexHull(points)

            # todo: any leg points not in hull polygon should go to lift

            leg_pairs = []
            last_p = None
            for p in hull.points:
                if last_p is not None:
                    leg_pairs.append((last_p, p))
                last_p = p

        # take your points and go home
        # calculate height for each point
        h = []
        C = self.CoP
        for p1, p2 in leg_pairs:
            # |(X1-Xc)*(Y5-Yc)-(X5-Xc)*(Y1-Yc)|
            area = 0.5 * math.fabs((p1[0] - C[0]) * (p2[1] - C[1]) - (p2[0] - C[0]) * (p1[1] - C[2]))
            # | L1 |=√((X5 - X1) ^ 2 + (Y5 - Y1) ^ 2)
            p1_p2_length = math.sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
            h1 = (2 * area) / math.fabs(p1_p2_length) if p1_p2_length > 0.0 else 0.0
            h.append(h1)

        min_h = reduce(lambda a, b: a if a < b else b, h)
        return min_h
        # print(f'CoP: {self.model_state.header.frame_id} supporting:{len(support_legs)}:{len(leg_pairs)} margin:{min_h:6.4f}')

    def control_state_callback(self, msg: ControlState):
        if not self.legs:
            return
        expected_prefix = 'preview' if self.previewMode else ''
        if msg.header.frame_id != expected_prefix:
            return

        # update base pose
        #self.odom = self.state.get_frame(self.odom_link)
        self.base_pose = to_kdl_frame(msg.base.pose)

        # get the robot heading
        rz, ry, rx = self.base_pose.M.GetEulerZYX()
        self.heading = rz
        #if math.isinf(self.target_heading):
        #    self.target_heading = self.heading

        # update the Leg positions
        msg_leg: Limb
        for msg_leg in msg.limbs:
            leg_name = msg_leg.name.replace('-foot', '')
            leg = self.legs[leg_name]
            leg.rect = to_kdl_frame(msg_leg.effector.pose)
            leg.error = msg_leg.effector.error
            #polar = leg.to_polar(leg.rect)
            #rect = leg.to_rect(polar)
            #polar2 = leg.to_polar(leg.rect, False)
            #rect2 = leg.to_rect(polar2, False)
            #if leg_name == 'left-front':
            #    print(f'  P{polar}   R{rect}    O{leg.rect.p}')

    def motion_callback(self, msg: Motion):
        if near_zero(msg.heading) and near_zero(msg.walking_speed):
            self.cancel_base_motion()
            self.turn_rate.set(0)
            self.walking_speed.set(0)
        else:
            # update turn rate, converted to radians
            self.turn_rate.filter(msg.heading * math.pi / 180.0)
            self.walking_speed.filter(msg.walking_speed / 60.0)

    def joy_callback(self, msg: Joy):
        heading = msg.axes[self.YAW] * 20.0
        walking_speed = lift(0.06, msg.axes[self.THROTTLE] / 32.0)
        if msg.buttons[self.REVERSE]:
            walking_speed = -walking_speed

        if near_zero(heading) and near_zero(walking_speed):
            self.cancel_base_motion()
            self.turn_rate.set(0)
            self.walking_speed.set(0)
        else:
            # update turn rate, converted to radians
            self.turn_rate.filter(heading * math.pi / 180.0)
            self.walking_speed.filter(walking_speed)

    def clear(self, target_state: bool = False, trajectories: bool = False, limp: bool = False):
        self.reset_client.wait_for_service()
        request = Reset.Request()
        request.trajectories = trajectories
        request.target_state = target_state
        request.limp = limp
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        rez = future.result().success if future.done() else False
        print(f'clear result: {rez}')
        return rez

    def clear_trajectory(self):
        return self.clear(trajectories=True)

    def limp(self):
        return self.clear(trajectories=True, limp=True)

    def single_trajectory(
            self,
            goal: SegmentTrajectory,
            id: str = None,
            complete: typing.Callable = None,
            progress: typing.Callable = None,
            rejected: typing.Callable = None) -> rclpy.task.Future:
        goal_reached_future = rclpy.task.Future()
        request = EffectorTrajectory.Goal()
        request.goal.header.stamp = self.get_clock().now().to_msg()
        #request.goal.header.frame_id = goal.reference_frame
        if id:
            request.goal.id = id
        request.goal.segment = goal

        def handle_result(future):
            result = future.result()
            goal_reached_future.set_result(result.result.result)
            if complete:
                complete(result.result.result)

        def handle_response(future):
            goal_h = future.result()
            if goal_h.accepted:
                goal_reached_future.goal_handle = goal_h
                result_future = goal_h.get_result_async()
                result_future.add_done_callback(handle_result)
            else:
                print(f'trajectory request for {goal.segment} rejected')
                if rejected:
                    rejected()
                    goal_reached_future.cancel()

        def feedback_callback(feedback_msg):
            if progress:
                progress(feedback_msg.feedback.progress)

        self.trajectory_client.wait_for_server()
        goal_handle = self.trajectory_client.send_goal_async(request, feedback_callback=feedback_callback)
        goal_handle.add_done_callback(handle_response)
        return goal_reached_future

    def coordinated_trajectory(
               self,
               goals: typing.List[SegmentTrajectory],
               id: str = None,
               sync_duration: bool = True,
               complete: typing.Callable = None,
               progress: typing.Callable = None,
               rejected: typing.Callable = None) -> rclpy.task.Future:
        goal_reached_future = rclpy.task.Future()
        request = CoordinatedEffectorTrajectory.Goal()
        request.goal.header.stamp = self.get_clock().now().to_msg()
        #request.goal.header.frame_id = self.base_link
        if id:
            request.goal.id = id
        request.goal.sync_duration = sync_duration
        request.goal.segments = goals

        # goal reached
        def handle_result(future):
            result = future.result()
            goal_reached_future.set_result(result.result.result)
            if complete:
                complete(result.result.result)

        # did the action server accept or reject our goal?
        def handle_response(future):
            goal_h = future.result()
            if goal_h.accepted:
                goal_reached_future.goal_handle = goal_h
                result_future = goal_h.get_result_async()
                result_future.add_done_callback(handle_result)
            else:
                print(f'coordinated trajectory request for {id} rejected')
                if rejected:
                    rejected()
                    goal_reached_future.cancel()

        def feedback_callback(feedback_msg):
            if progress:
                progress(feedback_msg.feedback.progress)

        self.coordinated_trajectory_client.wait_for_server()
        goal_future = self.coordinated_trajectory_client.send_goal_async(request, feedback_callback=feedback_callback)
        goal_future.add_done_callback(handle_response)
        return goal_reached_future

    def linear_trajectory(
            self,
            effectors: str or typing.List[str],
            twists: kdl.Twist or typing.List[kdl.Twist],
            linear_acceleration: float = 0.0,
            angular_acceleration: float = 0.0,
            mode_in: int = SegmentTrajectory.UNCHANGED,
            mode_out: int = SegmentTrajectory.UNCHANGED,
            sync_duration: bool = True,
            id: str = None,
            complete: typing.Callable = None,
            progress: typing.Callable = None,
            rejected: typing.Callable = None) -> rclpy.task.Future:
        goal_reached_future = rclpy.task.Future()
        if not isinstance(effectors, list):
            effectors = [effectors]
        if not isinstance(twists, list):
            twists = [twists]
        # convert twists into geo msgs
        twists = [to_geo_twist(t) for t in twists]
        if linear_acceleration == 0.0 and angular_acceleration == 0.0:
            raise ValueError('requires either linear or angular acceleration')

        goal = LinearEffectorTrajectory.Goal()
        goal.header.stamp = self.get_clock().now().to_msg()
        if id:
            goal.id = id
        #goal.sync_duration = sync_duration
        goal.mode_in = mode_in
        goal.mode_out = mode_out
        goal.linear_acceleration = linear_acceleration
        goal.angular_acceleration = angular_acceleration
        goal.effectors = effectors
        goal.velocity = twists

        def handle_result(future):
            result = future.result()
            goal_reached_future.set_result(result.result.result)
            if complete:
                complete(result.result.result)

        def handle_response(future):
            goal_h = future.result()
            if goal_h.accepted:
                goal_reached_future.goal_handle = goal_h
                result_future = goal_h.get_result_async()
                result_future.add_done_callback(handle_result)
            else:
                print(f'linear trajectory request for {id} rejected')
                if rejected:
                    rejected()
                    goal_reached_future.cancel()

        def feedback_callback(feedback_msg):
            if progress:
                progress(feedback_msg.feedback.progress)

        self.linear_trajectory_client.wait_for_server()
        goal_handle = self.linear_trajectory_client.send_goal_async(goal, feedback_callback=feedback_callback)
        goal_handle.add_done_callback(handle_response)
        return goal_reached_future

    def trajectory(
            self,
            units: typing.List[PathTrajectory or LinearTrajectory],
            complete: typing.Callable = None):
        # create a future to return to our caller
        future = rclpy.task.Future()

        def execute_unit(unit: PathTrajectory or LinearTrajectory) -> rclpy.task.Future:
            if isinstance(unit, PathTrajectory):
                if isinstance(unit.goal, list):
                    print(f'executing coordinated trajectory {unit.id}')
                    return self.coordinated_trajectory(
                        goals=unit.goal,
                        id=unit.id,
                        sync_duration=unit.synchronize,
                        complete=complete_unit,
                        progress=unit.progress,
                        rejected=rejected_unit)
                else:
                    print(f'executing single trajectory {unit.id}')
                    return self.single_trajectory(
                        goal=unit.goal,
                        id=unit.id,
                        complete=complete_unit,
                        progress=unit.progress,
                        rejected=rejected_unit)
            elif isinstance(unit, LinearTrajectory):
                print(f'executing linear trajectory {unit.id}')
                return self.linear_trajectory(
                    effectors=unit.effectors,
                    twists=unit.twists,
                    linear_acceleration=unit.linear_acceleration,
                    angular_acceleration=unit.angular_acceleration,
                    id=unit.id,
                    sync_duration=unit.synchronize,
                    mode_in=unit.mode_in,
                    mode_out=unit.mode_out,
                    complete=complete_unit,
                    progress=unit.progress,
                    rejected=rejected_unit)
            else:
                print(f'{unit.id} like not like a segment')

        def rejected_unit():
            nonlocal units
            if len(units) and callable(units[0].rejected):
                units[0].rejected()
            next_unit()

        def complete_unit(result):
            nonlocal units
            if len(units) and callable(units[0].complete):
                units[0].complete(result)
            next_unit()

        def next_unit():
            nonlocal units
            if len(units) == 0:
                print('end of units')
                future.set_result(True)
            else:
                execute_unit(units[0])
                units = units[1:]

        if complete:
            future.add_done_callback(complete)

        # queue up the first trajectory
        next_unit()
        return future

    def set_limb_mode(self, limbs: typing.List[str], mode: int or typing.List[int]):
        self.set_limb_client.wait_for_service()
        if not isinstance(mode, list):
            mode = [mode] * len(limbs)
        request = SetLimb.Request()
        request.limbs = limbs
        request.mode = mode

        future = self.set_limb_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        rez = future.result().success if future.done() else False
        print(f'set support: {rez}')
        return rez

    @staticmethod
    def is_goal_active(goal_handle: ClientGoalHandle):
        # check if goal_handle ref is valid, and that we have a cancel_goal_async method,
        # if no method then the goal must have completed
        # todo: is there a better way to check if a goal_handle is complete?
        return goal_handle and not goal_handle.done()

    @staticmethod
    def is_cancellable(self, goal_handle):
        return goal_handle and hasattr(goal_handle, 'cancel_goal_async')

    def are_legs_active(self):
        return Hexapod.is_goal_active(self.leg_goal)

    def update_base(self):
        def done(result):
            if result.code != -5:
                self.base_goal_handle = None
            print('turn stopped')
        # todo: do better checking than this
        if not self.current_base_twist or \
                not near_equal(self.base_twist.vel, self.current_base_twist.vel, 0.01) or \
                not near_equal(self.base_twist.rot, self.current_base_twist.rot, 0.01):
            print(f'speed   linear: {self.base_twist.vel}   rot: {self.base_twist.rot}')
            self.current_base_twist = kdl.Twist(self.base_twist.vel, self.base_twist.rot)
            self.base_goal_handle = self.linear_trajectory(
                id='body',
                effectors=self.base_link,
                twists=self.base_twist,
                angular_acceleration=0.1,
                complete=done
            )

    def cancel_base_motion(self):
        if self.base_goal_handle:
            if hasattr(self.base_goal_handle, 'goal_handle'):
                print('calling goal cancel')
                def cancel_done(f):
                    print(f'cancel goal complete {f.result().return_code}')
                cancel_future = self.base_goal_handle.goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(cancel_done)
            elif hasattr(self.base_goal_handle, 'cancel'):
                print('cancelling turning (future)')
                self.base_goal_handle.cancel()
            self.base_goal_handle = None
            print("stopped")

    def turn(self, speed: float):
        """" speed is in degrees/sec """
        rps = speed / 2*math.pi
        print(f'turn => {rps} rot/sec')
        self.base_twist.rot.z(speed)
        self.update_base()

    def walk(self, speed: float):
        """" speed is in degrees/sec """
        print(f'walk => {speed*100.0} cm/s')
        self.base_twist.vel.y(speed)
        self.update_base()

    def goto_state(self, state: int, legs: str or Leg or typing.List[Leg] = None):
        if legs is None:
            legs = self.legs
        elif isinstance(legs, str):
            legs = [self.legs[legs]]
        elif isinstance(legs, Leg):
            legs = [legs]
        for l in legs:
            if isinstance(l, str):
                self.legs[l].state = state
            else:
                l.state = state

    def tripod_gait(self):
        #
        # We manage a beat (like a musical beat) for coordinating leg motion.
        # We can move in-between beats if stability warrants it but try to stay on a bat
        # so our gaits look more natural.
        now = datetime.datetime.now()
        beat_now = False
        support_margins = None
        if not self.next_beat:
            self.next_beat = now + self.beat
        elif self.next_beat < now:
            # we are at a beat
            beat_now = True
            self.next_beat = now + self.beat
            # compute the support margins of our leg sets
            # we can react with new trajectories if our support is suffering
            support_margins = [self.compute_support_margin(s) for s in self.tripod_set]
            print(f'support margins: ', ', '.join([str(round(v, 4)) for v in support_margins]))

        # check if our legs are already performing an active trajectory
        legs_active = self.are_legs_active()

        if self.gait_state == Hexapod.IDLE:
            # our robot is sitting, perform the standing action
            if not legs_active:
                print('standing up')
                units = self.stand_up()

                def done(r):
                    print('stood up')
                    self.gait_state = Hexapod.STANDING

                self.leg_goal = self.trajectory(units, complete=done)

        elif self.gait_state == Hexapod.STANDING:
            # react to our leg states and initiate leg lifts if required
            if not legs_active:
                target_angle = 0.0  # was 0.4

                if beat_now:
                    # we are on a beat and encouraged to move legs at this time
                    move_tripod_set: typing.List[str] = None        # list of legs to lift

                    # if support margin is near equal, then choose based on angle
                    if near_equal(support_margins[0], support_margins[1], 0.01):
                        # see if we need to do a tripod leg move
                        # (in reaction to turning for example)
                        # start by seeing what legs are more stretched out
                        max_leg: Leg = None
                        max_angle = 0.0
                        for leg in self.legs.values():
                            polar = leg.polar
                            if not max_leg or abs(polar.angle) > max_angle:
                                max_leg = leg
                                max_angle = abs(polar.angle)
                        if max_angle > 0.21:   # 0.21 radians => ~6 degrees
                            for s in self.tripod_set:
                                if max_leg.name in s:
                                    move_tripod_set = s

                    if not move_tripod_set and len(support_margins) == 2:
                        # choose the tripod set with the lowest support margin
                        move_tripod_set = self.tripod_set[0] \
                            if support_margins[0] < support_margins[1] else self.tripod_set[1]

                    # todo: if still no set selected, see if we need an adjustment due to positional error

                    if move_tripod_set:
                        self.step_tripod_set(move_tripod_set, target_angle)

                    # see if there is a leg needing adjustment due to large position error
                    #return
                    #leg = max(self.legs.values(), key=attrgetter('error'))
                    #if leg.error > 0.02:
                    #    print(f'adjusting {leg.name}   e: {leg.error}')
                    #    self.leg_goal = self.trajectory(self.leg_adjustment(leg))
                    #    return

        else:
            print('entering standing state')
            # todo: analyze leg positions currently, and decide how to stand
            self.goto_state(Hexapod.IDLE)

    def step_tripod_set(self, leg_set, target_angle: float):
        """ Take a step with a tri-set using a single trajectory action sequence. """
        movements = []
        for sl_name in leg_set:
            leg = self.legs[sl_name]
            leg.state = Leg.LIFTING
            to = PolarCoord(
                angle=target_angle,
                distance=self.neutral_radius,
                z=-self.base_standing_z)
            if to.near(leg.polar, 0.08, 0.01):
                # this leg is already close enough to destination
                # no need to lift
                print(f'supressing leg {leg.name}')
                continue

            # single trajectory for up and down phase
            traj = leg.lift(to, velocity=self.walking_gait_velocity)
            movements.append(traj)

        self.leg_goal = self.coordinated_trajectory(
            movements,
            id='leg-lift')

    def step_tripod_set_2phase(self, leg_set, target_angle: float):
        """ Take a step with a tri-set using 2 consecutive trajectory action sequences.
            This is to try and prevent Orokos path following errors """
        movements_up = []
        movements_down = []  # switch which lift pattern is used
        for sl_name in leg_set:
            leg = self.legs[sl_name]
            leg.state = Leg.LIFTING
            to = PolarCoord(
                angle=target_angle,
                distance=self.neutral_radius,
                z=-self.base_standing_z)
            if to.near(leg.polar, 0.08, 0.01):
                # this leg is already close enough to destination
                # no need to lift
                print(f'supressing leg {leg.name}')
                continue

            # separate trajectory for up and down phase
            traj_up, traj_down = leg.lift_2stage(to, velocity=self.walking_gait_velocity)
            movements_up.append(traj_up)
            movements_down.append(traj_down)

        self.leg_goal = self.trajectory([
            PathTrajectory(id='leg-lift', goal=movements_up),
            PathTrajectory(id='leg-lift', goal=movements_down)
        ])

    def stand_up(self):
        """Return a path that stands the robot up """
        to_the_heavens = []
        to_standing_high_pose = []
        to_standing_pose = []
        the_legs = each_leg()
        for l_name in the_legs:
            leg = self.legs[l_name]
            to_the_heavens.append(
                default_segment_trajectory_msg(
                    leg.foot_link,
                    velocity=self.stand_velocity,
                    acceleration=0.1,
                    mode_in=SegmentTrajectory.HOLD,
                    points=[leg.to_rect(PolarCoord(
                        angle=0.0,  # was 0.4
                        distance=self.neutral_radius * 1.1,
                        z=0.02))]
                ))

            to_standing_high_pose.append(
                default_segment_trajectory_msg(
                    leg.foot_link,
                    velocity=self.stand_velocity,
                    acceleration=0.2,
                    mode_in=SegmentTrajectory.SUPPORT,
                    points=[leg.to_rect(PolarCoord(
                        angle=0.0,  # was 0.4
                        distance=self.neutral_radius,
                        z=-self.base_standing_z * 1.5))]
                ))

            to_standing_pose.append(
                default_segment_trajectory_msg(
                    leg.foot_link,
                    velocity=0.05,
                    acceleration=0.03,
                    mode_out=SegmentTrajectory.SUPPORT,
                    points=[leg.to_rect(PolarCoord(
                        angle=0.0,  # was 0.4
                        distance=self.neutral_radius,
                        z=-self.base_standing_z))]
                ))

        return [
            PathTrajectory(id='stand-lift', goal=to_the_heavens),
            PathTrajectory(id='stand-high', goal=to_standing_high_pose),
            PathTrajectory(id='stand-down', goal=to_standing_pose)
        ]

    def leg_adjustment(self, leg: str or Leg, height = 0.02):
        if isinstance(leg, str):
            leg = self.legs[leg]
        up = default_segment_trajectory_msg(
            leg.foot_link,
            velocity=self.stand_velocity,
            supporting=True,
            points=[leg.to_rect(PolarCoord(
                angle=0.0,  # was 0.4
                distance=self.neutral_radius,
                z=-self.base_standing_z + height))])
        down = default_segment_trajectory_msg(
            leg.foot_link,
            velocity=self.stand_velocity,
            supporting=True,
            points=[leg.to_rect(PolarCoord(
                angle=0.0,  # was 0.4
                distance=self.neutral_radius,
                z=-self.base_standing_z))])
        return [PathTrajectory(up, id='shift-up'),
                PathTrajectory(down, id='shift-down')]

    def move_robot(self):
        if not self.legs or not self.base_pose:
            return          # wait for all the meta information we need

        if self.walk_goal:
            if self.walk_goal.is_cancel_requested:
                self.walk_goal.canceled()
                # todo: return result here
                self.walk_goal = None
            else:
                # send feedback
                feedback = Walk.Feedback()
                feedback.heading = self.heading
                feedback.odom = to_vector3(self.odom.p)
                self.walk_goal.publish_feedback(feedback)

        if self.gait:
            self.gait()

    def walk_action_accept_callback(self, goal_handle):
        print('accept_callback')
        self.walk_goal = goal_handle
        #self.walk(
        #    math.pi / 2 + goal_handle.request.heading,
        #    goal_handle.request.speed)

    def handle_goal_callback(self, goal):
        print('goal_callback')
        if self.walk_goal:
            # cancel the existing goal
            self.walk_goal.canceled()
            self.walk_goal = None
        return GoalResponse.ACCEPT if goal.walking_speed > 0 else GoalResponse.REJECT

    def walk_action_callback(self, goal_handle):
        print('walk_callback')
        #self.gait = self.coordinated_tripod_gait
        self.walk(
            math.pi / 2 + goal_handle.request.heading,
            goal_handle.request.walking_speed)

    def walk_action_cancel_callback(self, goal_handle):
        print("cancel walking")
        # todo: make this an idle/still gait (it continuously moves legs to center, most stretched tri first)
        self.gait = None

    def set_gait(self, gait: typing.Callable):
        self.gait = gait

    def ctrl_c(self, signum, frame):
        print(f'shutdown requested by {signum}')
        self.cancel_base_motion()
        self.shutdown = 5

    def run(self):
        try:
            while rclpy.ok() and self.shutdown > 0:
                rclpy.spin_once(self, timeout_sec=0)
            if self.shutdown < 1000:
                self.shutdown -= 1
        except KeyboardInterrupt:
            print('shutting down')
        rclpy.shutdown()


def main(args=sys.argv):
    rclpy.init(args=args)
    node = Hexapod(args)
    signal.signal(signal.SIGINT, node.ctrl_c)
    node.clear_trajectory()
    node.set_gait(node.tripod_gait)
    node.run()


def test1():
    hip_base = kdl.Frame(kdl.Vector(1, 1, 0.5))
    hip_yaw = math.atan2(hip_base.p[1], hip_base.p[0])
    rot = kdl.Rotation.RPY(0.0, 0.0, hip_yaw)
    hip_base.M = rot

    leg = Leg(
        name='test',
        origin=hip_base)
    base_pose = kdl.Frame(kdl.Vector(0, 0, 0))
    rect = kdl.Vector(3, 3, 0)
    polar = leg.to_polar(rect, base_pose)
    rect2 = leg.to_rect(polar, base_pose)

    print(rect, rect2)

def test2():
    hip_base = kdl.Frame(kdl.Vector(1, 1, 0.5))
    hip_yaw = math.atan2(hip_base.p[1], hip_base.p[0])
    rot = kdl.Rotation.RPY(0.0, 0.0, hip_yaw)
    hip_base.M = rot

    leg = Leg(
        name='test',
        origin=hip_base)
    base_pose = kdl.Frame(kdl.Vector(1, -2, 0))
    rect = kdl.Vector(3, 3, 0)
    polar = leg.to_polar(rect, base_pose)
    rect2 = leg.to_rect(polar, base_pose)

def test3():
    hip_base = kdl.Frame(kdl.Vector(1, -1, 0.5))
    hip_yaw = math.atan2(hip_base.p[1], hip_base.p[0])
    rot = kdl.Rotation.RPY(0.0, 0.0, hip_yaw)
    hip_base.M = rot

    leg = Leg(
        name='test',
        origin=hip_base)
    base_pose = kdl.Frame(kdl.Vector(1, -2, 0))
    rect = kdl.Vector(3, -3, 0)
    polar = leg.to_polar(rect, base_pose)
    rect2 = leg.to_rect(polar, base_pose)
    print(rect, rect2)

if __name__ == '__main__':
    #test1()
    #test2()
    #test3()
    main()
