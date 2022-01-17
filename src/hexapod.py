import sys
import signal
import rclpy
import time
import math
import typing
from functools import reduce
from operator import attrgetter

from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer, GoalResponse, CancelResponse
from rclpy.action.client import ClientGoalHandle, GoalStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, String
from robot_model_msgs.msg import ModelState, ControlState, Limb, \
    TrajectoryProgress, TrajectoryComplete, SegmentTrajectory
from robot_model_msgs.action import EffectorTrajectory, CoordinatedEffectorTrajectory, LinearEffectorTrajectory
from robot_model_msgs.srv import Reset, ConfigureLimb
from lss_hexapod.msg import Motion
from lss_hexapod.action import Walk, Rotate
from scipy.spatial import ConvexHull

from robot import RobotState
from noisy import NoisyNumber
from ros_trajectory_builder import default_segment_trajectory_msg
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from tf_conv import to_kdl_rotation, to_kdl_vector, to_kdl_frame, to_vector3, to_transform, to_quaternion, to_geo_twist, P, R

from polar import PolarCoord
from leg import Leg, each_leg, tripod_set
from trajectory import PathTrajectory, LinearTrajectory

DynamicObject = lambda **kwargs: type("Object", (), kwargs)


class Hexapod(Node):
    # gait states
    IDLE = 0
    STANDING = 1
    WALKING = 2
    TURNING = 3
    SHIFT_LEG = 4

    tasks = None

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

    heading = 0
    target_heading = math.inf
    turn_rate: NoisyNumber

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
    turn_goal_handle: rclpy.task.Future = None

    def __init__(self, args):
        super().__init__('hexapod')
        self.shutdown = False

        self.state = RobotState()
        self.support_margin = math.nan

        self.turn_rate = NoisyNumber(0.0, 0.85)
        self.turn_rate.on_trigger(2.0, self.turn)

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
            self.odom = to_kdl_frame(msg.odom)
            self.base_pose = to_kdl_frame(msg.base_pose)

            self.CoP = to_kdl_vector(self.model_state.support.center_of_pressure)

            # get position of support legs relative to odom frame
            #support_legs = [self.base_pose * l.rect for l_name, l in self.legs.items() if l.state == Leg.SUPPORTING]
            support_legs = [self.base_pose * l.rect for l in self.legs.values() if
                              l.rect is not None and l.name in self.tripod_set[self.tripod_set_supporting]]
            if len(support_legs) < 3:
                return

            # now make triangles between each pair of support legs and the CoP,
            # figure out what the min(hyp) is of the triangles.
            # see: https://hackaday.io/project/21904-hexapod-modelling-path-planning-and-control/log/62326-3-fundamentals-of-hexapod-robot

            # convert the list of legs into pairs
            # for 2-3 legs this is trivial, for more than 3 legs we have an over-constrained kinematic connection
            # with the floor and must calculate the "convext hull" polygon of the legs in order to create our pairs
            if len(support_legs) <= 2:
                # no kinematic connection...we're probably falling. Ayeeeeeeeee!
                self.support_margin = 0
                return
            elif len(support_legs) <= 3:
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
                area = 0.5 * math.fabs((p1[0] - C[0])*(p2[1] - C[1])-(p2[0] - C[0])*(p1[1] - C[2]))
                # | L1 |=√((X5 - X1) ^ 2 + (Y5 - Y1) ^ 2)
                p1_p2_length = math.sqrt((p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1]))
                h1 = (2 * area) / math.fabs(p1_p2_length) if p1_p2_length > 0.0 else 0.0
                h.append(h1)

            min_h = reduce(lambda a, b: a if a < b else b, h)
            self.support_margin = min_h
            #print(f'CoP: {self.model_state.header.frame_id} supporting:{len(support_legs)}:{len(leg_pairs)} margin:{min_h:6.4f}')

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
        # update turn rate
        # self.target_heading = msg.target_heading
        self.turn_rate.filter(msg.heading)
        if self.turn_rate.filtered < 0.2:
            if self.turn_goal_handle:
                self.cancel_turning()

        if msg.walking_speed > 0.0:
            self.walking_gait_velocity = msg.walking_speed
            if self.gait != self.coordinated_tripod_gait:
                print(f'gait => walk   (speed={msg.walking_speed}')
                self.set_gait(self.coordinated_tripod_gait)
        else:
            if self.gait != self.standing_gait:
                self.set_gait(self.standing_gait)

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
            supporting: bool = False,
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
        goal.supporting = supporting
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
                print(f'executing coordinated trajectory {unit.id}')
                return self.coordinated_trajectory(
                    goals=unit.goal,
                    id=unit.id,
                    sync_duration=unit.synchronize,
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
                    supporting=unit.supporting,
                    complete=complete_unit,
                    progress=unit.progress,
                    rejected=rejected_unit)
            else:
                print(f'{unit.id} like not like a segment')

        def rejected_unit(result):
            nonlocal units
            if len(units) and callable(units[0].rejected):
                units[0].rejected(result)
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
                print('next unit')
                next_task = execute_unit(units[0])
                if next_task.done():
                    print('   unit is already done')
                units = units[1:]
                #next_task.add_done_callback(next_unit)

        if complete:
            future.add_done_callback(complete)

        # queue up the first trajectory
        next_unit()
        return future


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

    def cancel_turning(self):
        if self.turn_goal_handle:
            if hasattr(self.turn_goal_handle, 'cancel_goal_async'):
                print('cancelling turning (client goal)')
                self.turn_goal_handle.cancel_goal_async()
            elif hasattr(self.turn_goal_handle, 'cancel'):
                print('cancelling turning (future)')
                self.turn_goal_handle.cancel()
            self.turn_goal_handle = None
            print("stopped")

    def turn(self, speed: float = 0.0):
        """" speed is in degrees/sec """
        if abs(speed) < 0.01:
            self.cancel_turning()
        else:
            print(f'speed => {speed} deg/sec')
            speed *= math.pi / 180.0  # convert into radians/sec

            def done():
                self.turn_goal_handle = None
                print('turn stopped')

            self.turn_goal_handle = self.linear_trajectory(
                effectors=self.base_link,
                twists=kdl.Twist(kdl.Vector(), kdl.Vector(x=0.0, y=0.0, z=speed)),
                angular_acceleration=0.1,
                complete=done
            )

    def move_leg_to(
            self,
            leg: Leg or str,
            point: (PolarCoord or kdl.Frame),
            reference_frame: str,
            complete: int or typing.Callable,
            **kwargs):
        if isinstance(leg, str):
            leg = self.legs[leg]

        if isinstance(point, PolarCoord):
            point = leg.to_rect(point)
            if reference_frame == self.odom_link:
                point = self.base_pose * point

        if type(complete) == int:
            def next_state(result):
                leg.state = complete
            on_complete = next_state
        else:
            on_complete = complete

        if reference_frame == self.base_link:
            rot = kdl.Rotation()
        elif reference_frame == self.odom_link:
            rot = self.base_pose.M

        traj = default_segment_trajectory_msg(
            leg.foot_link,
            velocity=self.walking_gait_velocity,
            reference_frame=reference_frame,
            points=[to_vector3(point)],
            rotations=[to_quaternion(rot)])

        leg.state = Leg.LIFTING
        print(f'leg lift: {traj.segment}')
        self.single_trajectory(traj, id='move-leg', complete=on_complete, **kwargs)

    def lift_leg(self, leg: Leg or str, to: PolarCoord, on_complete: typing.Callable = None):
        if isinstance(leg, str):
            leg = self.legs[leg]

        traj = leg.lift(to, velocity=self.walking_gait_velocity)

        #def progress(p: TrajectoryProgress):
        #    #print(f'   leg lift progress: {p.segment}  {p.progress*100:3.1f}% of {p.duration:3.2f}s')
        #    pass

        def complete(r: TrajectoryComplete):
            #print(f'leg lift complete: {r.segment} {r.transform.translation}')
            leg.state = Leg.SUPPORTING
            # use the ending position of the trajectory as the support position
            # but now specify it relative to the odom frame
            #point = self.base_pose * to_kdl_frame(r.transform)
            #point = to_kdl_frame(r.transform)
            #self.support_leg(leg, point)
            if on_complete:
                on_complete(r)

        leg.state = Leg.LIFTING
        self.single_trajectory(traj, complete=complete)

    def stance_leg(self, leg: Leg or str, to: kdl.Vector, on_complete: typing.Callable = None):
        if isinstance(leg, str):
            leg = self.legs[leg]

        # compute direction to move
        current_pos = leg.rect.p
        target_pos = current_pos + to

        def complete(r: TrajectoryComplete):
            #print(f'leg lift complete: {r.segment} {r.transform.translation}')
            leg.state = Leg.SUPPORTING
            # use the ending position of the trajectory as the support position
            # but now specify it relative to the odom frame
            #point = self.base_pose * to_kdl_frame(r.transform)
            #point = to_kdl_frame(r.transform)
            #self.support_leg(leg, point)
            if on_complete:
                on_complete(r)

        leg.state = Leg.ADVANCING
        print(f'stance leg {leg.name} from {leg.polar} => {to}')
        support_mode = default_segment_trajectory_msg(
            leg.foot_link,
            id='stance-leg',
            velocity=self.walking_gait_velocity,
            supporting=True,
            reference_frame=self.base_link,
            points=[P(target_pos[0], target_pos[1], leg.origin.p[2] - self.base_standing_z)],
            rotations=[to_quaternion(leg.rect.M)])
        self.single_trajectory(support_mode, complete=complete)

    def take_coordinated_step(self):
        movements = []

        # get leg sets
        non_supporting = [l for l in self.legs.values() if
                          l.name not in self.tripod_set[self.tripod_set_supporting]]
        supporting = [l for l in self.legs.values() if
                      l.name in self.tripod_set[self.tripod_set_supporting]]

        supporting_state = list(set([l.state for l in supporting]))
        non_supporting_state = list(set([l.state for l in non_supporting]))

        # if non-supporting set is now supportive, then
        # lift change the supporting set to lift
        #non_supportive_is_supporting = reduce(
        #    lambda a, b: a and b,
        #    [l.state == Leg.SUPPORTING for l in non_supporting]
        #)

        # swap the supporting set
        self.tripod_set_supporting = (self.tripod_set_supporting + 1) % 2

        # perform a leg lift on the supportive set
        for leg in supporting:
            to = PolarCoord(
                angle=-0.2,     # was 0.4
                distance=self.neutral_radius,
                z=-self.base_standing_z)
            #leg.state = Leg.LIFTING
            #if leg.name == 'left-front':
            leg.state = Leg.LIFTING
            traj = leg.lift(to, velocity=self.walking_gait_velocity)
            movements.append(traj)

        # perform stance move on what is now the new supporting set
        # first calculate what distance we will move by calculating possible target positions for each foot
        targets = []
        for leg in non_supporting:
            foot = leg.rect
            dest = leg.to_rect(PolarCoord(angle=0.3, distance=0.10))
            delta = dest - foot.p
            distance = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
            targets.append(distance)

        # choose the min distance to be the effective stance move
        target_distance = min(targets)
        target_polar = PolarCoord(angle=-math.pi / 2, distance=target_distance)

        #dest = PolarCoord(angle=self.target_heading + math.pi/2, distance=0.2)
        #print(f'base to {dest.x}, {dest.y}    heading={self.target_heading * 180 / math.pi}  current heading={self.heading * 180 / math.pi}')
        for leg in non_supporting:
            #if leg.name == 'left-front':
            leg.state = Leg.SUPPORTING
            # compute direction to move
            to = kdl.Vector(target_polar.x, target_polar.y, 0.)
            current_pos = leg.rect.p
            target_pos = current_pos + to
            # create trajectory
            support_move = default_segment_trajectory_msg(
                leg.foot_link,
                velocity=self.walking_gait_velocity,
                supporting=True,
                reference_frame=self.base_link,
                points=[P(target_pos[0], target_pos[1], leg.origin.p[2] - self.base_standing_z)],
                rotations=[to_quaternion(leg.rect.M)])
            movements.append(support_move)

        def take_another_step(result: TrajectoryComplete):
            # resume walking only if we are still in our tripod gait
            if self.gait and self.gait == self.coordinated_tripod_gait:
                if result.code == TrajectoryComplete.SUCCESS:
                    self.take_coordinated_step()
                else:
                    print(f'walking code {result.code}   duration:{result.duration}')

        if len(movements):
            self.leg_goal = self.coordinated_trajectory(movements, id='walk', complete=take_another_step)


    def coordinated_tripod_gait(self):
        # if stability margin threshold is met, switch sets
        # if self.support_margin < 0.01:
        if self.gait_state == Hexapod.IDLE:
            ready = 0
            # move any IDLE legs to standing state
            for leg in self.legs.values():
                if leg.state == Leg.IDLE:
                    self.move_leg_to(
                        leg=leg,
                        point=PolarCoord(
                            angle=0.,
                            distance=self.neutral_radius,
                            z=-self.base_standing_z),
                        reference_frame=self.base_link,
                        complete=Leg.SUPPORTING)
                elif leg.state == Leg.SUPPORTING:
                    ready = ready + 1
            if ready == len(self.legs):
                #for leg in self.legs.values():
                #    self.support_leg(leg, PolarCoord(leg.neutral_angle, self.neutral_radius, z=-self.base_standing_z))
                print("All legs ready!!")
                self.gait_state = Hexapod.WALKING
                self.take_coordinated_step()
        if self.gait_state == Hexapod.STANDING and self.walking_gait_velocity > 0.0:
            self.gait_state = Hexapod.WALKING
            self.take_coordinated_step()
        #elif self.gait_state == Hexapod.WALKING:

    def goto_state(self, state: int, legs: str or Leg or typing.List[Leg] = None):
        if legs is None:
            legs = self.legs
        elif isinstance(legs, str):
            legs = [self.legs[legs]]
        elif isinstance(legs, Leg):
            legs [legs]
        for l in legs:
            if isinstance(l, str):
                l = self.legs[l]
            l.state = state

    def standing_gait(self):
        if self.gait_state == Hexapod.IDLE:
            if not self.are_legs_active():
                print('standing up')
                units = self.stand_up()

                def done(r):
                    print('stood up')
                    self.gait_state = Hexapod.STANDING

                self.leg_goal = self.trajectory(units, complete=done)

        elif self.gait_state == Hexapod.STANDING:
            if not self.are_legs_active():
                # see if we need to do a tripod leg move
                # (in reaction to turning for example)
                # start by seeing what legsis more stretched out
                max_leg: Leg = None
                max_angle = 0.0
                for leg in self.legs.values():
                    polar = leg.polar
                    if not max_leg or abs(polar.angle) > max_angle:
                        max_leg = leg
                        max_angle = abs(polar.angle)

                if max_angle > 0.35:    # 20 degrees
                    # move the set of legs that includes this leg
                    for s in self.tripod_set:
                        if max_leg.name in s:
                            # move this set
                            movements = []
                            for sl_name in s:
                                leg = self.legs[sl_name]
                                to = PolarCoord(
                                    angle=0.0,  # was 0.4
                                    distance=self.neutral_radius,
                                    z=-self.base_standing_z)
                                # leg.state = Leg.LIFTING
                                # if leg.name == 'left-front':
                                leg.state = Leg.LIFTING
                                traj = leg.lift(to, velocity=self.walking_gait_velocity)
                                movements.append(traj)
                            self.leg_goal = self.coordinated_trajectory(movements, id='turn-step')
                            return

                # see if there is a leg needing adjustment due to large position error
                leg = max(self.legs.values(), key=attrgetter('error'))
                if leg.error > 0.02:
                    print(f'adjusting {leg.name}   e: {leg.error}')
                    self.leg_goal = self.trajectory(self.leg_adjustment(leg))
                    return


        elif self.gait_state == Hexapod.WALKING:
            print('walk => standing')
            # stand the robot up
            to_standing_pose = []
            for l_name in each_leg():
                leg = self.legs[l_name]
                to_standing_pose.append(
                    default_segment_trajectory_msg(
                        leg.foot_link,
                        velocity=self.stand_velocity,
                        supporting=True,
                        points=[leg.to_rect(PolarCoord(
                            angle=0.0,  # was 0.4
                            distance=self.neutral_radius,
                            z=-self.base_standing_z))]
                    ))

            self.gait_state = Hexapod.STANDING
            self.leg_goal = self.coordinated_trajectory(
                to_standing_pose,
                id='stand-up',
                complete=lambda res: self.goto_state(Leg.SUPPORTING))

        else:
            print('entering standing state')
            # todo: analyze leg positions currently, and decide how to stand
            self.goto_state(Hexapod.IDLE)

    def stand_up(self):
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
                    points=[leg.to_rect(PolarCoord(
                        angle=0.0,  # was 0.4
                        distance=self.neutral_radius * 1.1,
                        z=0.02))]
                ))

            to_standing_high_pose.append(
                default_segment_trajectory_msg(
                    leg.foot_link,
                    velocity=self.stand_velocity,
                    acceleration=0.8,
                    supporting=True,
                    points=[leg.to_rect(PolarCoord(
                        angle=0.0,  # was 0.4
                        distance=self.neutral_radius,
                        z=-self.base_standing_z * 1.5))]
                ))

            to_standing_pose.append(
                default_segment_trajectory_msg(
                    leg.foot_link,
                    velocity=0.05,
                    acceleration=0.01,
                    supporting=True,
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
        if not self.legs or not self.base_pose or math.isnan(self.support_margin):
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

    def walk(self, heading: float, speed: float, distance: float = None):
        self.walking_gait_velocity = speed
        self.set_gait(self.coordinated_tripod_gait)


    def ctrl_c(self, signum, frame):
        print(f'shutdown requested by {signum}')
        self.cancel_turning()
        self.shutdown = True

    def run(self):
        try:
            while rclpy.ok() and not self.shutdown:
                rclpy.spin_once(self, timeout_sec=0)
        except KeyboardInterrupt:
            print('shutting down')
        rclpy.shutdown()


def main(args=sys.argv):
    rclpy.init(args=args)
    node = Hexapod(args)
    signal.signal(signal.SIGINT, node.ctrl_c)
    node.clear_trajectory()
    #node.stand_and_sit()
    #node.stand_up()
    #node.walk(math.pi/2, 0.5)
    node.set_gait(node.standing_gait)
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
