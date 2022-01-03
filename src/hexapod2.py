import sys
import signal
import rclpy
import time
import math
import typing
from functools import reduce

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Vector3
from robot_model_msgs.msg import MultiSegmentTrajectory, ModelState, ControlState, Limb, \
    MultiTrajectoryProgress, TrajectoryProgress, TrajectoryComplete, SegmentTrajectory
from robot_model_msgs.action import EffectorTrajectory

from scipy.spatial import ConvexHull

from robot import RobotState
from ros_trajectory_builder import default_segment_trajectory_msg
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from tf_conv import to_kdl_rotation, to_kdl_vector, to_kdl_frame, to_vector3, to_transform, to_quaternion, P, R

from polar import PolarCoord
from leg import Leg, each_leg, tripod_set


DynamicObject = lambda **kwargs: type("Object", (), kwargs)


class Task:
    on_init = None
    on_tick = None
    on_finish = None

    PENDING = 0
    RUNNING = 1
    DONE = 10

    state = 0

    def __init__(self, **kwargs):
        if 'init' in kwargs:
            self.on_init = kwargs['init']
        if 'tick' in kwargs:
            self.on_tick = kwargs['tick']
        if 'finish' in kwargs:
            self.on_finish = kwargs['finish']


class WaitTask(Task):
    delay = 0
    expires = None

    def __init__(self, seconds: float, **kwargs):
        self.delay = seconds
        self.expires = 0
        self.user_init = kwargs['init'] if 'init' in kwargs else None
        self.user_tick = kwargs['tick'] if 'tick' in kwargs else None
        kwargs['tick'] = self.tick
        kwargs['init'] = self.init
        super().__init__(**kwargs)

    def init(self):
        self.expires = time.time() + self.delay
        if self.user_init:
            self.user_init()

    def tick(self):
        if time.time() > self.expires:
            self.state = Task.DONE
        elif self.user_tick:
            self.user_tick()


class Hexapod(Node):
    # gait states
    IDLE = 0
    STANDING = 1
    WALKING = 2
    TURNING = 3

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

    heading = 0
    target_heading = math.inf

    urdf = None
    robot = None

    # odom => base transform
    odom: kdl.Frame

    # the current pose of the base
    base_pose: kdl.Frame
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

    def __init__(self, args):
        super().__init__('hexapod')
        self.shutdown = False

        self.state = RobotState()
        self.tasks = []
        self.support_margin = 0.

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

        # pull TF so we can figure out where our segments are,
        # and their relative positions
        #self.tf_state_sub = self.create_subscription(
        #    TFMessage,
        #    '/tf',
        #    self.tf_callback,
        #    reliable_profile)
        #self.tf_static_state_sub = self.create_subscription(
        #    TFMessage,
        #    '/tf_static',
        #    self.tf_callback,
        #    reliable_profile)

        # subscribe to model state
        self.model_state_sub = self.create_subscription(
            ModelState,
            '/robot_dynamics/model_state',
            self.model_state_callback,
            reliable_profile)

        # subscribe to model control state
        self.control_state_sub = self.create_subscription(
            ControlState,
            '/robot_dynamics/control_state',
            self.control_state_callback,
            reliable_profile)

        # create publisher for enabling/disabling joints
        self.effort_pub = self.create_publisher(
            Float64MultiArray,
            "/effort_controller/commands",
            best_effort_profile)

        # create publisher for segment trajectories
        self.trajectory_pub = self.create_publisher(
            MultiSegmentTrajectory,
            "/robot_dynamics/trajectory",
            reliable_profile)

        self.trajectory_client = ActionClient(
            self,
            EffectorTrajectory,
            '/robot_dynamics/trajectory')

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

            print(f'  limb: {leg} => {hip_tf.p}  yaw: {hip_yaw * 180 / math.pi}')

        print('created legs from URDF')

    def tf_callback(self, msg):
        # we should no longer need to subscribe to this
        if self.legs:
            return

        for tf in msg.transforms:
            preview = False
            orig_child_frame = tf.child_frame_id
            if self.previewMode:
                if tf.child_frame_id.startswith('preview/'):
                    preview = True
                    tf.child_frame_id = tf.child_frame_id[8:]
                if tf.header.frame_id.startswith('preview/'):
                    tf.header.frame_id = tf.header.frame_id[8:]

            if tf.child_frame_id in self.state.frame_map:
                # existing object
                frame = self.state.frame_map[tf.child_frame_id]
                if not self.previewMode or preview:
                    frame.parent = tf.header.frame_id
                    frame.preview = preview
                    frame.transform = to_kdl_frame(tf.transform)
                    #if tf.child_frame_id == self.base_link:
                    #    print(f'BASE {orig_child_frame}  {frame.transform.p[0]:6.4f} {frame.transform.p[1]:6.4f}')
            else:
                # new frame
                self.state.frame_map[tf.child_frame_id] = DynamicObject(
                    parent=tf.header.frame_id,
                    preview=preview,
                    transform=to_kdl_frame(tf.transform)
                )

        if not self.legs:
            # try gathering the data we need for each leg
            # we need to receive the origin of each hip and the location of each leg effector
            legs = dict()
            # for each leg - left-middle-hip-span1
            for leg_prefix in each_leg():
                leg_hip = leg_prefix+'-hip-span1'
                leg_effector = leg_prefix+'-foot'
                base_odom = self.state.get_frame_rel(self.base_link, self.odom_link)
                hip_base = self.state.get_frame_rel(leg_hip)
                bottom_plate = self.state.get_frame_rel("bottom_plate")
                effector_base = self.state.get_frame_rel(leg_effector)
                if hip_base and effector_base:
                    rot = kdl.Rotation()
                    #hip_yaw = hip_base.M.GetRPY()
                    #hip_yaw = hip_yaw[2]
                    hip_odom = base_odom * hip_base
                    #hip_base.p[2] = hip_odom.p[2]     # move Z origin of leg to Z origin of odom frame
                    hip_base.p[2] = bottom_plate.p[2]     # move Z origin of leg to Z origin of base_link
                    hip_yaw = math.atan2(hip_base.p[1], hip_base.p[0])
                    rot = rot.RPY(0.0, 0.0, hip_yaw)
                    print(f'  yaw: {leg_prefix} => {hip_yaw*180/math.pi}')
                    leg = legs[leg_prefix] = Leg(
                        name=leg_prefix,
                        origin=kdl.Frame(
                            rot,
                            hip_base.p
                        )
                    )
                    leg.origin_angle = hip_yaw
                    if 'front' in leg.name:
                        leg.neutral_angle = self.neutral_offset
                    elif 'back' in leg.name:
                        leg.neutral_angle = -self.neutral_offset

                    print(f'  hip {leg.name}: Y{hip_yaw}  A{leg.origin_angle*180/math.pi:4.2f} {hip_base.p}')

            # only accept the data if we found all 6 leg transforms
            if len(legs) == 6:
                # we have the hip data, now try determining the current leg state
                for l_name in each_leg('right'):
                    legs[l_name].reverse = True
                    print(f'reversing {l_name}')
                self.legs = legs

                # no need to subscribe to TF anymore
                self.tf_state_sub = None
                self.tf_static_state_sub = None

        if self.legs:
            # update base pose
            self.odom = self.state.get_frame(self.odom_link)
            self.base_pose = self.state.get_frame(self.base_link)

            # get the robot heading
            rz, ry, rx = self.base_pose.M.GetEulerZYX()
            self.heading = rz
            #if math.isinf(self.target_heading):
            #    self.target_heading = self.heading

            # update the Leg positions
            for leg in self.legs.values():
                leg.rect = self.state.get_frame_rel(leg.foot_link)
                polar = leg.to_polar(leg.rect)
                rect = leg.to_rect(polar)
                #if leg.rect != rect:
                #    print('not equal')
                #foot_hip = self.state.get_frame_rel(leg.foot_link, leg.hip_link)
                #if foot_hip:
                #    #leg.rect = foot_hip
                #    polar = PolarCoord.from_rect(foot_hip.p[0], foot_hip.p[1], foot_hip.p[2])
                #    polar.angle += math.pi
                #    if not leg.polar == polar:
                #        print(f"   {leg.name} => {polar}")
                #    leg.polar = polar

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
            polar = leg.to_polar(leg.rect)
            rect = leg.to_rect(polar)

            polar2 = leg.to_polar(leg.rect, False)
            rect2 = leg.to_rect(polar2, False)
            if leg_name == 'left-front':
                print(f'  P{polar}   R{rect}    O{leg.rect.p}')


    def enable_motors(self):
        return False
        print("enable motors")
        # turn the motors on or off by sending efforts directly to ros2 control
        msg = Float64MultiArray()
        msg.data = [1.0] * 18
        self.effort_pub.publish(msg)

    def disable_motors(self):
        print("disable motors")
        # turn the motors on or off by sending efforts directly to ros2 control
        msg = Float64MultiArray()
        msg.data = [0.0] * 18
        self.effort_pub.publish(msg)

    def clear_trajectory(self):
        msj = MultiSegmentTrajectory()
        msj.header.stamp = self.get_clock().now().to_msg()
        msj.header.frame_id = 'odom'
        msj.mode = MultiSegmentTrajectory.REPLACE_ALL
        self.trajectory_pub.publish(msj)

    def trajectory(self, goal: SegmentTrajectory,
                   complete=typing.Callable,
                   progress: typing.Callable = None,
                   rejected: typing.Callable = None):
        request = EffectorTrajectory.Goal()
        request.goal.header.stamp = self.get_clock().now().to_msg()
        request.goal.header.frame_id = goal.reference_frame
        request.goal.segment = goal

        def handle_result(future):
            result = future.result()
            if complete:
                complete(result.result.result)

        def handle_response(future):
            goal_h = future.result()
            if goal_h.accepted:
                result_future = goal_h.get_result_async()
                result_future.add_done_callback(handle_result)
            else:
                print(f'trajectory request for {goal.segment} rejected')
                if rejected:
                    rejected()

        def feedback_callback(feedback_msg):
            if progress:
                progress(feedback_msg.feedback.progress)

        self.trajectory_client.wait_for_server()
        goal_handle = self.trajectory_client.send_goal_async(request, feedback_callback=feedback_callback)
        goal_handle.add_done_callback(handle_response)

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
            rot = self.base_pose.M
        elif reference_frame == self.odom_link:
            rot = kdl.Rotation()

        traj = default_segment_trajectory_msg(
            leg.foot_link,
            id='move-leg',
            velocity=self.walking_gait_velocity,
            reference_frame=reference_frame,
            points=[to_vector3(point)],
            rotations=[to_quaternion(rot)])

        leg.state = Leg.LIFTING
        print(f'leg lift: {traj.segment}')
        self.trajectory(traj, complete=on_complete, **kwargs)

    # change the leg to a supportive mode
    # if point is a kdl.Frame it must be relative to the odom frame. If no point is
    # given then the current position as recorded by tf state is used.
    # todo: this shouldnt be used not since we dont switch reference frames anymore
    def support_leg(self, leg: Leg or str, point: (PolarCoord or kdl.Frame) = None):
        if not point:
            point = self.state.get_frame_rel(leg.foot_link, relative_to=self.odom_link)
        elif isinstance(point, PolarCoord):
            #point = kdl.Frame(self.base_pose.M, leg.to_rect(point, self.base_pose))
            point = leg.to_rect(point)
            point = self.base_pose * point
            #point = kdl.Frame(self.base_pose.M, leg.to_rect(point, self.base_pose))
        # resolve a support trajectory
        leg.state = Leg.SUPPORTING
        #point.p[2] = -self.base_pose.p[2] # - self.base_standing_z
        #-self.base_standing_z
        #leg.rect = self.base_pose.Inverse() * point     # store the updated rect relative to base
        support_mode = default_segment_trajectory_msg(
            leg.foot_link,
            id='support-leg',
            velocity=self.walking_gait_velocity,
            reference_frame=self.odom_link,
            points=[to_vector3(point.p)],
            rotations=[to_quaternion(point.M)])
        self.transmit_trajectory([support_mode])

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
        self.trajectory(traj, complete=complete)

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
            reference_frame=self.base_link,
            points=[P(target_pos[0], target_pos[1], leg.origin.p[2] - self.base_standing_z)],
            rotations=[to_quaternion(leg.rect.M)])
        self.trajectory(support_mode, complete=complete)

    def transmit_trajectory(self, tsegs, transform: kdl.Frame = None, mode=MultiSegmentTrajectory.INSERT):
        if type(tsegs) != list:
            tsegs = [tsegs]
        if transform:
            for t in tsegs:
                t.points = [transform * p for p in t.points]
        msj = MultiSegmentTrajectory()
        msj.header.stamp = self.get_clock().now().to_msg()
        msj.header.frame_id = 'odom'
        msj.mode = mode
        msj.segments = tsegs
        self.trajectory_pub.publish(msj)

    def tripod_gait(self):
        if self.base_pose and self.support_margin:
            # get leg sets
            non_supporting = [l for l in self.legs.values() if
                              l.name not in self.tripod_set[self.tripod_set_supporting]]
            supporting = [l for l in self.legs.values() if
                              l.name in self.tripod_set[self.tripod_set_supporting]]

            supporting_state = list(set([l.state for l in supporting]))
            non_supporting_state = list(set([l.state for l in non_supporting]))

            # if stability margin threshold is met, switch sets
            # if self.support_margin < 0.01:

            if self.gait_state == Hexapod.IDLE:
                print("starting tripod gait")
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
                    self.gait_state = Hexapod.WALKING

            elif self.gait_state == Hexapod.WALKING:
                print("begin walking")
                # if non-supporting set is now supportive, then
                # lift change the supporting set to lift
                non_supportive_is_supporting = reduce(
                    lambda a, b: a and b,
                    [l.state == Leg.SUPPORTING for l in non_supporting]
                )

                if len(non_supporting_state)==1 and non_supporting_state[0] == Leg.SUPPORTING \
                    and len(supporting_state)==1 and supporting_state[0] == Leg.SUPPORTING:
                    # swap the supporting set
                    self.tripod_set_supporting = (self.tripod_set_supporting + 1) % 2

                    # perform a leg lift on the supportive set
                    for leg in supporting:
                        to = PolarCoord(
                            angle=-0.4 + leg.neutral_angle,
                            distance=self.neutral_radius,
                            z=-self.base_standing_z)
                        #leg.state = Leg.LIFTING
                        #if leg.name == 'left-front':
                        self.lift_leg(leg, to)

                    # perform stance move on what is now the new supporting set
                    # first calculate what distance we will move by calculating possible target positions for each foot
                    targets = []
                    for leg in non_supporting:
                        foot = leg.rect
                        dest = leg.to_rect(PolarCoord(angle=0.4, distance=0.12))
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
                        self.stance_leg(leg, to=kdl.Vector(target_polar.x, target_polar.y, 0.))

    def test_gait(self):
        if not hasattr(self, 'gait_points'):
            self.gait_points = [
                PolarCoord(angle=0.6, distance=0.15, z=-self.base_standing_z),
                PolarCoord(angle=0.0, distance=0.15, z=-self.base_standing_z),
                PolarCoord(angle=-0.6, distance=0.15, z=-self.base_standing_z)
            ]
            self.gait_n = len(self.gait_points)

        def next_gait(r: TrajectoryComplete):
            if self.gait_n + 1 < len(self.gait_points):
                self.gait_n = self.gait_n + 1
                self.lift_leg(leg, self.gait_points[self.gait_n], on_complete=next_gait)
            else:
                # wrap around, do a stance move
                self.stance_leg(leg)

        if self.base_pose and self.support_margin:
            leg = self.legs['left-middle']

            if self.gait_state == Hexapod.IDLE:
                self.gait_state = Hexapod.WALKING
                for l in self.legs.values():
                    # move any IDLE legs to standing state
                    if leg == l:
                        next_gait(None)
                    else:
                        self.move_leg_to(
                            leg=l,
                            point=PolarCoord(l.neutral_angle, self.neutral_radius, z=-self.base_standing_z),
                            reference_frame=self.base_link,
                            complete=l.SUPPORTING)
            else:
                polar = leg.polar()
                if leg.state == Leg.SUPPORTING and polar.angle > 0.6:
                    self.gait_n = 0
                    next_gait(None)

    def stand(self):
        print("stand")
        t = default_segment_trajectory_msg(
            self.base_link,
            reference_frame='odom')
        t.points = [
            P(0.0, 0.0, self.base_standing_z)
        ]
        self.transmit_trajectory(t)

    def sit(self):
        print("sit")
        t = default_segment_trajectory_msg(
            self.base_link,
            reference_frame='odom')
        t.points = [
            P(0.0, 0.0, self.base_sitting_z)
        ]
        self.transmit_trajectory(t)

    def move_leg(self, name: str, polar: PolarCoord, relative_frame: str):
        if not relative_frame:
            relative_frame = self.base_link
        def leg_movement():
            if name not in self.legs:
                return False
            rel_frame = self.base_pose if relative_frame == self.odom_link else kdl.Frame()
            leg = self.legs[name]
            #p = PolarCoord(
            #    origin=kdl.Vector(
            #        self.base_pose.p[0] + leg.origin.p[0],
            #        self.base_pose.p[1] + leg.origin.p[1], 0),
            #    angle=leg.origin_angle + (-polar.angle if leg.reverse else polar.angle),
            #    distance=polar.distance,
            #    z=polar.z
            #)
            fr = leg.to_rect(polar, rel_frame)
            #print(f'leg {name} to {p.x:6.2f}, {p.y:6.2f}, {p.z:6.2f}  {p.angle:6.2f} @ {p.distance:6.2f}')
            print(f'leg {name} to {fr}')
            t = default_segment_trajectory_msg(
                leg.foot_link,
                reference_frame=relative_frame,
                points=[to_vector3(fr)],
                rotations=[R(0., 0., 0., 1.)])
            self.transmit_trajectory(t)
        return leg_movement

    def move_legs_local(self, to: PolarCoord, legs: typing.Iterable = None):
        def local_leg_movement():
            trajectories = list()
            print("move legs local")
            for leg in self.resolve_legs(legs if legs else each_leg()):
                fr = leg.to_rect(to, use_neutral=True)
                #fr2 = leg.to_rect(adjusted_to, self.base_pose)
                #print(f'leg {leg.name} to {foot_base.p[0]}, {foot_base.p[1]}, {foot_base.p[2]}')
                t = default_segment_trajectory_msg(
                    leg.foot_link,
                    id='move_legs_local',
                    reference_frame=self.base_link,
                    points=[to_vector3(fr)],
                    rotations=[R(0., 0., 0., 1.)])
                trajectories.append(t)
            self.transmit_trajectory(trajectories)
        return local_leg_movement

    def move_robot(self):
        # self.transmit_trajectory(1.0, 1.5)
        if not self.legs:
            return          # wait for all the meta information we need

        if self.gait:
            self.gait()

        self.timestep += 0.1
        if self.tasks and len(self.tasks):
            task = self.tasks[0]
            if task.state == Task.PENDING:
                task.state = Task.RUNNING
                if task.on_init:
                    task.on_init()
            elif task.state == Task.RUNNING:
                if task.on_tick:
                    task.on_tick()

            # check if task is done
            if task.state == Task.DONE or (task.expires and 0 < task.expires < time.time()):
                # done, remove the task
                self.tasks = self.tasks[1:]
                if task.on_finish:
                    task.on_finish()
                #if len(self.tasks) == 0:
                #    print("done")
                #    #self.executor.shutdown(timeout_sec=0)
                #    #self.destroy_node()
                #    rclpy.shutdown(context=self.context)
                #    exit(0)

    def stand_and_sit(self):
        self.tasks = [
            #WaitTask(0.5, init=self.enable_motors),
            WaitTask(2.0, init=self.stand),
            WaitTask(2.0),
            WaitTask(2.0, init=self.sit),
            #WaitTask(2.5, init=self.disable_motors),
        ]

    def stand_up(self):
        self.tasks.extend([
            #WaitTask(0.5, init=self.enable_motors),
            WaitTask(1.7, init=self.move_legs_local(PolarCoord(0.0, 0.15, 0.02))),
            WaitTask(2.4, init=self.move_legs_local(PolarCoord(0.0, 0.13, -self.base_standing_z)))
            #WaitTask(1.0, init=self.stand),
            #WaitTask(2.0, init=self.move_legs_local(PolarCoord(0.0, 0.15, -self.base_standing_z))),
            #WaitTask(2.5, init=self.disable_motors)
        ])

    def sit_down(self):
        self.tasks.append(
            WaitTask(2.0, init=self.sit),
        )

    def enter_preview_mode(self):
        def enter_preview():
            self.previewMode = True
            print('entering preview mode')
        self.tasks.append(WaitTask(0.5, init=enter_preview))

    def enable_gait(self, gait: typing.Callable):
        def start_gait():
            self.gait = gait
        return WaitTask(2.0, init=start_gait)

    def walk(self, heading: float, distance: float):
        self.tasks.extend([
            #WaitTask(2.0, init=self.move_base(heading=heading, distance=distance)),
            self.enable_gait(gait=self.tripod_gait)
        ])

    def step(self, name: str, to: PolarCoord):
        def local_movement():
            leg = self.legs[name]
            base_tf = self.state.get_frame_rel(self.base_link, self.odom_link)
            traj = leg.lift(to, velocity=self.walking_gait_velocity)
            self.transmit_trajectory([traj])
        return local_movement

    def merge_trajectory_test(self):
        self.tasks = [
            WaitTask(0.5, init=self.enable_motors)
        ]
        for leg in each_leg():
            self.tasks.append(WaitTask(0.5, init=self.move_legs_local(PolarCoord(0, 0.1, 0.1), [leg])))
        self.tasks.append(WaitTask(2.0))
        self.tasks.append(WaitTask(3.0, init=self.move_legs_local(PolarCoord(0.0, 0.13, -0.02))))
        self.tasks.append(WaitTask(2.5, init=self.disable_motors))

    def ctrl_c(self, signum, frame):
        print(f'shutdown requested by {signum}')
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
    node.enter_preview_mode()
    #node.stand_and_sit()
    node.stand_up()
    #node.merge_trajectory_test()
    #node.trajectory_test()
    node.walk(math.pi/2, 0.5)
    #node.tasks.append(node.enable_gait(node.test_gait))
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
