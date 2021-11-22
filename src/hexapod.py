import sys
import rclpy
import time
import math
import typing
import functools

from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Vector3
from humanoid_model_msgs.msg import MultiSegmentTrajectory, SegmentTrajectory
from humanoid_model_msgs.msg import ModelState

#from ros_trajectory_builder import TrajectoryBuilder, Event
import PyKDL as kdl
from tf_conv import to_kdl_rotation, to_kdl_vector, to_kdl_frame, to_vector3, to_transform, to_quaternion, P, R

DynamicObject = lambda **kwargs: type("Object", (), kwargs)


def default_segment_trajectory_msg(
        segment: str, start: float = 0.0, velocity: float = 1.0, reference_frame: str = 'base_link', points: [] = None,
        rotations: [] = None):
    t = SegmentTrajectory()
    t.start.sec = int(start)
    t.start.nanosec = int((start - t.start.sec) * 1000000000)
    t.segment = segment
    t.profile = 'velocity/trap'
    t.velocity = velocity
    t.acceleration = 0.1
    t.path = 'rounded'
    t.reference_frame = reference_frame
    t.coordinate_mode = 0
    # if points:
    #    t.points = [to_vector3(f) for f in points]
    # if rotations:
    #    t.rotations = [to_quaternion(r) for r in rotations]
    if points:
        t.points = points
    if rotations:
        t.rotations = rotations
    return t


class PolarCoord:
    origin: kdl.Vector
    angle: float
    distance: float
    zlocal: float           # z is optional and part of the rectangular coordinate system

    def __init__(self, angle: float, distance: float, z: float = 0.0, origin=None):
        self.origin = origin if origin else kdl.Vector(0, 0, 0)
        self.angle = angle
        self.distance = distance
        self.zlocal = z

    def __eq__(self, other):
        return (math.isclose(self.angle, other.angle, rel_tol=0.001) and
                math.isclose(self.distance, other.distance, rel_tol=0.000001) and
                math.isclose(self.z, self.z, rel_tol=0.000001)) if isinstance(other, PolarCoord) else False

    def __neq__(self, other):
        return (not math.isclose(self.angle, other.angle, rel_tol=0.001) or
                not math.isclose(self.distance, other.distance, rel_tol=0.000001) or
                not math.isclose(self.z, self.z, rel_tol=0.000001)) if isinstance(other, PolarCoord) else True


    def __str__(self):
        return f'{self.angle*180/math.pi:4.1f}° {self.distance:6.4f}m Z{self.z}'

    @staticmethod
    def from_rect(x: float, y: float, z: float = 0.0):
        return PolarCoord(
            angle=math.atan2(y, x),
            distance=math.sqrt(x * x + y * y),
            z=z
        )

    @property
    def x(self):
        return self.origin[0] + self.distance * math.cos(self.angle)

    @property
    def y(self):
        return self.origin[1] + self.distance * math.sin(self.angle)

    @property
    def z(self):
        return self.origin[2] + self.zlocal

    def to_xyz(self):
        return self.x, self.y, self.z

    def to_kdl_vector(self):
        return kdl.Vector(x=self.x, y=self.y, z=self.z)

    def to_vector3(self):
        return Vector3(x=self.x, y=self.y, z=self.z)


class Tween:
    begin = 0.0
    end = 1.0
    type = float

    def __init__(self, begin, end):
        if type(begin) != type(end):
            raise TypeError("tweening requires begin and end arguments to be the same type")
        self.begin = begin
        self.end = end
        self.type = type(self.begin)
        if self.type == float:
            self.get = functools.partial(Tween.tween_float, begin, end)
        elif self.type == PolarCoord:
            self.get = functools.partial(Tween.tween_polar, begin, end)
        elif self.type == kdl.Vector:
            self.get = functools.partial(Tween.tween_kdl_vector, begin, end)
        elif self.type == Vector3:
            self.get = functools.partial(Tween.tween_ros_vector, begin, end)
        else:
            raise ValueError("type not supported for tweening "+(str(self.type)))

    @staticmethod
    def tween_float(begin: float, end: float, p: float):
        return begin + (end - begin) * p

    @staticmethod
    def tween_polar(begin: PolarCoord, end: PolarCoord, p: float):
        return PolarCoord(
            angle=Tween.tween_float(begin.angle, end.angle, p),
            distance=Tween.tween_float(begin.distance, end.distance, p),
            z=Tween.tween_float(begin.z, end.z, p))

    @staticmethod
    def tween_kdl_vector(begin: kdl.Vector, end: kdl.Vector, p: float):
        return kdl.Vector(
            x=Tween.tween_float(begin[0], end[0], p),
            y=Tween.tween_float(begin[1], end[1], p),
            z=Tween.tween_float(begin[2], end[2], p))

    @staticmethod
    def tween_ros_vector(begin: Vector3, end: Vector3, p: float):
        return Vector3(
            x=Tween.tween_float(begin.x, end.x, p),
            y=Tween.tween_float(begin.y, end.y, p),
            z=Tween.tween_float(begin.z, end.z, p))


class Leg:
    IDLE = 0
    LIFTING = 1
    SUPPORTING = 2

    name: str = None
    foot_link: str = None
    hip_link: str = None

    state = IDLE

    # polar coordinate system centered around the hip joint
    origin: kdl.Frame

    # this is the angle opposite to the vector from hip origin to base center
    # all angles will be offset from this angle, including the neutral_angle offset
    origin_angle = 0.0

    # if set, the polarity of this leg is reversed (left legs in our case)
    reverse = False

    # neutral pose is the spider's resting position
    neutral_angle = 0.0       # offset angle from hip's URDF default yaw (can make the spider's form wide or compact)
    neutral_distance = 0.1    # distance from hip joint (center of polar coord)

    # walking limits
    lift_max = 0.04
    walking_min = -0.1
    walking_max = -0.1

    # relative to base_link
    rect: kdl.Frame = None

    def __init__(self, name: str, origin: kdl.Frame):
        self.name = name
        self.foot_link = name + '-foot'
        self.hip_link = name + '-hip-span1'
        self.origin = origin

        # compute the origin angle of the leg (it's neutral heading)
        self.origin_angle = math.atan2(origin.p[1], origin.p[0])

    def to_polar(self, rect: kdl.Vector, base_pose: kdl.Frame = None) -> PolarCoord:
        if not base_pose:
            base_pose = kdl.Frame()
        if not rect:
            rect = self.rect
        if isinstance(rect, kdl.Vector):
            rect = kdl.Frame(rect)
        # get hip location in odom space
        hip_odom = base_pose * self.origin
        #hip_odom.p[2] = 0
        # get foot location in odom space
        foot_odom = base_pose * rect
        # now get the final
        xx = foot_odom * hip_odom
        xx = hip_odom.Inverse() * foot_odom
        #xx = kdl.Frame(rect).Inverse() * base_pose * self.origin
        coord = PolarCoord.from_rect(xx.p[0], xx.p[1], xx.p[2])
        if self.reverse:
            coord.angle *= -1
        #coord.angle -= self.origin_angle
        return coord

    # return the rectangular coordinates from the polar coordinates relative to the base_link
    def to_rect(self, coord: PolarCoord, base_pose: kdl.Frame = None) -> kdl.Vector:
        if not base_pose:
            base_pose = kdl.Frame()
        if self.reverse:
            coord = PolarCoord(angle=-coord.angle, distance=coord.distance, z=coord.z)
        #coord2 = PolarCoord(angle=coord.angle + self.origin_angle, distance=coord.distance, z=coord.z)
        rect = coord.to_kdl_vector()
        # reverse coordinate system
        #rect.p[0] = -rect.p[0]
        #rect.p[1] = -rect.p[1]

        hip_odom = base_pose * self.origin
        #hip_odom.p[2] = 0
        xx = hip_odom * rect

        # todo: calculate opposite direction then inverse
        #xx = rect.Inverse() * self.origin.Inverse() * base_pose.Inverse()

        # transform to be relative to base_link
        xx_base = base_pose.Inverse() * xx
        coord_check = self.to_polar(xx_base, base_pose)

        return xx_base

    def polar(self, base_pose: kdl.Frame = None) -> PolarCoord:
        return self.to_polar(self.rect.p, base_pose)

    def lift(self, polar: PolarCoord, base_pose: kdl.Frame, base_velocity: kdl.Vector):
        # how long will it take our leg to complete the trajectory?
        duration = 2.0

        fr = self.polar(base_pose)
        #print(f'lift leg {self.name} from {fr} => {polar}')
        tw_dist = Tween(fr, polar)

        # a helper function to tween between current leg position and target position [0 - 1.]
        def get_point(p: float):
            # lift leg in a semicircle as we proceed to target
            v: PolarCoord = tw_dist.get(p)

            # where will base be when our leg lands?
            base_future = kdl.Frame(base_pose)
            point_vel = base_velocity * (duration * p)
            base_future.p += point_vel

            # transform from polar to rectangular in odom frame
            r = self.to_rect(v, base_future)

            # add in Z lift as an arc
            r[2] += self.lift_max * math.sin(p * math.pi)
            #print(f'   {p:2.1f} => {v} => {r[0]} {r[1]} {r[2]}')
            fo = base_future * r
            return to_vector3(fo)

        return default_segment_trajectory_msg(
            self.foot_link,
            #velocity=2.0,
            reference_frame='odom',
            points=[get_point(d/10) for d in range(1, 11)],
            rotations=[R(0., 0., 0., 1.)])


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


#
# Generator expression for enumerating legs
#
# example:
#     for leg in each_leg(['left','right'], ['middle'], suffix='foot):
#         # generates [left-middle-foot, right-middle-foot]
#
def each_leg(sides=None, ordinals=None,
             prefix: typing.Optional[str] = None, suffix: typing.Optional[str] = None,
             separator: str = '-'):
    if ordinals is None:
        ordinals = ['front', 'middle', 'back']
    elif type(ordinals) == str:
        ordinals = [ordinals]
    if sides is None:
        sides = ['left', 'right']
    elif type(sides) == str:
        sides = [sides]
    for s in sides:
        for o in ordinals:
            yield separator.join(filter(None, [prefix, s, o, suffix]))


def tripod_set(ordinal: int,
             prefix: typing.Optional[str] = None, suffix: typing.Optional[str] = None,
             separator: str = '-'):
    if ordinal == 0:
        legs = ['left-front', 'right-middle', 'left-back']
    elif ordinal == 1:
        legs = ['right-front', 'left-middle', 'right-back']
    for l_name in legs:
        yield separator.join(filter(None, [prefix, l_name, suffix]))


class Hexapod(Node):
    frame_names: list = []
    frame_map: dict = {}
    model_state = None
    tasks = None

    previewMode = False
    # preview_prefix - make it so we just set this to the prefix we want to base our trajectories on

    # segment names
    base_link = 'base_link'
    odom_link = 'odom'

    base_standing_z = 0.06
    base_sitting_z = 0.042

    heading = 0

    # odom => base transform
    odom: kdl.Frame

    # the current pose of the base
    base_pose: kdl.Frame

    # the XYZ location of each hip joint
    # we'll form a polar coordinate system centered on each hip to plan leg movements
    legs: dict = None

    gait: typing.Callable = None
    neutral_offset = 0.5

    timestep = 0

    def __init__(self, args):
        super().__init__('hexapod')

        self.tasks = []

        best_effort_profile = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE)

        reliable_profile = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE)

        transient_local_reliable_profile = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)

        # pull TF so we can figure out where our segments are,
        # and their relative positions
        self.model_state_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            reliable_profile)
        self.model_state_sub = self.create_subscription(
            TFMessage,
            '/tf_static',
            self.tf_callback,
            reliable_profile)

        # subscribe to model state
        self.model_state_sub = self.create_subscription(
            ModelState,
            '/robot_dynamics/model_state',
            self.model_state_callback,
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

        self.move_timer_ = self.create_timer(
            0.1, self.move_robot)

    def resolve_legs(self, legs: typing.Iterable):
        # resolve an array of leg names to an array leg objects
        return [self.legs[leg] for leg in legs if leg in self.legs]

    def get_frame_rel(self, segment: str, relative_to='base_link'):
        if segment in self.frame_map:
            frame = self.frame_map[segment]
            if frame.parent == relative_to:
                # we hit the last parent frame
                return frame.transform
            else:
                # we have to go back further, so recurse
                parent_transform = self.get_frame_rel(frame.parent, relative_to)
                return parent_transform * frame.transform if parent_transform else None
        return None

    def get_frame(self, segment: str):
        return self.frame_map[segment].transform if segment in self.frame_map else None

    def tf_callback(self, msg):
        for tf in msg.transforms:
            preview = False
            orig_child_frame = tf.child_frame_id
            if self.previewMode:
                if tf.child_frame_id.startswith('preview/'):
                    preview = True
                    tf.child_frame_id = tf.child_frame_id[8:]
                if tf.header.frame_id.startswith('preview/'):
                    tf.header.frame_id = tf.header.frame_id[8:]

            if tf.child_frame_id in self.frame_map:
                # existing object
                frame = self.frame_map[tf.child_frame_id]
                if not self.previewMode or preview:
                    frame.parent = tf.header.frame_id
                    frame.preview = preview
                    frame.transform = to_kdl_frame(tf.transform)
                    #if tf.child_frame_id == self.base_link:
                    #    print(f'BASE {orig_child_frame}  {frame.transform.p[0]:6.4f} {frame.transform.p[1]:6.4f}')
            else:
                # new frame
                self.frame_map[tf.child_frame_id] = DynamicObject(
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
                base_odom = self.get_frame_rel(self.base_link, self.odom_link)
                hip_base = self.get_frame_rel(leg_hip)
                effector_base = self.get_frame_rel(leg_effector)
                if hip_base and effector_base:
                    rot = kdl.Rotation()
                    #hip_yaw = hip_base.M.GetRPY()
                    #hip_yaw = hip_yaw[2]
                    hip_odom = base_odom * hip_base
                    #hip_base.p[2] = hip_odom.p[2]     # move Z origin of leg to Z origin of odom frame
                    hip_base.p[2] = 0     # move Z origin of leg to Z origin of base_link
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

        if self.legs:
            # update base pose
            self.odom = self.get_frame(self.odom_link)
            self.base_pose = self.get_frame(self.base_link)

            # update the Leg positions
            for leg in self.legs.values():
                leg.rect = self.get_frame_rel(leg.foot_link)
                polar = leg.to_polar(leg.rect, self.base_pose)
                rect = leg.to_rect(polar, self.base_pose)
                #if leg.rect != rect:
                #    print('not equal')
                #foot_hip = self.get_frame_rel(leg.foot_link, leg.hip_link)
                #if foot_hip:
                #    #leg.rect = foot_hip
                #    polar = PolarCoord.from_rect(foot_hip.p[0], foot_hip.p[1], foot_hip.p[2])
                #    polar.angle += math.pi
                #    if not leg.polar == polar:
                #        print(f"   {leg.name} => {polar}")
                #    leg.polar = polar

    def model_state_callback(self, msg):
        # simply store model state until next IMU message
        self.model_state = msg

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
        if self.base_pose:
            velocity = 0.02
            base_velocity = kdl.Vector(
                velocity * math.cos(self.heading),
                velocity * math.sin(self.heading),
                0)

            for name, leg in self.legs.items():
                polar: PolarCoord = leg.polar(self.base_pose)
                polar.zlocal += self.base_standing_z
                out_of_range = (polar.angle > (0.5 + leg.neutral_angle) or polar.distance > 0.15)
                # todo: because of front/back non-symettry
                #     [1] the base encroaches on the front legs causing them to sort of run over them
                #     [2] the base runs from the back legs causing them to re-send a trajectory on the
                #         leg apex because the leg is still too far behind
                #     * have to predict where the base will be 2 seconds ahead, and set XY to that
                if leg.state == Leg.LIFTING and polar.z > leg.lift_max * 0.75:
                   leg.state = Leg.IDLE
                   print(f'   leg {leg.name} => {polar.z} | {leg.lift_max} | {out_of_range}')
                elif leg.state != Leg.LIFTING and out_of_range:
                    # lift this leg
                    leg.state = Leg.LIFTING

                    to = PolarCoord(
                        angle=-0.6 + leg.neutral_angle,
                        distance=0.128,
                        z=-0.04)

                    print(f'   leg {leg.name} => {to.x:6.4f}, {to.y:6.4f}, {to.z:6.4f}')
                    self.transmit_trajectory(
                        [leg.lift(to, base_pose=self.base_pose, base_velocity=base_velocity)])


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

    def move_base(self, heading: float, distance: float):
        dest = PolarCoord(angle=heading, distance=distance, z=self.base_standing_z)
        def base_movement():
            self.heading = heading
            print(f'base to {dest.x}, {dest.y}    heading={heading * 180 / math.pi}')
            t = default_segment_trajectory_msg(
                self.base_link,
                reference_frame='odom',
                velocity=0.02,
                points=[P(dest.x, dest.y, dest.z)],
                rotations=[R(0., 0., 0., 1.)])
            self.transmit_trajectory(t)
        return base_movement

    def move_heading(self, heading: float):
        # a simple function for moving towards a heading
        # the distance is chosen so that if we stop getting input then the robot will
        # naturally slow to a stop over a short distance
        self.heading = heading
        return self.move_base(heading, 0.1)

    def move_leg(self, name: str, polar: PolarCoord):
        def leg_movement():
            if name not in self.legs:
                return False
            leg = self.legs[name]
            #p = PolarCoord(
            #    origin=kdl.Vector(
            #        self.base_pose.p[0] + leg.origin.p[0],
            #        self.base_pose.p[1] + leg.origin.p[1], 0),
            #    angle=leg.origin_angle + (-polar.angle if leg.reverse else polar.angle),
            #    distance=polar.distance,
            #    z=polar.z
            #)
            fr = leg.to_rect(polar, self.base_pose)
            #print(f'leg {name} to {p.x:6.2f}, {p.y:6.2f}, {p.z:6.2f}  {p.angle:6.2f} @ {p.distance:6.2f}')
            print(f'leg {name} to {fr}')
            t = default_segment_trajectory_msg(
                leg.foot_link,
                reference_frame=self.odom_link,
                points=[to_vector3(fr)],
                rotations=[R(0., 0., 0., 1.)])
            self.transmit_trajectory(t)
        return leg_movement

    def move_legs_local(self, to: PolarCoord, legs: typing.Iterable = None):
        def local_leg_movement():
            trajectories = list()
            print("move legs local")
            for leg in self.resolve_legs(legs if legs else each_leg()):
                foot_base: PolarCoord = leg.polar(self.base_pose)
                adjusted_to = PolarCoord(
                    angle=to.angle + leg.neutral_angle,
                    distance=to.distance,
                    z=to.z
                )
                fr = leg.to_rect(adjusted_to, self.base_pose)
                #print(f'leg {leg.name} to {foot_base.p[0]}, {foot_base.p[1]}, {foot_base.p[2]}')
                t = default_segment_trajectory_msg(
                    leg.foot_link,
                    reference_frame=self.odom_link,
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
            WaitTask(1, init=self.move_legs_local(PolarCoord(0.0, 0.13, 0))),
            WaitTask(1.0, init=self.stand),
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
        self.tasks.append(WaitTask(0.5, init=enter_preview))

    def walk(self, heading: float, distance: float):
        # heading and distance are turned into a base move trajectory
        self.heading = heading   # todo: this is also set in the move_base callback
        self.gait = self.tripod_gait
        self.tasks.extend([
            WaitTask(2.0, init=self.move_base(heading=heading, distance=distance))
        ])

    def wave_leg(self):
        up = PolarCoord(angle=-0.2, distance=0.17, z=0.08)
        down = PolarCoord(angle=0.2, distance=0.12, z=-0.04)
        up_tasks = [WaitTask(0.5, init=self.move_leg(l_name, up)) for l_name in each_leg()]
        down_tasks = [WaitTask(0.5, init=self.move_leg(l_name, down)) for l_name in each_leg()]
        self.tasks = [
            #WaitTask(0.5, init=self.enable_motors),
            #WaitTask(2.0, init=self.move_leg('left-front-foot', 0.1418, -0.1844, 0.085)),
        ] + up_tasks + down_tasks + [
            #WaitTask(2.5, init=self.disable_motors),
        ]

    def step(self, name: str, to: PolarCoord):
        def local_movement():
            leg = self.legs[name]
            base_tf = self.get_frame_rel(self.base_link, self.odom_link)
            traj = leg.lift(to, base_tf)
            self.transmit_trajectory([traj])
        return local_movement

    def single_step(self):
        self.tasks.extend([
            #WaitTask(2.5, init=self.move_legs_local(PolarCoord(0.7, 0.128, 0.02))),
            #WaitTask(5.0, init=self.move_heading(-math.pi/2)),
            WaitTask(3., init=self.step("left-middle", PolarCoord(angle=0.5, distance=0.128, z=0.0))),
            WaitTask(3., init=self.step("left-middle", PolarCoord(angle=-0.5, distance=0.128, z=0.0))),
            WaitTask(3., init=self.step("left-middle", PolarCoord(angle=0.5, distance=0.128, z=0.0))),
            WaitTask(3., init=self.step("left-middle", PolarCoord(angle=-0.5, distance=0.128, z=0.0)))
        ])

    def take_steps(self):
        up = PolarCoord(angle=-0.2, distance=0.17, z=0.03)
        over = PolarCoord(angle=0.2, distance=0.17, z=0.03)
        down = PolarCoord(angle=0.2, distance=0.12, z=0.)
        back = PolarCoord(angle=-0.2, distance=0.17, z=0.)

        heading = -math.pi/2
        distance = 0.04

        task_set = []
        task_set.append([WaitTask(0.01, init=self.move_leg(l_name, up)) for l_name in tripod_set(0)])
        task_set.append([WaitTask(1.2)])
        task_set.append([WaitTask(0.01, init=self.move_leg(l_name, over)) for l_name in tripod_set(0)])
        task_set.append([WaitTask(1.2)])
        task_set.append([WaitTask(0.01, init=self.move_leg(l_name, down)) for l_name in tripod_set(0)])
        task_set.append([WaitTask(1.2)])
        task_set.append([WaitTask(0.01, init=self.move_leg(l_name, up)) for l_name in tripod_set(1)])
        #task_set.append([WaitTask(0.01, init=self.move_leg(l_name, back)) for l_name in tripod_set(0)])
        task_set.append([WaitTask(1.2, init=self.move_base(heading=heading, distance=distance))])
        #task_set.append([WaitTask(1.2)])
        task_set.append([WaitTask(0.01, init=self.move_leg(l_name, over)) for l_name in tripod_set(1)])
        task_set.append([WaitTask(1.2)])
        task_set.append([WaitTask(0.01, init=self.move_leg(l_name, down)) for l_name in tripod_set(1)])
        task_set.append([WaitTask(1.2)])
        task_set.append([WaitTask(0.01, init=self.move_leg(l_name, up)) for l_name in tripod_set(0)])
        task_set.append([WaitTask(1.2, init=self.move_base(heading=heading, distance=distance*2))])
        task_set.append([WaitTask(0.01, init=self.move_leg(l_name, down)) for l_name in tripod_set(0)])

        #self.tasks.append(WaitTask(0.5, init=self.enable_motors))
        for s in task_set:
            self.tasks.extend(s)
        #self.tasks.append(WaitTask(0.5, init=self.disable_motors))

    def wake_up(self):
        self.tasks = [
            WaitTask(0.5, init=self.enable_motors),
            WaitTask(2.5, init=self.move_legs_local(PolarCoord(0, 0.1, 0.1))),
            WaitTask(1.5, init=self.move_legs_local(PolarCoord(-0.35, 0.1, 0.1))),
            WaitTask(2.0, init=self.move_legs_local(PolarCoord(0.35, 0.1, 0.1))),
            WaitTask(1.5, init=self.move_legs_local(PolarCoord(0.0, 0.1, 0.1))),
            WaitTask(8.0, init=self.move_legs_local(PolarCoord(0.0, 0.13, -0.08))),
            WaitTask(3.0, init=self.move_legs_local(PolarCoord(0.0, 0.13, -0.02))),
            WaitTask(2.5, init=self.disable_motors)
        ]

    def merge_trajectory_test(self):
        self.tasks = [
            WaitTask(0.5, init=self.enable_motors)
        ]
        for leg in each_leg():
            self.tasks.append(WaitTask(0.5, init=self.move_legs_local(PolarCoord(0, 0.1, 0.1), [leg])))
        self.tasks.append(WaitTask(2.0))
        self.tasks.append(WaitTask(3.0, init=self.move_legs_local(PolarCoord(0.0, 0.13, -0.02))))
        self.tasks.append(WaitTask(2.5, init=self.disable_motors))

    #def spin(self):
    #    # if we have all the model state we need, we are ready to move
    #    #if self.legs:
    #    #    self.move_robot()
    #    if self.node:
    #        rclpy.spin_once(self, timeout_sec=0)


def main(args=sys.argv):
    rclpy.init(args=args)
    node = Hexapod(args)
    node.clear_trajectory()
    node.enter_preview_mode()
    #node.stand_and_sit()
    #node.wave_leg()
    node.stand_up()
    #node.take_steps()
    #node.single_step()
    #node.wake_up()
    #node.merge_trajectory_test()

    node.walk(-math.pi/2, 0.5)
    rclpy.spin(node)
    rclpy.shutdown()


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
