import sys
import rclpy
import time
import math
import typing

from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3
from humanoid_model_msgs.msg import MultiSegmentTrajectory, SegmentTrajectory
from humanoid_model_msgs.msg import ModelState

from ros_trajectory_builder import default_segment_trajectory_msg
import PyKDL as kdl
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
