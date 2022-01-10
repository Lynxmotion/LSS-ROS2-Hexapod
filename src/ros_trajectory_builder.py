# Ros2 node imports
import rclpy
import tf2_kdl
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Vector3, Transform, TransformStamped, Quaternion
from robot_model_msgs.msg import SegmentTrajectory
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from robot_model_msgs.msg import ModelState

from joint import Joint, JointGroup
from tf_conv import to_kdl_rotation, to_kdl_vector, to_kdl_frame, to_vector3, to_transform, to_quaternion, P

import unittest


class Event(list):
    """Event subscription.

    A list of callable objects. Calling an instance of this will cause a
    call to each item in the list in ascending order by index.

    Example Usage:
    >> def f(x):
    ...     print 'f(%s)' % x
    >> def g(x):
    ...     print 'g(%s)' % x
    >> e = Event()
    >> e()
    >> e.append(f)
    >> e(123)
    f(123)
    >> e.remove(f)
    >> e()
    >> e += (f, g)
    >> e(10)
    f(10)
    g(10)
    >> del e[0]
    >> e(2)
    g(2)

    """
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)


def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r/2.0), np.sin(p/2.0), np.sin(y/2.0)
    cr, cp, cy = np.cos(r/2.0), np.cos(p/2.0), np.cos(y/2.0)
    return [sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy]


def urdf_pose_to_kdl_frame(pose):
    pos = [0., 0., 0.]
    rot = [0., 0., 0.]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation
    return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)),
                     kdl.Vector(*pos))


def urdf_joint_to_kdl_joint(jnt):
    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    if jnt.joint_type == 'fixed' or jnt.joint_type == 'floating':
        return kdl.Joint(jnt.name)
    axis = kdl.Vector(*jnt.axis)
    if jnt.joint_type == 'revolute':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'continuous':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'prismatic':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.TransAxis)
    print("Unknown joint type: %s." % jnt.joint_type)
    return kdl.Joint(jnt.name)


def urdf_inertial_to_kdl_rbi(i):
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = kdl.RigidBodyInertia(i.mass, origin.p,
                               kdl.RotationalInertia(i.inertia.ixx,
                                                     i.inertia.iyy,
                                                     i.inertia.izz,
                                                     i.inertia.ixy,
                                                     i.inertia.ixz,
                                                     i.inertia.iyz))
    return origin.M * rbi


def joint_list_to_kdl(q):
    if q is None:
        return None
    if type(q) == np.matrix and q.shape[1] == 0:
        q = q.T.tolist()[0]
    q_kdl = kdl.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl


##
# Returns a PyKDL.Tree generated from a urdf_parser_py.urdf.URDF object.
def kdl_tree_from_urdf_model(urdf):
    root = urdf.get_root()
    tree = kdl.Tree(root)
    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            for joint, child_name in urdf.child_map[parent]:
                child = urdf.link_map[child_name]
                if child.inertial is not None:
                    kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                else:
                    kdl_inert = kdl.RigidBodyInertia()
                urdf_joint = urdf.joint_map[joint]
                kdl_jnt = urdf_joint_to_kdl_joint(urdf_joint)
                kdl_origin = urdf_pose_to_kdl_frame(urdf_joint.origin)
                kdl_sgm = kdl.Segment(child_name, kdl_jnt,
                                      kdl_origin, kdl_inert)
                tree.addSegment(kdl_sgm, parent)
                add_children_to_tree(child_name)
    add_children_to_tree(root)
    return tree


def default_segment_trajectory_msg(
        segment: str, start: float = 0.0,
        velocity: float = 1.0,
        acceleration: float = 1.2,
        supporting: bool = False,
        reference_frame: str = 'base_link',
        points: [] = None,
        rotations: [] = None):
    t = SegmentTrajectory()
    t.start.sec = int(start)
    t.start.nanosec = int((start - t.start.sec) * 1000000000)
    t.segment = segment
    t.profile = 'velocity/trap'
    t.velocity = velocity
    t.acceleration = acceleration
    t.supporting = supporting
    t.path = 'rounded'
    t.reference_frame = reference_frame
    t.coordinate_mode = 0
    # if points:
    #    t.points = [to_vector3(f) for f in points]
    # if rotations:
    #    t.rotations = [to_quaternion(r) for r in rotations]
    if points:
        t.points = [to_vector3(p) if isinstance(p, kdl.Vector) else p for p in points]
    if rotations:
        t.rotations = [to_quaternion(r) if isinstance(r, kdl.Rotation) else r for r in rotations]
    return t


class TrajectoryBuilder(Node):
    debug_urdf = r'/home/guru/src/lss-humanoid/ros2/humanoid/src/lss_humanoid/urdf/lss_humanoid.urdf'

    frame_names: list = []
    frame_map: dict = {}
    limbs: list = []

    on_urdf_update = Event()

    on_frame_names_changed = Event()
    on_limbs_changed = Event()

    on_tf_update = Event()
    urdf = None

    # selected limb
    tree: kdl.Tree = None

    limb_chain: kdl.Chain = None
    limb_fk_kdl: kdl.ChainFkSolverPos_recursive = None

    rel_frame_chain: kdl.Chain = None
    rel_frame_fk_kdl: kdl.ChainFkSolverPos_recursive = None

    def __init__(self, args):
        super().__init__('trajectory_builder')

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

        # pull URDF so we can determine tf linkages
        self.urdf_sub = self.create_subscription(
            String,
            '/robot_description',
            self.urdf_callback,
            transient_local_reliable_profile)

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

    def spin(self):
        rclpy.spin_once(self, timeout_sec=0)
        rclpy.spin_once(self, timeout_sec=0)
        rclpy.spin_once(self, timeout_sec=0)

    def urdf_callback(self, msg):
        self.urdf = msg.data
        self.robot = URDF.from_xml_string(self.urdf)
        self.tree = kdl_tree_from_urdf_model(self.robot)
        self.on_urdf_update()

    def load_urdf(self):
        #self.robot = URDF.from_xml_file(self.debug_urdf)
        if self.urdf:
            self.robot = URDF.from_xml_string(self.urdf)
            self.tree = kdl_tree_from_urdf_model(self.robot)

    def set_limb(self, limb_name):
        if not self.tree:
            self.load_urdf()
            #raise RuntimeError('No loaded URDF tree')
        self.limb_chain = self.tree.getChain("base_link", limb_name)
        self.limb_fk_kdl = kdl.ChainFkSolverPos_recursive(self.limb_chain)
        print("set active limb to ", limb_name)

    def set_rel_frame(self, rel_frame_name):
        if not self.tree:
            print("cannot set relative frame before URDF is loaded")
            #self.load_urdf()
            return False
            #raise RuntimeError('No loaded URDF tree')
        self.rel_frame_chain = self.tree.getChain("base_link", rel_frame_name)
        self.rel_frame_fk_kdl = kdl.ChainFkSolverPos_recursive(self.rel_frame_chain)
        print("set relative frame to ", rel_frame_name)
        return True

    def update_limb_fk(self, joints: list):
        if not self.limb_chain:
            return None
        end_effector_f = kdl.Frame()
        lnr = self.limb_chain.getNrOfSegments()
        jnr = self.limb_chain.getNrOfJoints()
        joints_kdl = joint_list_to_kdl(joints)
        kinematics_status = self.limb_fk_kdl.JntToCart(joints_kdl,
                                                       end_effector_f,
                                                       lnr)
        #print(kinematics_status)
        return end_effector_f

    def compute_fk(self, chain, fk):
        # since we subscribe to TF the joint FK is already resolved and we can
        # just walk the segments and calculate the end effector frame
        if not chain:
            return None
        end_effector_f = kdl.Frame()
        for i in range(0, chain.getNrOfSegments()):
            seg = chain.getSegment(i)
            name = seg.getName()
            if name not in self.frame_map:
                return None
            seg_tf = self.frame_map[name]
            end_effector_f = end_effector_f * seg_tf
        return end_effector_f

    def get_end_effector(self):
        # since we subscribe to TF the joint FK is already resolved and we can
        # just walk the segments and calculate the end effector frame
        end_effector_f = self.compute_fk(self.limb_chain, self.limb_fk_kdl)
        if end_effector_f and self.rel_frame_fk_kdl:
            rel_frame_f = self.compute_fk(self.rel_frame_chain, self.rel_frame_fk_kdl)
            if not rel_frame_f:
                return None
            end_effector_f = rel_frame_f.Inverse() * end_effector_f
        return end_effector_f

    def tf_callback(self, msg):
        new_frame_names = list(l.child_frame_id for l in msg.transforms if l.child_frame_id not in self.frame_names)
        #if not all(a == b for a, b in zip(frame_names, self.frame_names)):
        if len(new_frame_names):
            print("update frames")
            self.frame_names += new_frame_names
            self.on_frame_names_changed(self.frame_names)

        # convert transforms to KDL frame map
        self.frame_map.update({t.child_frame_id: to_kdl_frame(t.transform) for t in msg.transforms})
        self.on_tf_update(self.frame_map)

    def model_state_callback(self, msg):
        if not self.urdf:
            return
        # simply store model state until next IMU message
        self.model_state = msg

        limbs = list(l.name for l in msg.contact.limbs)
        if not self.limbs or not all(a == b for a, b in zip(limbs, self.limbs)):
            print("update")
            self.limbs = limbs
            self.on_limbs_changed(limbs)

    def transmit_trajectory(self, tsegs):
        if type(tsegs) != list:
            tsegs = [tsegs]
        msj = MultiSegmentTrajectory()
        msj.header.stamp = self.get_clock().now().to_msg()
        msj.header.frame_id = 'odom'
        msj.segments = tsegs
        self.trajectory_pub.publish(msj)


class TrajectoryBuilderTests(unittest.TestCase):
    urdf = r'/home/guru/src/lss-humanoid/ros2/humanoid/src/lss_humanoid/urdf/lss_humanoid.urdf'

    def test_load_urdf(self):
        robot = URDF.from_xml_file(self.urdf)
        tree = kdl_tree_from_urdf_model(robot)
        chain = tree.getChain("base_link", "LHand")
        for i in range(chain.getNrOfSegments()):
            print(chain.getSegment(i).getName())

        fk_kdl = kdl.ChainFkSolverPos_recursive(chain)
        endeffec_frame = kdl.Frame()
        lnr = chain.getNrOfSegments()
        jnr = chain.getNrOfJoints()
        joints = joint_list_to_kdl([0, 0, 0, 0, 0, 0])
        kinematics_status = fk_kdl.JntToCart(joints,
                                                   endeffec_frame,
                                                   lnr)
        print(kinematics_status)

if __name__ == '__main__':
    unittest.main()
