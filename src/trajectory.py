import typing

import PyKDL as kdl
from robot_model_msgs.msg import TrajectoryProgress, TrajectoryComplete, SegmentTrajectory

#from .polar import PolarCoord
#from .leg import Leg, each_leg


class Trajectory:
    id: str
    synchronize: bool = True
    complete: typing.Callable = None
    progress: typing.Callable = None
    rejected: typing.Callable = None

    def __init__(
            self,
            synchronize: bool = True,
            id: str = None,
            complete: typing.Callable = None,
            progress: typing.Callable = None,
            rejected: typing.Callable = None):
        self.synchronize = synchronize
        self.id = id
        self.complete = complete
        self.progress = progress
        self.rejected = rejected


class PathTrajectory(Trajectory):
    goal: SegmentTrajectory or typing.List[SegmentTrajectory]

    def __init__(
            self,
            goal: SegmentTrajectory or typing.List[SegmentTrajectory],
            synchronize: bool = True,
            id: str = None,
            complete: typing.Callable = None,
            progress: typing.Callable = None,
            rejected: typing.Callable = None):
        super(PathTrajectory, self).__init__(
            synchronize=synchronize,
            id=id,
            complete=complete,
            progress=progress,
            rejected=rejected)
        self.goal = goal


class LinearTrajectory(Trajectory):
    effectors: str or typing.List[str]
    twists: kdl.Twist or typing.List[kdl.Twist]
    linear_acceleration: float = 0.0
    angular_acceleration: float = 0.0
    mode_in: int
    mode_out: int
    supporting: bool = False

    def __init__(
            self,
            effectors: str or typing.List[str],
            twists: kdl.Twist or typing.List[kdl.Twist],
            linear_acceleration: float = 0.0,
            angular_acceleration: float = 0.0,
            mode_in: int = SegmentTrajectory.UNCHANGED,
            mode_out: int = SegmentTrajectory.UNCHANGED,
            supporting: bool = False,
            synchronize: bool = True,
            id: str = None,
            complete: typing.Callable = None,
            progress: typing.Callable = None,
            rejected: typing.Callable = None):
        super(LinearTrajectory, self).__init__(
            synchronize=synchronize,
            id=id,
            complete=complete,
            progress=progress,
            rejected=rejected)
        self.effectors = effectors
        self.twists = twists
        self.linear_acceleration = linear_acceleration
        self.angular_acceleration = angular_acceleration
        self.mode_in = mode_in
        self.mode_out = mode_out
        self.supporting = supporting

