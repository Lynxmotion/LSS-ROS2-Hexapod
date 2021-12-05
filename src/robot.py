# prevent type issues with circular dependencies (such as StateEvent and RobotState)
from __future__ import annotations

import math
import typing

import PyKDL as kdl
from tf_conv import to_vector3, P, R

from polar import PolarCoord
from tween import Tween
from ros_trajectory_builder import default_segment_trajectory_msg


class StateEvent:
    target: typing.Callable = None
    handler: typing.Callable = None

    def waitfor(self, target: typing.Callable, handler: typing.Callable):
        self.target = target
        self.handler = handler

    def check(self, state: RobotState):
        if self.target():
            self.handler()
            return True
        else:
            return False


class RobotState:
    frame_map: dict

    model_state = None

    def __init__(self):
        self.frame_map = {}

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

