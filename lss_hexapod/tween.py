import functools

import PyKDL as kdl
from geometry_msgs.msg import Vector3

from .polar import PolarCoord

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
