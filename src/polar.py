import math

from geometry_msgs.msg import Vector3

import PyKDL as kdl
from tf_conv import to_kdl_rotation, to_kdl_vector, to_kdl_frame, to_vector3, to_transform, to_quaternion, P, R

epsilon = 0.00001

def near_zero(x1: float):
    return abs(x1) < epsilon


def near_equal(x1: float or kdl.Vector, x2: float or kdl.Vector, eps: float = epsilon):
    if type(x1) != type(x2):
        raise TypeError('near_equal requires the args to be the same type')
    elif isinstance(x1, kdl.Vector):
        return near_equal(x1.x(), x2.x(), eps) and near_equal(x1.y(), x2.y(), eps) and  near_equal(x1.z(), x2.z(), eps)
    else:
        return abs(x1 - x2) < eps


class PolarCoord:
    origin: kdl.Vector
    angle: float
    distance: float
    zlocal: float           # z is optional and part of the rectangular coordinate system

    def __init__(self, angle: float, distance: float, z: float = math.nan, origin=None):
        self.origin = origin if origin else kdl.Vector(0, 0, 0)
        #while angle > 2*math.pi:
        #    angle -= 2*math.pi
        #while angle < 0:
        #    angle += 2*math.pi
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

    def near(self, other, angle_eps: float = 0.035, distance_eps = 0.005):
        if isinstance(other, PolarCoord):
            return near_equal(self.angle, other.angle, angle_eps) \
                   and near_equal(self.distance, other.distance, distance_eps)
        else:
            return False

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
        return self.origin[2] + self.zlocal if not math.isnan(self.zlocal) else 0.0

    def to_xyz(self):
        return self.x, self.y, self.z

    def to_kdl_vector(self):
        return kdl.Vector(x=self.x, y=self.y, z=self.z)

    def to_vector3(self):
        return Vector3(x=self.x, y=self.y, z=self.z)

