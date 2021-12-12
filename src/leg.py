import math
import typing

import PyKDL as kdl
from tf_conv import to_vector3, P, R, to_quaternion

from polar import PolarCoord
from tween import Tween
from ros_trajectory_builder import default_segment_trajectory_msg

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


class Leg:
    IDLE = 0
    MOVING = 1
    LIFTING = 2
    SUPPORTING = 3

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
    lift_max = 0.02
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


    def tick(self, state: dict):
        pass

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

        fr = self.polar()

        print(f'lift leg {self.name} from {fr} => {polar}')
        if math.isnan(polar.zlocal):
            polar.zlocal = fr.zlocal
        tw_dist = Tween(fr, polar)

        # a helper function to tween between current leg position and target position [0 - 1.]
        def get_point(p: float):
            # lift leg in a semicircle as we proceed to target
            v: PolarCoord = tw_dist.get(p)

            # where will base be when our leg lands?
            #base_future = kdl.Frame(base_pose)
            #point_vel = base_velocity * (duration * p)
            #base_future.p += point_vel

            # transform from polar to rectangular in odom frame
            r = self.to_rect(v)

            # add in Z lift as an arc
            r[2] += self.lift_max * math.sin(p * math.pi)
            #print(f'   {p:2.1f} => {v} => {r[0]} {r[1]} {r[2]}')
            #fo = base_future * r
            return to_vector3(r)

        return default_segment_trajectory_msg(
            self.foot_link,
            id='lift-leg',
            velocity=2.0,
            reference_frame='base_link',
            points=[get_point(d/10) for d in range(1, 11)],
            rotations=[to_quaternion(self.rect.M)])
