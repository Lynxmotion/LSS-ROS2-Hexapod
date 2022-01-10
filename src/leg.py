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
    ADVANCING = 4

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

    # the amount of error between current position and target
    error: float

    def __init__(self, name: str, origin: kdl.Frame):
        self.name = name
        self.foot_link = name + '-foot'
        self.hip_link = name + '-hip-span1'
        self.origin = origin

        # compute the origin angle of the leg (it's neutral heading)
        self.origin_angle = math.atan2(origin.p[1], origin.p[0])

    def tick(self, state: dict):
        pass

    def to_polar(self, rect: kdl.Vector, use_neutral: bool = True) -> PolarCoord:
        if not rect:
            rect = self.rect.p
        elif isinstance(rect, kdl.Frame):
            rect = rect.p
        # get foot location relative to limb origin
        hip_foot = self.origin.Inverse() * rect
        coord = PolarCoord.from_rect(hip_foot[0], hip_foot[1], hip_foot[2])
        #coord.angle -= self.origin_angle
        if self.reverse:
            coord.angle *= -1
        if use_neutral:
            coord.angle -= self.neutral_angle
        return coord

    # return the rectangular coordinates from the polar coordinates relative to the base_link
    def to_rect(self, coord: PolarCoord, use_neutral: bool = True) -> kdl.Vector:
        coord = PolarCoord(
            angle=coord.angle,
            distance=coord.distance,
            z=coord.z
        )
        if use_neutral:
            coord.angle += self.neutral_angle
        if self.reverse:
            coord.angle = -coord.angle
        #coord.angle -= self.origin_angle
        return self.origin * coord.to_kdl_vector()

    @property
    def polar(self) -> PolarCoord:
        return self.to_polar(self.rect.p)

    def lift(self, polar: PolarCoord, velocity: float, lift: float = 0.02):
        fr = self.polar

        print(f'lift leg {self.name} from {fr} => {polar}')
        if math.isnan(polar.zlocal):
            polar.zlocal = fr.zlocal
        tw_dist = Tween(fr, polar)

        # a helper function to tween between current leg position and target position [0 - 1.]
        def get_point(p: float):
            # lift leg in a semicircle as we proceed to target
            v: PolarCoord = tw_dist.get(p)

            # transform from polar to rectangular in odom frame
            r = self.to_rect(v)

            # add in Z lift as an arc
            r[2] += lift * math.sin(p * math.pi)
            #print(f'   {p:2.1f} => {v} => {r[0]} {r[1]} {r[2]}')
            return to_vector3(r)

        return default_segment_trajectory_msg(
            self.foot_link,
            velocity=velocity,
            reference_frame='base_link',
            points=[get_point(d/10) for d in range(1, 11)],
            #rotations=[])
            rotations=[to_quaternion(self.rect.M)])
