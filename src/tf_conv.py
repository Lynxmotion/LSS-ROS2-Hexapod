import PyKDL
from geometry_msgs.msg import Vector3, Transform, TransformStamped, Quaternion, Pose

def to_kdl_rotation(quaternion):
    return PyKDL.Rotation.Quaternion(x=quaternion.x, y=quaternion.y, z=quaternion.z, w=quaternion.w)


def to_kdl_vector(vector3):
    return PyKDL.Vector(x=vector3.x, y=vector3.y, z=vector3.z)


def to_kdl_frame(transform):
    if isinstance(transform, Pose):
        return PyKDL.Frame(
            to_kdl_rotation(transform.orientation),
            to_kdl_vector(transform.position))
    else:
        return PyKDL.Frame(
            to_kdl_rotation(transform.rotation),
            to_kdl_vector(transform.translation))


def to_vector3(vector: PyKDL.Vector):
    return Vector3(x=vector[0], y=vector[1], z=vector[2])


def to_quaternion(rotation: PyKDL.Rotation):
    x, y, z, w = rotation.GetQuaternion()
    return Quaternion(x=x, y=y, z=z, w=w)


def to_transform(f: PyKDL.Frame):
    return Transform(
        translation=to_vector3(f.p),
        rotation=to_quaternion(f.M)
    )


def P(x: float, y: float, z: float):
    return Vector3(x=x, y=y, z=z)


def R(x: float, y: float, z: float, w: float):
    return Quaternion(x=x, y=y, z=z, w=w)
