import unittest


class Joint(object):
    name: str
    position: float
    velocity: float
    effort: float

    def __init__(self, name: str):
        self.name = name
        self.position = 0.0
        self.velocity = 0.0
        self.effort = 0.0


class JointGroup(object):
    joints: list

    def __init__(self, joints: list):
        self.joints = []

        # add each joint as a object member
        for j in joints:
            if type(j) == str:
                joint = Joint(j)
            elif isinstance(j, Joint):
                joint = j
                j = joint.name
            elif isinstance(j, JointGroup):
                self.joints = self.joints + j.joints
                for k in j.joints:
                    self.__dict__[k.name] = k
                continue
            else:
                raise TypeError("JointGroup expects array of joint names, joints, and/or joint groups.")
            self.__dict__[j] = joint
            self.joints.append(joint)

    @property
    def position(self):
        return list(j.position for j in self.joints)

    @property
    def effort(self):
        return list(j.effort for j in self.joints)

    @effort.setter
    def effort(self, value):
        """ Apply a value to every joint through the 'func' callback. Usually the func would be a lambda function. """
        if isinstance(value, list):
            n = 0
            while n < len(self.joints) and n < len(value):
                self.joints[n].effort = value[n]
                n = n + 1
        elif isinstance(value, float):
            for j in self.joints:
                j.effort = value
        elif callable(value):
            for n, j in enumerate(self.joints):
                j.effort = value(j, n)
        else:
            raise TypeError("joint group effort must be float or array of floats")

    def __add__(self, other):
        """ Return a new JointGroup that combines the joints of both operands """
        if isinstance(other, JointGroup):
            return JointGroup([self, other])
        if isinstance(other, Joint):
            return JointGroup(self.joints + [other])
        if type(other) == list:
            return JointGroup(self.joints + other)
        else:
            raise TypeError("can only add joints to a joint group")

class JointGroupTests(unittest.TestCase):
    def test_creation(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        self.assertIsNotNone(group)

    def test_creation_wtih_joints(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        self.assertIsNotNone(group.J1)
        self.assertIsNotNone(group.J2)
        group.J1.position = 1.1
        group.J2.position = 1.2
        self.assertEqual(group.J1.position, 1.1)
        self.assertEqual(group.J2.position, 1.2)
        group2 = JointGroup([group.J1, group.J2])
        self.assertIsNotNone(group2.J1)
        self.assertIsNotNone(group2.J2)
        self.assertEqual(group2.J1.position, 1.1)
        self.assertEqual(group2.J2.position, 1.2)

    def test_creation_wtih_joint_groups(self):
        group1 = JointGroup('J1,J2'.split(','))
        group2 = JointGroup('J3,J4,J5'.split(','))
        self.assertIsNotNone(group1.J1)
        self.assertIsNotNone(group2.J3)
        group1.J1.position = 1.1
        group2.J3.position = 1.2
        self.assertEqual(group1.J1.position, 1.1)
        self.assertEqual(group2.J3.position, 1.2)
        group3 = JointGroup([group1, group2])
        self.assertIsNotNone(group3.J1)
        self.assertIsNotNone(group3.J2)
        self.assertIsNotNone(group3.J3)
        self.assertIsNotNone(group3.J4)
        self.assertIsNotNone(group3.J5)
        self.assertEqual(group3.J1.position, 1.1)
        self.assertEqual(group3.J3.position, 1.2)

    def test_joint_indices(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        self.assertIsNotNone(group.J1)
        self.assertIsNotNone(group.J2)
        self.assertIsNotNone(group.J3)
        self.assertIsNotNone(group.J4)
        self.assertIsNotNone(group.J5)
        self.assertEqual(group.J1.name, 'J1')
        self.assertEqual(group.J2.name, 'J2')
        self.assertEqual(group.J3.name, 'J3')
        self.assertEqual(group.J4.name, 'J4')
        self.assertEqual(group.J5.name, 'J5')

    def test_joint_current_member(self):
        group = JointGroup('J1,J2'.split(','))
        self.assertIsNotNone(group.J1)
        self.assertEqual(group.J1.name, 'J1')
        self.assertEqual(group.J2.name, 'J2')
        self.assertEqual(group.J1.effort, 0)
        self.assertEqual(group.J2.effort, 0)
        group.J1.effort = 3.0
        group.J2.effort = 2.5
        self.assertEqual(group.J1.effort, 3)
        self.assertEqual(group.J2.effort, 2.5)

    def test_positions(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        group.J1.position = 1.1
        group.J2.position = 2.1
        group.J3.position = 3.1
        group.J4.position = 4.1
        group.J5.position = 5.1
        positions = group.position
        #print(','.join(str(p) for p in positions))
        self.assertEqual(positions, [1.1, 2.1, 3.1, 4.1, 5.1])

    def test_efforts(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        group.J1.effort = 1.1
        group.J2.effort = 2.1
        group.J3.effort = 3.1
        group.J4.effort = 4.1
        group.J5.effort = 5.1
        efforts = group.effort
        #print(','.join(str(p) for p in efforts))
        self.assertEqual(efforts, [1.1, 2.1, 3.1, 4.1, 5.1])

    def test_efforts_apply_value(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        group.effort = 1.1
        efforts = group.effort
        self.assertEqual(efforts, [1.1, 1.1, 1.1, 1.1, 1.1])

    def test_efforts_apply_array(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        group.effort = [1.1, 2.1, 3.1, 4.1, 5.1]
        efforts = group.effort
        self.assertEqual(efforts, [1.1, 2.1, 3.1, 4.1, 5.1])

    def test_efforts_apply_undersized_array(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        group.effort = [1.1, 2.1, 3.1]
        efforts = group.effort
        self.assertEqual(efforts, [1.1, 2.1, 3.1, 0.0, 0.0])

    def test_efforts_oversized_array(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        group.effort = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
        self.assertEqual(group.effort, [1.1, 2.1, 3.1, 4.1, 5.1])

    def test_efforts_apply_lambda(self):
        group = JointGroup('J1,J2,J3,J4,J5'.split(','))
        group.effort = lambda j, i: i + 1.1
        self.assertEqual(group.effort, [1.1, 2.1, 3.1, 4.1, 5.1])


if __name__ == '__main__':
    unittest.main()
