# Copyright 2020 ROS2-Control Development Team (2020)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

import xacro


def generate_launch_description():
    lss_package_path = os.path.join(get_package_share_directory('lss_hexapod'))

    # Get URDF via xacro
    robot_description_path = os.path.join(
        lss_package_path,
        'urdf',
        'lss_hexapod.xacro.urdf')
    robot_description_config = xacro.process_file(robot_description_path)
    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_controllers = os.path.join(
        lss_package_path,
        'config',
        'hexapod.yaml'
        )

    #joint_state_publisher_node = Node(
    #    package='joint_state_publisher_gui',
    #    executable='joint_state_publisher_gui',
    #)
    #robot_state_publisher_node = Node(
    #    package='robot_state_publisher',
    #    executable='robot_state_publisher',
    #    output='both',
    #    parameters=[robot_description]
    #)

    return LaunchDescription([
      Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, robot_controllers],
        output={
          'stdout': 'screen',
          'stderr': 'screen',
          },
        ),
        #joint_state_publisher_node,
        #robot_state_publisher_node
    ])
