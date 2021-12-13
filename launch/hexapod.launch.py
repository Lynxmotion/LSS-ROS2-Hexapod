#!/usr/bin/python3

import os
import sys
from pathlib import Path
import launch

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node, LifecycleNode
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir, LaunchConfiguration


def generate_launch_description():

    # Set LOG format
    os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '{time}: [{name}] [{severity}]\t{message}'

    humanoid_config_dir = os.path.join(get_package_share_directory('lss_humanoid'), 'config')
    config_dir = os.path.join(get_package_share_directory('lss_hexapod'), 'config')

     # URDF file to be loaded by Robot State Publisher
    hexapod_urdf = os.path.join(
        get_package_share_directory('lss_hexapod'),
            'urdf', 'lss_hexapod.urdf'
    )

    hexapod_config = Path(config_dir, 'hexapod.yaml')
    assert hexapod_config.is_file()

    presence_config = Path(config_dir, 'presence.yaml')
    assert presence_config.is_file()

    # URDF file to be loaded by Robot State Publisher
    rviz_config = os.path.join(
        config_dir, 'lss_hexapod.rviz'
    )
    assert os.path.isfile(rviz_config)

    # see this file for one way to start ros2 controllers
    # https://github.com/ros-planning/moveit2/blob/main/moveit_ros/moveit_servo/launch/servo_cpp_interface_demo.launch.py


    # configure a publisher for the URDF
    urdf_publisher = Node(
        package='resource_publisher',
        executable='resource_publisher',
        output='screen',
        arguments=[
            '-package', 'lss_hexapod',
            '-xacro', 'urdf/lss_hexapod.xacro.urdf',
            '-topic', 'robot_description',
            '-targets', '*,gazebo']
    )

    localization = Node(
        name='localization',
        package='robot_localization',
        executable='ekf_node',
        output='screen',
        parameters=[presence_config],
        remappings=[
            ('odometry/filtered', 'odom')
        ]
    )

    robot_dynamics_node = Node(
        name="robot_dynamics",
        package='humanoid_dynamic_model',
        executable='robot_dynamics',
        output='screen',
        parameters=[hexapod_config, {'sim-mode': False}]
    )

    # Robot Visualizer
    rviz_node = Node(
        name = 'rviz2',        # must match the node name in config -> YAML
        package = 'rviz2',
        executable = 'rviz2',
        output = 'screen',
        arguments = [
            '-d', rviz_config
        ]
    )

    return LaunchDescription([
        urdf_publisher,
        #localization,
        #robot_dynamics_node,
        rviz_node
    ])



