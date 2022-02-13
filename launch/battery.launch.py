#!/usr/bin/python3

import os
import sys
from pathlib import Path
import launch

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node, LifecycleNode, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
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

    robot_dynamics_config = Path(config_dir, 'robot_dynamics.yaml')
    assert robot_dynamics_config.is_file()

    robot_control_config = Path(config_dir, 'robot_control.yaml')
    assert robot_control_config.is_file()

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

    srdf_publisher = Node(
        package='resource_publisher',
        executable='resource_publisher',
        output='screen',
        arguments=[
            '-package', 'lss_hexapod',
            '-xacro', 'urdf/lss_hexapod.srdf',
            '-topic', 'robot_description/srdf']
    )

    robot_dynamics_container = ComposableNodeContainer(
            name='robot_dynamics_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='robot_dynamics',
                    plugin='robot_dynamics::Dynamics',
                    name='robot_dynamics',
                    parameters=[robot_dynamics_config, {'sim-mode': False}]),
                ComposableNode(
                    package='robot_dynamics',
                    plugin='robot_dynamics::Control',
                    name='robot_control',
                    parameters=[robot_control_config]),
            ],
            output='screen'
    )

    return LaunchDescription([
        launch.actions.DeclareLaunchArgument('hardware', default_value='motion'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([ThisLaunchFileDir(),'/',LaunchConfiguration('hardware'), '.launch.py'])),
        urdf_publisher,
        srdf_publisher,
        robot_dynamics_container
    ])



