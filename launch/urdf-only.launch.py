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

    # URDF file to be loaded by Robot State Publisher
    rviz_config = os.path.join(
        get_package_share_directory('lss_hexapod'),
        'config', 'lss_hexapod.rviz'
    )
    assert os.path.isfile(rviz_config)

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

    # Robot State Publisher
    # URDF publishing is now provided by resource_publisher and TF is provided by robot_dynamics
    rsp_node = Node(
        name = 'hexapod_state_publisher',        # must match the node name in config -> YAML
        package = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output = 'screen',
        #parameters = [hexapod_config]
        arguments = [hexapod_urdf]
    )

    # Joint State Publisher
    # Provide a UI to publish joint angles to test the Robot limbs
    joint_state_publisher_gui_node = Node(
        name = 'joint_state_publisher_gui',        # must match the node name in config -> YAML
        package = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        output = 'screen',
        #parameters = [hexapod_config]
        #arguments = [hexapod_urdf]
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
        #urdf_publisher,
        #rsp_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])



