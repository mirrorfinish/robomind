#!/usr/bin/env python3
"""
Sample ROS2 Launch File for Testing RoboMind Parser

This file demonstrates common ROS2 launch patterns that RoboMind should detect:
- DeclareLaunchArgument
- Node declarations with parameters and remappings
- ComposableNodeContainer with ComposableNode
- TimerAction for delayed starts
- GroupAction for grouping nodes
- ExecuteProcess for external commands
- IfCondition for conditional logic
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, GroupAction
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    enable_voice_arg = DeclareLaunchArgument(
        'enable_voice',
        default_value='true',
        description='Enable voice control system'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='test_robot',
        description='Name of the robot'
    )

    # Composable node container
    hardware_container = ComposableNodeContainer(
        name='hardware_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='robot_state_publisher',
                plugin='robot_state_publisher::RobotStatePublisher',
                name='robot_state_publisher',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time')
                }]
            )
        ],
        output='screen',
    )

    # Stage 1: Hardware nodes (immediate)
    stage1_hardware = GroupAction([
        Node(
            package='test_pkg',
            executable='odometry_node',
            name='odometry_publisher',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'publish_rate': 50.0,
            }],
            output='screen',
            respawn=True,
            respawn_delay=2.0
        ),

        Node(
            package='test_pkg',
            executable='motor_controller',
            name='motor_controller',
            parameters=[{
                'max_speed': 1.5,
                'wheel_base': 0.17,
            }],
            remappings=[
                ('/cmd_vel', '/robot/cmd_vel'),
                ('/odom', '/robot/odom'),
            ],
            output='screen',
        ),
    ])

    # Stage 2: Sensor nodes (3s delay)
    stage2_sensors = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='rplidar_ros',
                executable='rplidar_node',
                name='rplidar_node',
                parameters=[{
                    'serial_port': '/dev/ttyUSB0',
                    'frame_id': 'laser_frame',
                }],
                output='screen',
            )
        ]
    )

    # Stage 3: Voice system (5s delay, conditional)
    stage3_voice = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='voice_pkg',
                executable='voice_node',
                name='voice_control',
                condition=IfCondition(LaunchConfiguration('enable_voice')),
                parameters=[{'sensitivity': 0.7}],
                output='screen',
            )
        ]
    )

    # External process
    nav2_launch = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'launch', 'nav2_bringup', 'navigation_launch.py'],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        enable_voice_arg,
        robot_name_arg,
        hardware_container,
        stage1_hardware,
        stage2_sensors,
        stage3_voice,
        nav2_launch,
    ])
