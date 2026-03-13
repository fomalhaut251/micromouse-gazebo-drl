from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mapper_node = Node(
        package="drl_agent",
        executable="structured_online_mapper.py",
        name="structured_online_mapper",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "map_topic": "/map",
                "frame_id": "odom",
                "grid_size": 16,
                "x_min": -8.0,
                "y_min": -8.0,
                "cell_size": 1.0,
                "wall_thickness_threshold": 0.12,
                "wall_confirm_hits": 3,
                "cell_pixels": 10,
                "publish_rate": 5.0,
            }
        ],
    )

    return LaunchDescription([mapper_node])
