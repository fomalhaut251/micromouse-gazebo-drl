import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 获取包路径
    pkg_share = get_package_share_directory('drl_agent_gazebo')
    
    # 设定默认地图路径：指向 worlds 文件夹下的 museum_map.yaml
    default_map_path = os.path.join(pkg_share, 'worlds', 'museum_map.yaml')

    # 声明启动参数
    map_file_arg = DeclareLaunchArgument(
        'map',
        default_value=default_map_path,
        description='Full path to map yaml file to load'
    )

    # 1. 启动 Map Server (加载地图)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': LaunchConfiguration('map')}]
    )

    # 2. 启动 Lifecycle Manager (激活地图服务器)
    # Nav2 的节点需要生命周期管理才能进入 Active 状态
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server']
        }]
    )

    # 3. 发布 map -> odom 的静态坐标变换
    # 这样 Rviz 才能正确把地图叠在里程计坐标系下
    tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    return LaunchDescription([
        map_file_arg,
        map_server_node,
        lifecycle_manager_node,
        tf_publisher
    ])