forked from
[DRL Agent for Mobile Robot Navigation](https://github.com/anurye/Mobile-Robot-Navigation-Using-Deep-Reinforcement-Learning-and-ROS#drl-for-mobile-robot-navigation-using-ros2)

## Project Structure

```txt
.
├── 📂 docs/: contains demo videos
│   ├── 📄 dynamic_environment.mp4
│   ├── 📄 slam.mp4
│   └── 📄 simulation.mp4
├── 📂 drl_agent/: main deep reinforcement learning agent directory
│   ├── 📂 config/: contains configuration files
│   ├── 📂 launch/: contains launch files
│   ├── 📂 scripts/: contains code for environment, policies, and utilities
│   └── 📂 temp/: stores models, logs, and results
├── 📂 drl_agent_description/: contains robot description files, models, and URDFs
│   ├── 📂 launch/: launch files for agent description
│   ├── 📂 meshes/: 3D models of the robot
│   ├── 📂 models/: contains specific model files for kinect sensors
│   └── 📂 urdf/: URDF files for camera, laser, and robot description
├── 📂 drl_agent_gazebo/: contains Gazebo simulation configuration and world files
│   ├── 📂 config/: simulation and SLAM configuration files
│   ├── 📂 launch/: Gazebo launch files for various setups
│   ├── 📂 models/: Gazebo models used in the simulation
│   └── 📂 worlds/: simulation worlds for training and testing environments
├── 📂 drl_agent_interfaces/: custom action, message, and service definitions
│   ├── 📂 action/: defines DRL session actions
│   ├── 📂 msg/: empty for now
│   └── 📂 srv/: service definitions for environment and robot interactions
├── 📂 velodyne_simulator/: Velodyne LiDAR simulation setup
```

## Requirements

- Ubuntu 22.04
- ROS2 Humble
- Gazebo
- gazebo_ros_pkgs: `sudo apt install ros-humble-gazebo-*`
- PyTorch 2.3.1
- Nav2 Map Server: `sudo apt install ros-humble-nav2-map-server ros-humble-nav2-lifecycle-manager`
- Python dependencies: `pip install -r requirements.txt`

## Build

```bash
mkdir -p ~/drl_agent_ws/src
cd ~/drl_agent_ws/src
git clone https://github.com/fomalhaut251/micromouse-gazebo-drl.git .
cd ~/drl_agent_ws
rosdep install --from-path src -yi --rosdistro humble
colcon build
echo 'export DRL_AGENT_SRC_PATH=~/drl_agent_ws/src/' >> ~/.bashrc
source ~/.bashrc
```

## Quick Start

### Terminal 1: Launch Simulation

```bash
cd ~/drl_agent_ws
source install/setup.bash
ros2 launch drl_agent_gazebo simulation.launch.py
```

### Terminal 2: Launch Environment Interface

```bash
cd ~/drl_agent_ws
source install/setup.bash
ros2 run drl_agent environment.py
```

### Terminal 3: Launch Map Server (Optional)

```bash
cd ~/drl_agent_ws
source install/setup.bash
ros2 launch drl_agent_gazebo map_server.launch.py
```

### Terminal 4: Run Agent

**Train:**
```bash
cd ~/drl_agent_ws
source install/setup.bash
ros2 run drl_agent train_td7_agent.py
```

**Test:**
```bash
cd ~/drl_agent_ws
source install/setup.bash
ros2 run drl_agent test_td7_agent.py
```

**Manual Control:**
```bash
cd ~/drl_agent_ws
source install/setup.bash
python3 src/drl_agent/scripts/policy/keyboard_test_agent.py
```

## Troubleshooting

### /gazebo/set_entity_state not available

If the environment node keeps waiting for `/gazebo/set_entity_state`, ensure your `.world` file includes the Gazebo ROS state plugin so the service is provided:

```xml
<plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
	<ros>
		<namespace>/gazebo</namespace>
	</ros>
</plugin>
```

Example world file: `drl_agent_gazebo/worlds/museum1.world`.

## Configuration

- **Test Settings:** `drl_agent/config/test_config.yaml`
- **Training Settings:** `drl_agent/config/train_config.yaml`
- **World File:** `drl_agent_gazebo/worlds/museum1.world`
