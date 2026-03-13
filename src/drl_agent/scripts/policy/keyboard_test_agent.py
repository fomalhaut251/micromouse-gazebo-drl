#!/usr/bin/env python3
import sys
import os
import select
import termios
import tty
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

# 引入环境接口
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '../environment')
sys.path.append(env_path)

from environment_interface import EnvInterface
from drl_agent_interfaces.srv import Step

# 定义按键映射
# 格式: '按键': (线速度 v, 角速度 w)
key_bindings = {
    'w': (0.5, 0.0),   # 前进
    'x': (-0.5, 0.0),  # 后退
    'a': (0.05, -0.5),   # 左转 (带一点线速度以防原地打滑)
    'd': (0.05, 0.5),  # 右转
    's': (0.0, 0.0),   # 停止/原地等待一步
    'W': (1.0, 0.0),   # 高速前进 (Shift+w)
    'A': (0.0, -1.0),   # 快速左旋 (Shift+a)
    'D': (0.0, 1.0),  # 快速右旋 (Shift+d)
}

msg = """
---------------------------
键盘控制测试模式 (Step-based)
---------------------------
按键说明:
   w : 前进 (0.5 m/s)
   x : 后退 (-0.5 m/s)
   a : 左转
   d : 右转
   s : 停止/空过一步
   
   (按住 Shift 可以加速 W/A/D)

CTRL-C 或 q 退出
---------------------------
注意: 每按一次键，环境会执行一个 Step (前进 time_delta 秒)。
如果不按键，仿真世界将保持暂停状态。
---------------------------
"""

class KeyboardTestAgent(EnvInterface):
    def __init__(self):
        super().__init__("keyboard_test_agent")
        self.settings = termios.tcgetattr(sys.stdin)
        self.get_logger().info("Keyboard Test Agent Initialized.")

    def getKey(self):
        """读取单个按键输入，不阻塞"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def step(self, action):
        """
        重写 step 方法，直接发送物理数值，跳过神经网络的归一化处理。
        """
        request = Step.Request()
        # 直接使用 action [v, w]
        request.action = np.array(action, dtype=np.float32).tolist()
        
        while not self.step_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /step not available, waiting again...")
        
        try:
            future = self.step_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            return response.state, response.reward, response.done, response.target
        except Exception as e:
            self.get_logger().error(f"Service call /step failed: {e}")
            return None, 0.0, False, False

    def run(self):
        print(msg)
        self.reset()
        
        try:
            while rclpy.ok():
                key = self.getKey()
                
                if key in key_bindings.keys():
                    v, w = key_bindings[key]
                    print(f"Action: v={v}, w={w}", end="")
                    
                    state, reward, done, target = self.step([v, w])
                    
                    print(f" | Reward: {reward:.2f} | Done: {done} | Target: {target}")

                    if done:
                        print("\n=== Episode Ended (Collision or Goal) ===\nResetting environment...\n")
                        self.reset()
                        print(msg)
                        
                elif key == 'q' or key == '\x03': # q 或 Ctrl-C
                    break
                else:
                    # 如果没有有效按键，什么都不做，相当于世界暂停
                    pass

        except Exception as e:
            print(e)
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

def main(args=None):
    rclpy.init(args=args)
    agent = KeyboardTestAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()