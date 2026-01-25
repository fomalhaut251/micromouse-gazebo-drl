import xml.etree.ElementTree as ET
import numpy as np
import cv2
import yaml
import math
import os

def parse_world_to_map(world_file, map_name="my_map", resolution=0.05, border_margin=2.0):
    """
    解析 Gazebo world 文件并生成 2D 栅格地图
    
    :param world_file: .world 文件路径
    :param map_name: 输出的地图文件名（不带后缀）
    :param resolution: 地图分辨率 (米/像素)
    :param border_margin: 地图边缘留白 (米)
    """
    print(f"正在解析: {world_file} ...")
    
    try:
        tree = ET.parse(world_file)
        root = tree.getroot()
    except Exception as e:
        print(f"错误: 无法解析文件 {world_file}: {e}")
        return

    # 1. 提取所有墙体信息
    walls = []
    
    # 查找所有的 <model> 标签
    # 注意：根据你的文件结构，可能需要根据层级调整，这里使用 iter 遍历所有子节点
    for model in root.iter('model'):
        model_name = model.get('name', '')
        
        # 只处理名字里带 'wall' 的模型，或者你可以根据需要修改过滤条件
        # 你上传的文件中墙体命名格式为 'wall_xxx'
        if 'wall' not in model_name.lower():
            continue

        # 获取 Pose (x y z R P Y)
        pose_elem = model.find('pose')
        if pose_elem is None:
            # 有时 pose 在 link 里面
            link = model.find('link')
            if link is not None:
                pose_elem = link.find('pose')
        
        if pose_elem is None:
            continue

        pose_str = pose_elem.text.split()
        x, y = float(pose_str[0]), float(pose_str[1])
        yaw = float(pose_str[5])

        # 获取几何尺寸 (Box Size)
        # 路径通常是 link -> collision -> geometry -> box -> size
        # 或者 link -> visual -> geometry -> box -> size
        size_elem = None
        for link in model.iter('link'):
            for geo in link.iter('geometry'):
                box = geo.find('box')
                if box is not None:
                    size_elem = box.find('size')
                    break
            if size_elem is not None:
                break
        
        if size_elem is None:
            continue
            
        size_str = size_elem.text.split()
        size_x, size_y = float(size_str[0]), float(size_str[1])
        
        walls.append({
            'x': x, 'y': y, 'yaw': yaw,
            'size_x': size_x, 'size_y': size_y
        })

    if not walls:
        print("警告: 没有在文件中找到任何墙体 (wall_*)。")
        return

    print(f"找到 {len(walls)} 个墙体对象。")

    # 2. 计算地图边界
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # 简单的边界估算（未考虑旋转，为了保险起见加上了 margin）
    for w in walls:
        radius = math.sqrt(w['size_x']**2 + w['size_y']**2) / 2
        min_x = min(min_x, w['x'] - radius)
        max_x = max(max_x, w['x'] + radius)
        min_y = min(min_y, w['y'] - radius)
        max_y = max(max_y, w['y'] + radius)

    # 加上留白
    min_x -= border_margin
    max_x += border_margin
    min_y -= border_margin
    max_y += border_margin

    width_m = max_x - min_x
    height_m = max_y - min_y
    
    width_px = int(math.ceil(width_m / resolution))
    height_px = int(math.ceil(height_m / resolution))

    print(f"地图尺寸: {width_m:.2f}m x {height_m:.2f}m ({width_px} x {height_px} 像素)")

    # 3. 绘制地图 (255=空闲, 0=占用)
    # 初始化全白地图
    grid_map = np.ones((height_px, width_px), dtype=np.uint8) * 255

    for w in walls:
        # 计算四个角点
        cx, cy = w['x'], w['y']
        sx, sy = w['size_x'], w['size_y']
        angle = w['yaw']

        # 矩形相对中心的四个角点
        corners = np.array([
            [ sx/2,  sy/2],
            [-sx/2,  sy/2],
            [-sx/2, -sy/2],
            [ sx/2, -sy/2]
        ])

        # 旋转矩阵
        R = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle),  math.cos(angle)]
        ])

        # 旋转并平移
        rotated_corners = np.dot(corners, R.T)
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy

        # 转换为像素坐标
        # u = (x - min_x) / res
        # v = height - (y - min_y) / res  (图像坐标系Y向下，世界坐标系Y向上，需要翻转)
        pixel_corners = []
        for pt in rotated_corners:
            u = int((pt[0] - min_x) / resolution)
            v = int(height_px - (pt[1] - min_y) / resolution)
            pixel_corners.append([u, v])
        
        pixel_corners = np.array(pixel_corners, dtype=np.int32)
        
        # 在图上填充黑色多边形
        cv2.fillPoly(grid_map, [pixel_corners], 0)

    # 4. 保存 PGM 图片
    pgm_filename = f"{map_name}.pgm"
    cv2.imwrite(pgm_filename, grid_map)
    print(f"已保存图片: {pgm_filename}")

    # 5. 保存 YAML 配置文件
    yaml_filename = f"{map_name}.yaml"
    yaml_data = {
        'image': pgm_filename,
        'mode': 'trinary',
        'resolution': resolution,
        'origin': [min_x, min_y, 0.0], # origin 通常指地图左下角的物理坐标
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196
    }

    with open(yaml_filename, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f"已保存配置: {yaml_filename}")

if __name__ == "__main__":
    # 这里的路径需要根据你实际存放 museum.world 的位置修改
    # 假设你把它放在了当前目录，或者指定绝对路径
    # 例如：/home/ubuntu22/drl_agent_ws/src/drl_agent_gazebo/worlds/museum.world
    
    # 自动查找当前目录下的 .world 文件，或者你可以手动指定
    world_path = "museum.world" 
    
    if not os.path.exists(world_path):
        # 尝试去标准路径找
        world_path = "src/drl_agent_gazebo/worlds/museum.world"
    
    if os.path.exists(world_path):
        parse_world_to_map(world_path, map_name="museum_map")
    else:
        print("错误: 找不到 museum.world 文件，请在代码中修改 world_path 路径。")