import xml.etree.ElementTree as ET
import math

def optimize_world(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    world = root.find('world')

    walls = []
    models_to_remove = []

    # 1. 提取所有墙壁信息
    for model in world.findall('model'):
        name = model.get('name')
        if name and name.startswith('wall_'):
            pose_str = model.find('pose').text
            vals = [float(x) for x in pose_str.split()]
            x, y, z, roll, pitch, yaw = vals
            
            # 判断是否垂直 (Yaw 接近 1.57 或 -1.57)
            is_vertical = abs(yaw) > 0.5 
            
            walls.append({
                'x': x, 'y': y, 'z': z, 
                'vertical': is_vertical,
                'original_element': model
            })
            models_to_remove.append(model)

    # 移除旧模型
    for model in models_to_remove:
        world.remove(model)

    # 2. 排序与分组
    # 如果是水平墙: 先按Y(行)排, 再按X(列)排
    # 如果是垂直墙: 先按X(列)排, 再按Y(行)排
    walls.sort(key=lambda w: (w['vertical'], w['x'] if w['vertical'] else w['y'], w['y'] if w['vertical'] else w['x']))

    merged_walls = []
    if not walls:
        print("未找到墙壁模型！")
        return

    current_segment = [walls[0]]

    for i in range(1, len(walls)):
        prev = current_segment[-1]
        curr = walls[i]

        is_connected = False
        if prev['vertical'] == curr['vertical']:
            if not prev['vertical']: # --- 水平墙 ---
                # Y 必须相同 (同一直线)，X 必须相邻 (距离约 1.0)
                if abs(prev['y'] - curr['y']) < 0.05 and abs(curr['x'] - prev['x'] - 1.0) < 0.05:
                    is_connected = True
            else: # --- 垂直墙 ---
                # X 必须相同 (同一直线)，Y 必须相邻 (距离约 1.0)
                if abs(prev['x'] - curr['x']) < 0.05 and abs(curr['y'] - prev['y'] - 1.0) < 0.05:
                    is_connected = True
        
        if is_connected:
            current_segment.append(curr)
        else:
            merged_walls.append(current_segment)
            current_segment = [curr]
    
    merged_walls.append(current_segment)

    # 3. 生成新墙壁
    print(f"优化前墙壁数量: {len(walls)}")
    print(f"优化后墙壁数量: {len(merged_walls)}")

    for idx, segment in enumerate(merged_walls):
        first = segment[0]
        last = segment[-1]
        count = len(segment)
        
        # 计算新的中心点
        new_x = (first['x'] + last['x']) / 2.0
        new_y = (first['y'] + last['y']) / 2.0
        new_z = first['z']
        
        # 【关键修正】：统一尺寸逻辑
        # 无论横竖，我们都创建一个“沿本地X轴延伸”的长方体
        # 长度 = count * 1.0，厚度 = 0.066
        size_x = count * 1.0
        size_y = 0.066
        
        # 通过 Yaw 旋转来决定它在世界坐标系是横是竖
        if first['vertical']:
            yaw = 1.570796  # 垂直墙：旋转90度
        else:
            yaw = 0.0       # 水平墙：不旋转

        # 创建 XML 节点
        new_model = ET.SubElement(world, 'model', {'name': f'optimized_wall_{idx}'})
        static = ET.SubElement(new_model, 'static')
        static.text = '1'
        pose = ET.SubElement(new_model, 'pose')
        pose.text = f"{new_x} {new_y} {new_z} 0 0 {yaw}"
        
        link = ET.SubElement(new_model, 'link', {'name': 'link'})
        
        # Collision
        collision = ET.SubElement(link, 'collision', {'name': 'collision'})
        geo_c = ET.SubElement(collision, 'geometry')
        box_c = ET.SubElement(geo_c, 'box')
        size_c = ET.SubElement(box_c, 'size')
        size_c.text = f"{size_x} {size_y} 0.5"
        
        # Visual
        visual = ET.SubElement(link, 'visual', {'name': 'visual'})
        geo_v = ET.SubElement(visual, 'geometry')
        box_v = ET.SubElement(geo_v, 'box')
        size_v = ET.SubElement(box_v, 'size')
        size_v.text = f"{size_x} {size_y} 0.5"
        
        material = ET.SubElement(visual, 'material')
        script = ET.SubElement(material, 'script')
        uri = ET.SubElement(script, 'uri')
        uri.text = "file://media/materials/scripts/gazebo.material"
        name = ET.SubElement(script, 'name')
        name.text = "Gazebo/Grey"

    tree.write(output_file)
    print(f"已生成优化后的文件: {output_file}")

if __name__ == "__main__":
    optimize_world('museum.world', 'museum_optimized.world')