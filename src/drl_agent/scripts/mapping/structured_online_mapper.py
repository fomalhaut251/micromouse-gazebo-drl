#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class StructuredOnlineMapper(Node):
    """从现成 2D 占据栅格中提取 16x16 单元格墙壁。"""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def __init__(self):
        super().__init__("structured_online_mapper")

        self.declare_parameter("generate_local_map", True)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/front_laser/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("walls_topic", "/structured_walls")
        self.declare_parameter("grid_size", 16)
        self.declare_parameter("x_min", -8.0)
        self.declare_parameter("y_min", -8.0)
        self.declare_parameter("cell_size", 1.0)
        self.declare_parameter("cell_pixels", 10)
        self.declare_parameter("sensor_offset_x", 0.125)
        self.declare_parameter("sensor_offset_y", 0.0)
        self.declare_parameter("min_point_range", 0.15)
        self.declare_parameter("max_point_range", 2.0)
        self.declare_parameter("laser_no_hit_margin", 0.05)
        self.declare_parameter("occ_hit_threshold", 2)
        self.declare_parameter("wall_strip_px", 2)
        self.declare_parameter("occupied_value_threshold", 50)
        self.declare_parameter("wall_occ_ratio_threshold", 0.25)
        self.declare_parameter("wall_known_ratio_threshold", 0.20)
        self.declare_parameter("min_wall_occ_pixels", 3)
        self.declare_parameter("wall_dilate_pixels", 0)
        self.declare_parameter("publish_rate", 2.0)
        self.declare_parameter("wall_marker_width", 0.05)
        self.declare_parameter("wall_marker_height", 0.12)

        self.generate_local_map = bool(self.get_parameter("generate_local_map").value)
        self.odom_topic = self.get_parameter("odom_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.map_topic = self.get_parameter("map_topic").value
        self.walls_topic = self.get_parameter("walls_topic").value
        self.grid_size = int(self.get_parameter("grid_size").value)
        self.x_min = float(self.get_parameter("x_min").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.cell_size = float(self.get_parameter("cell_size").value)
        self.cell_pixels = max(4, int(self.get_parameter("cell_pixels").value))
        self.sensor_offset_x = float(self.get_parameter("sensor_offset_x").value)
        self.sensor_offset_y = float(self.get_parameter("sensor_offset_y").value)
        self.min_point_range = float(self.get_parameter("min_point_range").value)
        self.max_point_range = float(self.get_parameter("max_point_range").value)
        self.laser_no_hit_margin = float(self.get_parameter("laser_no_hit_margin").value)
        self.occ_hit_threshold = max(1, int(self.get_parameter("occ_hit_threshold").value))
        self.wall_strip_px = max(1, int(self.get_parameter("wall_strip_px").value))
        self.occupied_value_threshold = int(
            self.get_parameter("occupied_value_threshold").value
        )
        self.wall_occ_ratio_threshold = float(
            self.get_parameter("wall_occ_ratio_threshold").value
        )
        self.wall_known_ratio_threshold = float(
            self.get_parameter("wall_known_ratio_threshold").value
        )
        self.min_wall_occ_pixels = max(
            1, int(self.get_parameter("min_wall_occ_pixels").value)
        )
        self.wall_dilate_pixels = max(
            0, int(self.get_parameter("wall_dilate_pixels").value)
        )
        self.publish_rate = max(0.5, float(self.get_parameter("publish_rate").value))
        self.wall_marker_width = float(self.get_parameter("wall_marker_width").value)
        self.wall_marker_height = float(
            self.get_parameter("wall_marker_height").value
        )

        self.map_lock = threading.Lock()
        self.latest_map = None
        self.wall_confirmed = np.zeros((self.grid_size, self.grid_size, 4), dtype=bool)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.has_pose = False

        self.map_resolution = self.cell_size / float(self.cell_pixels)
        self.map_width = self.grid_size * self.cell_pixels
        self.map_height = self.grid_size * self.cell_pixels
        self.local_known = np.zeros((self.map_height, self.map_width), dtype=bool)
        self.local_occ_hits = np.zeros((self.map_height, self.map_width), dtype=np.uint16)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        if self.generate_local_map:
            self.create_subscription(Odometry, self.odom_topic, self._odom_cb, sensor_qos)
            self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, sensor_qos)
        else:
            self.create_subscription(OccupancyGrid, self.map_topic, self._map_cb, map_qos)

        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, map_qos)
        self.walls_pub = self.create_publisher(MarkerArray, self.walls_topic, map_qos)
        self.create_timer(1.0 / self.publish_rate, self._publish_walls)

        self.get_logger().info(
            "Structured wall extractor started: "
            f"local_map={self.generate_local_map}, map_topic={self.map_topic}, walls_topic={self.walls_topic}, "
            f"grid={self.grid_size}x{self.grid_size}, cell={self.cell_size:.3f}m"
        )

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self.map_lock:
            self.pose_x = msg.pose.pose.position.x
            self.pose_y = msg.pose.pose.position.y
            self.pose_yaw = yaw
            self.has_pose = True

    def _scan_cb(self, msg: LaserScan):
        with self.map_lock:
            if not self.has_pose:
                return
            px, py, yaw = self.pose_x, self.pose_y, self.pose_yaw

        sx = px + math.cos(yaw) * self.sensor_offset_x - math.sin(yaw) * self.sensor_offset_y
        sy = py + math.sin(yaw) * self.sensor_offset_x + math.cos(yaw) * self.sensor_offset_y

        angle = msg.angle_min
        for r in msg.ranges:
            if not math.isfinite(r):
                angle += msg.angle_increment
                continue
            if r < msg.range_min or r > msg.range_max:
                angle += msg.angle_increment
                continue

            no_hit = r >= (msg.range_max - self.laser_no_hit_margin)
            used_range = min(r, self.max_point_range)
            if no_hit:
                used_range = min(msg.range_max, self.max_point_range)
            if used_range < self.min_point_range:
                angle += msg.angle_increment
                continue

            world_ang = yaw + angle
            ex = sx + used_range * math.cos(world_ang)
            ey = sy + used_range * math.sin(world_ang)
            endpoint_hit = (not no_hit) and (r <= self.max_point_range)

            with self.map_lock:
                self._integrate_ray(sx, sy, ex, ey, endpoint_hit)

            angle += msg.angle_increment

    def _map_cb(self, msg: OccupancyGrid):
        with self.map_lock:
            self.latest_map = msg

    def _publish_walls(self):
        with self.map_lock:
            if self.generate_local_map:
                self.latest_map = self._build_local_map_msg()
                self.map_pub.publish(self.latest_map)
            if self.latest_map is None:
                return
            self._extract_structured_walls_from_map(self.latest_map)
            marker_array = self._build_wall_markers(self.latest_map.header.frame_id)

        self.walls_pub.publish(marker_array)

    def _world_to_pixel(self, x: float, y: float):
        ix = int(math.floor((x - self.x_min) / self.map_resolution))
        iy = int(math.floor((y - self.y_min) / self.map_resolution))
        ix = max(0, min(self.map_width - 1, ix))
        iy = max(0, min(self.map_height - 1, iy))
        return ix, iy

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int):
        pts = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            pts.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return pts

    def _integrate_ray(self, sx: float, sy: float, ex: float, ey: float, endpoint_hit: bool):
        s_ix, s_iy = self._world_to_pixel(sx, sy)
        e_ix, e_iy = self._world_to_pixel(ex, ey)
        cells = self._bresenham(s_ix, s_iy, e_ix, e_iy)
        if not cells:
            return
        for ix, iy in cells:
            self.local_known[iy, ix] = True

        free_cells = cells[:-1] if endpoint_hit else cells
        for ix, iy in free_cells:
            if self.local_occ_hits[iy, ix] > 0:
                self.local_occ_hits[iy, ix] = self.local_occ_hits[iy, ix] - 1

        if endpoint_hit:
            end_ix, end_iy = cells[-1]
            self.local_occ_hits[end_iy, end_ix] = min(65535, int(self.local_occ_hits[end_iy, end_ix]) + 1)

    def _build_local_map_msg(self):
        occ = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        occ[self.local_known] = 0
        occ[self.local_occ_hits >= self.occ_hit_threshold] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.x_min
        msg.info.origin.position.y = self.y_min
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = occ.reshape(-1).tolist()
        return msg

    def _extract_structured_walls_from_map(self, map_msg: OccupancyGrid):
        width = map_msg.info.width
        height = map_msg.info.height
        if width == 0 or height == 0:
            self.wall_confirmed.fill(False)
            return

        grid = np.array(map_msg.data, dtype=np.int16).reshape((height, width))
        known = grid >= 0
        occupied = grid >= self.occupied_value_threshold

        map_resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y
        pix_per_cell = max(1, int(round(self.cell_size / map_resolution)))
        strip = min(self.wall_strip_px, pix_per_cell)

        self.wall_confirmed.fill(False)

        def world_cell_bounds(u: int, v: int):
            wx0 = self.x_min + u * self.cell_size
            wy0 = self.y_min + v * self.cell_size
            px0 = int(round((wx0 - origin_x) / map_resolution))
            py0 = int(round((wy0 - origin_y) / map_resolution))
            px1 = int(round((wx0 + self.cell_size - origin_x) / map_resolution))
            py1 = int(round((wy0 + self.cell_size - origin_y) / map_resolution))
            px0 = max(0, min(width, px0))
            py0 = max(0, min(height, py0))
            px1 = max(0, min(width, px1))
            py1 = max(0, min(height, py1))
            return px0, py0, px1, py1

        def decide(mask):
            total = int(np.count_nonzero(mask))
            if total == 0:
                return False
            known_count = int(np.count_nonzero(np.logical_and(mask, known)))
            if known_count == 0:
                return False
            known_ratio = known_count / float(total)
            if known_ratio < self.wall_known_ratio_threshold:
                return False
            occ_count = int(np.count_nonzero(np.logical_and(mask, occupied)))
            if occ_count < self.min_wall_occ_pixels:
                return False
            occ_ratio = occ_count / float(total)
            return occ_ratio >= self.wall_occ_ratio_threshold

        for v in range(self.grid_size):
            for u in range(self.grid_size):
                x0, y0, x1, y1 = world_cell_bounds(u, v)
                if x1 <= x0 or y1 <= y0:
                    continue

                west_mask = np.zeros((height, width), dtype=bool)
                west_mask[y0:y1, x0:min(x0 + strip, x1)] = True
                east_mask = np.zeros((height, width), dtype=bool)
                east_mask[y0:y1, max(x1 - strip, x0):x1] = True
                south_mask = np.zeros((height, width), dtype=bool)
                south_mask[y0:min(y0 + strip, y1), x0:x1] = True
                north_mask = np.zeros((height, width), dtype=bool)
                north_mask[max(y1 - strip, y0):y1, x0:x1] = True

                self.wall_confirmed[v, u, self.WEST] = decide(west_mask)
                self.wall_confirmed[v, u, self.EAST] = decide(east_mask)
                self.wall_confirmed[v, u, self.SOUTH] = decide(south_mask)
                self.wall_confirmed[v, u, self.NORTH] = decide(north_mask)

        for v in range(self.grid_size):
            for u in range(self.grid_size):
                if v + 1 < self.grid_size:
                    val = (
                        self.wall_confirmed[v, u, self.NORTH]
                        or self.wall_confirmed[v + 1, u, self.SOUTH]
                    )
                    self.wall_confirmed[v, u, self.NORTH] = val
                    self.wall_confirmed[v + 1, u, self.SOUTH] = val
                if u + 1 < self.grid_size:
                    val = (
                        self.wall_confirmed[v, u, self.EAST]
                        or self.wall_confirmed[v, u + 1, self.WEST]
                    )
                    self.wall_confirmed[v, u, self.EAST] = val
                    self.wall_confirmed[v, u + 1, self.WEST] = val

    def _build_wall_markers(self, frame_id: str):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id or "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "structured_walls_delete"
        delete_marker.id = 999999
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        line_marker = Marker()
        line_marker.header.frame_id = delete_marker.header.frame_id
        line_marker.header.stamp = delete_marker.header.stamp
        line_marker.ns = "structured_walls"
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = self.wall_marker_width
        line_marker.color.a = 1.0
        line_marker.color.r = 0.95
        line_marker.color.g = 0.3
        line_marker.color.b = 0.1

        for v in range(self.grid_size):
            y0 = self.y_min + v * self.cell_size
            y1 = y0 + self.cell_size
            for u in range(self.grid_size):
                x0 = self.x_min + u * self.cell_size
                x1 = x0 + self.cell_size

                if self.wall_confirmed[v, u, self.WEST]:
                    line_marker.points.extend([
                        Point(x=x0, y=y0, z=self.wall_marker_height),
                        Point(x=x0, y=y1, z=self.wall_marker_height),
                    ])
                if self.wall_confirmed[v, u, self.EAST]:
                    line_marker.points.extend([
                        Point(x=x1, y=y0, z=self.wall_marker_height),
                        Point(x=x1, y=y1, z=self.wall_marker_height),
                    ])
                if self.wall_confirmed[v, u, self.SOUTH]:
                    line_marker.points.extend([
                        Point(x=x0, y=y0, z=self.wall_marker_height),
                        Point(x=x1, y=y0, z=self.wall_marker_height),
                    ])
                if self.wall_confirmed[v, u, self.NORTH]:
                    line_marker.points.extend([
                        Point(x=x0, y=y1, z=self.wall_marker_height),
                        Point(x=x1, y=y1, z=self.wall_marker_height),
                    ])

        marker_array.markers.append(line_marker)
        return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = StructuredOnlineMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
