#!/usr/bin/env python
import math
import rospy

from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Point32, PoseStamped
from nav_msgs.msg import Path


def publish_obstacle_msg(grid_map):
    obstacle_msg = ObstacleArrayMsg()
    obstacle_msg.header.stamp = rospy.Time.now()
    obstacle_msg.header.frame_id = "map"

    # Add point obstacles
    cnt = 0
    height, width = len(grid_map), len(grid_map)
    for i in range(0, height-1):
        for j in range(0, width):
            if grid_map[i][j] < 1:
                if (j == 0 and grid_map[i][1] > 10) or (j == width-1 and grid_map[i][width-2] > 10):
                    continue
                obstacle_msg.obstacles.append(ObstacleMsg())
                obstacle_msg.obstacles[cnt].id = cnt
                obstacle_msg.obstacles[cnt].radius = 7
                obstacle_msg.obstacles[cnt].polygon.points = [Point32()]

                obstacle_msg.obstacles[cnt].polygon.points[0].x = (j - width/2) *7
                obstacle_msg.obstacles[cnt].polygon.points[0].y = (height-i) * 7
                obstacle_msg.obstacles[cnt].polygon.points[0].z = 0

                cnt += 1
    for i in range(height, height+3):
        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[cnt].id = cnt
        obstacle_msg.obstacles[cnt].radius = 7
        obstacle_msg.obstacles[cnt].polygon.points = [Point32()]

        obstacle_msg.obstacles[cnt].polygon.points[0].x = - width / 2 * 7
        obstacle_msg.obstacles[cnt].polygon.points[0].y = (height - i) *7
        obstacle_msg.obstacles[cnt].polygon.points[0].z = 0

        cnt += 1

        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[cnt].id = cnt
        obstacle_msg.obstacles[cnt].radius = 7
        obstacle_msg.obstacles[cnt].polygon.points = [Point32()]

        obstacle_msg.obstacles[cnt].polygon.points[0].x = (width /2 - 1) * 7
        obstacle_msg.obstacles[cnt].polygon.points[0].y = (height - i) *7
        obstacle_msg.obstacles[cnt].polygon.points[0].z = 0

        cnt += 1

    obs_pub = rospy.Publisher('/teb_planner/obstacles', ObstacleArrayMsg, queue_size=1)

    while obs_pub.get_num_connections() < 1:
        continue

    obs_pub.publish(obstacle_msg)

    return

def publish_waypoints_msg(aux):
    via_points_msg = Path()
    via_points_msg.header.stamp = rospy.Time.now()
    via_points_msg.header.frame_id = "map"
        # Add via-points
    point = PoseStamped()
    aux = round(aux * 5) / 5
    rad = aux * math.pi


    point.pose.position.x = -100 * math.sin(rad)
    point.pose.position.y = 100 * math.cos(rad)
    via_points_msg.poses.append(point)

    wp_pub = rospy.Publisher('/teb_planner/via_points', Path, queue_size=1)
    while wp_pub.get_num_connections() < 1:
        continue
    wp_pub.publish(via_points_msg)

    return

class TEB:
    def __init__(self, waypoints, map_, aux):
        self.waypoints = waypoints
        self.grid_map = map_
        self.aux = aux

    def plan(self):
        publish_obstacle_msg(self.grid_map)
        publish_waypoints_msg(self.aux)

        return

