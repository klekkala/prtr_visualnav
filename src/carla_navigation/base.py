#!/usr/bin/env python

import math
import numpy as np
import rospy
import tf2_ros

import message_filters
from std_msgs.msg import Float64
from std_msgs.msg import Int64
from geometry_msgs.msg import Pose, Point, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_geometry_msgs import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

from geometry_msgs.msg import Twist
from carla_navigation.msg import FloatArray
from teb import TEB
import os
import cv2
from geometry_msgs.msg import Point32, PoseStamped
from nav_msgs.msg import Path
# LINEAR_VELOCITY = 740
# ANGULAR_VELOCITY = 1500
# ROBOT_WIDTH = 0.65
# WHEEL_RADIUS = 0.37

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians


class Base:
    def __init__(self):
        self.way_points = [(10, 10)]
        self.aux = 0
        self.pos = None
        self.enable = True  # True when receive a goal and begin navigation
        self.mk_msg = MarkerArray()
        self.tf_buffer = None

        self.vel_setpoint_pub = None
        self.ang_setpoint_pub = None
        self.vel_state_pub = None
        self.ang_state_pub = None
        self.cmd_pub = None
        self.marker_pub = None

        root = "/home/administrator/img2cmd/anchor"
        self.anchors = []
        for root, subdirs, files in os.walk(root):
            for i, f in enumerate(sorted(files)):
                if ".jpg" in f:
                    im = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
                    im = cv2.resize(im, (10, 10), interpolation=cv2.INTER_LINEAR)
                    _, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
                    self.anchors.append(im)
                    if len(self.anchors) == 599:
                        print(f)

        self.prev_ang = 0

    def run(self):
        rospy.init_node('base', anonymous=True)

        # tf transformer
        self.tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # costmap subscriber
        map_sub = rospy.Subscriber(
            "/occ_map", Int64, self.map_callback)
        # local plan subscriber
        teb_sub = rospy.Subscriber(
            "/teb_planner/teb_poses", PoseArray, self.teb_callback)

        # control effort subscriber
        vel_ctrl_effort_sub = message_filters.Subscriber(
            "/vel/control_effort", Float64)
        ang_ctrl_effort_sub = message_filters.Subscriber(
            "/ang/control_effort", Float64)
        cs = message_filters.ApproximateTimeSynchronizer(
            [vel_ctrl_effort_sub, ang_ctrl_effort_sub], queue_size=10, slop=0.01, allow_headerless=True)
        cs.registerCallback(self.controller_callback)

        # setpoint publisher
        self.vel_setpoint_pub = rospy.Publisher(
            '/vel/setpoint', Float64, queue_size=1)
        self.ang_setpoint_pub = rospy.Publisher(
            '/ang/setpoint', Float64, queue_size=1)
        # state publisher
        self.vel_state_pub = rospy.Publisher(
            '/vel/state', Float64, queue_size=1)
        self.ang_state_pub = rospy.Publisher(
            '/ang/state', Float64, queue_size=1)

        # velocity command publisher
        self.cmd_pub = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)
            
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)

        print("Controller is ready. Waiting for incoming data..")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
               try:
               	trans = tfBuffer.lookup_transform('base_link', 'map', rospy.Time())
               except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                       rate.sleep()
                       continue
		
               self.aux_compute(trans.transform)
               print(self.aux)
               rate.sleep()

    def map_callback(self, 
    map):
        # compute local plan by TEB
        selected_waypoints = None
        id = map.data
        local_planner = TEB(selected_waypoints, self.anchors[id], self.aux)
        local_planner.plan()


        return

    def aux_compute(self, transform):
        tranx = transform.translation.x
        trany = transform.translation.y
        def norm(rad):
          if rad < -math.pi:
            rad += 2*math.pi
          elif rad > math.pi:
            rad -= 2*math.pi
            
          rad = rad / math.pi
          return rad
        x = transform.rotation.x
        y = transform.rotation.y
        z = transform.rotation.z
        w = transform.rotation.w

        orientation = euler_from_quaternion(x, y, z, w)[2]
        dis_v = np.arctan2(self.way_points[0][1]-trany, self.way_points[0][0]-tranx)
        diff = norm(orientation - dis_v)
        
        self.aux = diff

    def teb_callback(self, teb_poses):
        local_waypoints = teb_poses.poses

        if not local_waypoints:
            print("TEB Planner failed to compute a path..")
            return

        # message templates

        # publish state and setpoint
        self.vel_state_pub.publish(0)
        self.ang_state_pub.publish(0)
        for i in range(len(local_waypoints)):
            if self.euclidean_distance((0, 0),
                                       (local_waypoints[i].position.x, local_waypoints[i].position.y)) >= 10:
                # translate local waypoint to ego_vehicle coordinates
                pt = PointStamped()
                pt.header.stamp = rospy.Time(0)
                pt.header.frame_id = "map"
                pt.point.x, pt.point.y, pt.point.z = local_waypoints[i].position.x, local_waypoints[i].position.y, 0

                self.vel_setpoint_pub.publish(float(pt.point.y / 10))
                self.ang_setpoint_pub.publish(float(pt.point.x / 10))

                return

    def controller_callback(self, vel, ang):
        if not self.enable:
            return
        print("forward difference: " + str(vel))
        print("side difference: " + str(ang))
        # publish cmd_vel
        vel_msg = Twist()
        vel_msg.linear.x = vel.data
        vel_msg.angular.z = - ang.data / 2 #0 if np.abs(ang.data) > 0.9 else ang.data

        self.cmd_pub.publish(vel_msg)
        print(vel_msg)

        return

    def euclidean_distance(self, start, goal):
        return math.sqrt(math.pow((goal[0] - start[0]), 2) + math.pow((goal[1] - start[1]), 2))

if __name__ == '__main__':
    base = Base()
    base.run()
    
