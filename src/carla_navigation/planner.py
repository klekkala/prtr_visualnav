#!/usr/bin/env python

import rospy

# misc. external lib imports
from carla_navigation.msg import FloatList
from carla_navigation.msg import FloatArray

from geometry_msgs.msg import PoseStamped


class Planner:
    def __init__(self):
        self.way_points = None    # global plan
        self.marker_pub = None
        self.plan_pub = None
        self.wait_for_teb = False

    def run(self):
        # Initialize Node
        rospy.init_node('planner', anonymous=True)

        # subscriber
        goal_sub = rospy.Subscriber(
            '/goal', PoseStamped, self.goal_callback)

        # publisher
        self.plan_pub = rospy.Publisher('/plan', FloatArray, queue_size=1)

        rospy.spin()

    def goal_callback(self, goal):
        self.goal = (goal.pose.position.x, goal.pose.position.y)
        print("Navigating to (" + str(self.goal[0]) + ", " + str(self.goal[1]) + "). Computing global path..")
        # compute global plan by astar
        # global_planner = Astar(curr_pos, 0, goal, lmap, self.tf_buffer)
        # self.way_points = global_planner.plan()
        self.way_points = [self.goal]

        # create messages sent to the controller
        pts_array = FloatArray()
        for wp in self.way_points:
            float_list = FloatList()
            float_list.elements = [float(wp[0]), float(wp[1])]
            pts_array.lists.append(float_list)

        while self.plan_pub.get_num_connections() < 1:
            continue

        self.plan_pub.publish(pts_array)
        print("Global path sent to the controller..")


if __name__ == "__main__":
    planner = Planner()
    planner.run()
