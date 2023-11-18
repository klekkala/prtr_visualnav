import rospy
import rosbag
from sensor_msgs.msg import CompressedImage
from carla_navigation.msg import TimedTwist
import time

rospy.init_node('pub', anonymous=True)

bag = rosbag.Bag('/home2/carla/2023_08_20/cam1/test_2023-08-20-01-07-49.bag')

image_pub = rospy.Publisher('image', CompressedImage, queue_size=10)
action_pub = rospy.Publisher('action', TimedTwist, queue_size=10)

while image_pub.get_num_connections() < 1 and action_pub.get_num_connections() < 1:
    continue

prev_t = -1
for topic, msg, t in bag.read_messages(topics=['/cam1/color/image_raw/compressed', '/cmd_vel']):
    if topic == "/cam1/color/image_raw/compressed":
        msg.header.stamp = t
        image_pub.publish(msg)
    elif topic == "/cmd_vel":
        timed_msg = TimedTwist()
        timed_msg.header.stamp = t
        timed_msg.twist = msg
        action_pub.publish(timed_msg)
    if prev_t > 0:
        time.sleep((t.to_nsec() - prev_t) / 1000000000.0)
    prev_t = t.to_nsec()


bag.close()
