## ROS Visual Navigation
This ROS package integrates pretrained models and a trained policy into a real-world robot, enabling advanced navigation and perception capabilities. 
The package is designed to support both reinforcement learning policy and local trajectory planner for effective path planning. 

## Prerequisites
To utilize this package, ensure you have the following prerequisites installed:
1. [ROS Noetic](http://wiki.ros.org/noetic/Installation)
2. Python Packages: \
   • Ray \
   • PyTorch
3. [LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM): This package is crucial for LiDAR-based localization. Ensure you have it installed and configured properly.
4. Velodyne LiDAR sensor
5. [TEB Local Planner](http://wiki.ros.org/teb_local_planner)

## Usage
1. lego_loam: roslaunch lego_loam run.launch
2. carla_navigation: roslaunch carla_navigation carla_navigation.launch
## Support
For issues and questions, raise an issue in the repository or contact the maintainers.
