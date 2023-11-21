## Pose extraction software

Majority of novel code is contained in extract_pose.py. Several other .py files are imported from the PathPlanningAstar repo at this [link](https://github.com/sanchithaseshadri/PathPlanningAstar.git")

##### How to run
Run `python3 extract_pose.py`. Ensure that the needed rostopics are available and that `roscore` is running.

##### Nodes
Subscribes to:
* /base_odometry/odom 
* /base_scan
* /localmap

Publishes to:
* /cmd_vel

##### Comments
Changes to `astar.py`:
* Modified I modified the `MOVES`, `TOLERANCE`, and `G_MULTIPLIER` parameters. These can be further played around with.

Changes to `node.py`:
There were several transformations taking place that were unneeded for our application. 
* I changed the `is_valid_move` and `is_valid` functions to omit transformations.
* I added the `__lt__` method to `Node` to allow `Node` objects to be compared when performing `heapq` operations. 

Changes to buildMap.py:
* Reworked class initializer to work with arbitrary occupancy map, as prior implementation used a hard-coded one. 

Additionally, a `TODO` note has been added to `safegoto.py` where all of the subscriber/publisher nodes are handled. These (in particular the `rospy.init_node()`, line 21 will have to be modified or removed)
