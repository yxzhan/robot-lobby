import py_trees
import math
from lobby import robots, set_goal

class DynamicFollower(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.follow_distance = 1.0

    def update(self):
        # Get all robots that are currently capable of driving
        all_robots = [r for r in robots() if r['drives']]
        
        # If there are fewer than 2 robots, no one can follow anyone
        if len(all_robots) < 2:
            return py_trees.common.Status.SUCCESS

        # The first robot in the list is the leader.
        # Starting from the second robot, each one follows the one in front of it.
        for i in range(1, len(all_robots)):
            follower_name = all_robots[i]['name']
            leader = all_robots[i-1]
            
            # Get the leader's current pose
            lx = leader['pose']['x']
            ly = leader['pose']['y']
            lyaw = leader['pose']['yaw']
            
            # Calculate the target position behind the leader based on its heading (yaw)
            tx = lx - math.cos(lyaw) * self.follow_distance
            ty = ly - math.sin(lyaw) * self.follow_distance
            
            # Set the goal for the follower to move to that position and face the same way
            set_goal(follower_name, tx, ty, lyaw)
            
        return py_trees.common.Status.RUNNING

# Define the root of the behavior tree as "DynamicFollower"
root = DynamicFollower("DynamicFollower")