import numpy as  np
import pybullet as p

from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
import copy


# Function to check for collisions using PyBullet
# def is_collision_free(pos):
#     drone_col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
#     col_check = p.createMultiBody(0, drone_col_id, basePosition=position)
#     collisions = p.getClosestPoints(col_check, table_id, distance=0.01)
#     p.removeBody(col_check)
#     return not collisions

# def check_collisions(object1, object2):
#     closest_points = p.getClosestPoints(object1, object2, distance=0.2)
#     if closest_points:
#         return True
#     else:
#         return False

MAX_ITER = 1000
STEP_SIZE = 0.1
D_STATIC = 0.1
D_DYNAMIC = 0.1
R_RRT = 0.2
EPS = 0.2

R_COLLISION_SHAPE_DRONE = 0.07

class RRT_star:
    def __init__(self, start_pos, goal_pos, static_obstacles, dynamic_obstacles, d_static=D_STATIC, d_dynamic=D_DYNAMIC, max_iter=MAX_ITER, 
                step_size = STEP_SIZE, r_rrt = R_RRT, r_collision_shape_drone = R_COLLISION_SHAPE_DRONE, eps=0.1, bounds=((-4, 4), (-4, 4), (0, 2))) -> None:
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.d_static = d_static
        self.d_dynamic = d_dynamic
        self.max_iter = max_iter
        self.step_size = step_size
        self.r_rrt = r_rrt
        self.eps = eps
        self.r_collision_shape_drone = r_collision_shape_drone
        self.bounds = bounds

    class Node:
        def __init__(self, position, parent=None):
            self.position = position
            self.parent = parent
            self.cost = 0

    def is_collision_obstacles(self, pos):
        drone_col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.r_collision_shape_drone)
        col_check = p.createMultiBody(0, drone_col_id, basePosition=pos)
        for obst in self.static_obstacles:
            collisions = p.getClosestPoints(col_check, obst, distance=self.d_static)
            if collisions:
                p.removeBody(col_check)
                return True
        p.removeBody(col_check)

        return False

    def  is_collision_drones(self,pos):
        for traj in self.dynamic_obstacles:
            for b_point in traj:
                if np.linalg.norm(pos - b_point) < self.d_dynamic:
                    return True
        return False

    # # Generate a random node within the environment bounds
    def random_node(self):
        return np.array([np.random.uniform(low, high) for low, high in self.bounds])

    # # Function to find the nearest node in the tree to a given point
    def nearest_node(self, tree, point):
        positions = np.array([node.position for node in tree])
        tree_kd = KDTree(positions)
        _, idx = tree_kd.query(point)
        return tree[idx]

    # # Function to check if a path is collision-free
    def is_path_collision(self, start, end):
        direction = end - start
        distance = np.linalg.norm(direction)
        steps = int(distance / self.step_size)
        for step in range(steps):
            point = start + direction * (step / steps)
            if self.is_collision_obstacles(point) or self.is_collision_drones(point):
                return True
        # return is_collision_free(end)
        return False

    # # Function to find the optimal path to the goal using RRT*
    def get_traj(self):
        start_node = self.Node(self.start_pos)
        goal_node = self.Node(self.goal_pos)

        if self.is_collision_obstacles(self.goal_pos):
            print('init goal state collision')
            return False
            
        tree = [start_node]
        for _ in range(self.max_iter):
            rand_pos = self.random_node()
            nearest = self.nearest_node(tree, rand_pos)
            direction = rand_pos - nearest.position
            new_pos = nearest.position + self.step_size * direction / np.linalg.norm(direction)
            if not self.is_collision_obstacles(new_pos) or not self.is_collision_drones(new_pos):
                new_node = self.Node(new_pos, nearest)
                new_node.cost = nearest.cost + np.linalg.norm(direction)

                # Find neighbors within a radius and find the optimal parent
                neighbors = [node for node in tree if np.linalg.norm(node.position - new_pos) <= self.r_rrt]
                best_parent = nearest
                min_cost = new_node.cost
                for neighbor in neighbors:
                    potential_cost = neighbor.cost + np.linalg.norm(neighbor.position - new_pos)
                    if potential_cost < min_cost and not self.is_path_collision(neighbor.position, new_pos):
                        best_parent = neighbor
                        min_cost = potential_cost
                new_node.parent = best_parent
                new_node.cost = min_cost
                tree.append(new_node)

                # Rewire the tree with the new node
                for neighbor in neighbors:
                    if neighbor != best_parent:
                        potential_cost = new_node.cost + np.linalg.norm(neighbor.position - new_node.position)
                        if potential_cost < neighbor.cost and not self.is_path_collision(new_node.position, neighbor.position):
                            neighbor.parent = new_node
                            neighbor.cost = potential_cost

                # Check if the goal is reachable from the new node
                if np.linalg.norm(new_node.position - self.goal_pos) <= self.eps and not self.is_path_collision(new_node.position, self.goal_pos):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + np.linalg.norm(new_node.position - self.goal_pos)
                    tree.append(goal_node)
                    break

        # Extract the final path
        path = []
        node = goal_node
        while node is not None:
            path.append(node.position)
            node = node.parent
        return np.array(path[::-1])  # Reverse the path

class BiRRT_Star(RRT_star):
    def __init__(self, start_pos, goal_pos, static_obstacles, dynamic_obstacles, goal_bias=0.1, max_neighbor = 10, bounds=((-4, 4), (-4, 4), (0, 2))) -> None:
        super().__init__(start_pos, goal_pos, static_obstacles, dynamic_obstacles, bounds=bounds)
        self.goal_bias = goal_bias
        self.max_neighbor = max_neighbor
    
    def random_node(self):
        # Implement a goal bias in random sampling
        if np.random.rand() < self.goal_bias:
            return self.goal_pos
        return super().random_node()
        
    def extend_tree(self, tree, rand_pos):
        nearest = self.nearest_node(tree, rand_pos)
        direction = rand_pos - nearest.position
        new_pos = nearest.position + self.step_size * direction / (np.linalg.norm(direction) + 1e-5)
        if not self.is_collision_obstacles(new_pos) and not self.is_collision_drones(new_pos):
            new_node = self.Node(new_pos, nearest)
            new_node.cost = nearest.cost + np.linalg.norm(direction)

            # Find neighbors within a radius and find the optimal parent
            # neighbors = [node for node in tree if np.linalg.norm(node.position - new_pos) <= self.r_rrt]
            neighbors = sorted([node for node in tree if np.linalg.norm(node.position - new_pos) <= self.r_rrt],
                               key=lambda node: np.linalg.norm(node.position - new_pos))[:self.max_neighbor]
            
            best_parent = nearest
            min_cost = new_node.cost
            for neighbor in neighbors:
                potential_cost = neighbor.cost + np.linalg.norm(neighbor.position - new_pos)
                if potential_cost < min_cost and not self.is_path_collision(neighbor.position, new_pos):
                    best_parent = neighbor
                    min_cost = potential_cost
            new_node.parent = best_parent
            new_node.cost = min_cost
            tree.append(new_node)

            # Rewire the tree with the new node
            for neighbor in neighbors:
                if neighbor != best_parent:
                    potential_cost = new_node.cost + np.linalg.norm(neighbor.position - new_node.position)
                    if potential_cost < neighbor.cost and not self.is_path_collision(new_node.position, neighbor.position):
                        neighbor.parent = new_node
                        neighbor.cost = potential_cost

            return new_node
        return None

    def connect_trees(self, node_a, tree_b):
        nearest_b = self.nearest_node(tree_b, node_a.position)
        if not self.is_path_collision(node_a.position, nearest_b.position):
            return nearest_b
        return None

    def get_traj(self):
        start_node = self.Node(self.start_pos)
        goal_node = self.Node(self.goal_pos)
        tree_start = [start_node]
        tree_goal = [goal_node]

        for i in range(self.max_iter):
            rand_pos = self.random_node()
            if i % 2 == 0:
                # Grow forward tree
                new_node = self.extend_tree(tree_start, rand_pos)
                if new_node:
                    connected = self.connect_trees(new_node, tree_goal)
                    if connected:
                        # if goal_node != connected:
                        #     goal_node.parent = connected
                        # else:
                        #     goal_node.parent = copy.deepcopy(connected)
                        # connected.partent = new_node
                        start_tree_end_node = new_node
                        goal_tree_end_node = connected
                        break
            else:
                # Grow backward tree
                new_node = self.extend_tree(tree_goal, rand_pos)
                if new_node:
                    connected = self.connect_trees(new_node, tree_start)
                    if connected:
                        # start_node.parent = connected
                        # new_node.parent = connected
                        # connected.parent = new_node
                        start_tree_end_node = connected
                        goal_tree_end_node = new_node
                        break

        # Extract the final paths
        path_start = []
        node = start_tree_end_node
        while node is not None:
            path_start.append(node.position)
            node = node.parent

        path_goal = []
        node = goal_tree_end_node
        while node is not None:
            path_goal.append(node.position)
            node = node.parent

        path = path_start[::-1] + path_goal[1:]
        return np.array(path)

def smooth_trajectory(waypoints, times=None, num_samples=500):

    num_waypoints = waypoints.shape[0]
    # Check if `times` is provided; otherwise, generate equally spaced indices
    if times is None:
        times = np.linspace(0, num_waypoints - 1, num=num_waypoints)

    if len(times) != num_waypoints:
        raise ValueError("Length of `times` must match the number of waypoints.")

    # Create cubic splines for each coordinate
    x_spline = CubicSpline(times, waypoints[:, 0], bc_type='natural')
    y_spline = CubicSpline(times, waypoints[:, 1], bc_type='natural')
    z_spline = CubicSpline(times, waypoints[:, 2], bc_type='natural')

    # Generate equally spaced time samples for the smooth trajectory
    t_smooth = np.linspace(times[0], times[-1], num=num_samples)

    # Evaluate the interpolated trajectory at the smooth samples
    x_smooth = x_spline(t_smooth)
    y_smooth = y_spline(t_smooth)
    z_smooth = z_spline(t_smooth)

    # Combine the smooth x, y, z coordinates into a single array
    smoothed_traj = np.column_stack((x_smooth, y_smooth, z_smooth))

    return smoothed_traj
