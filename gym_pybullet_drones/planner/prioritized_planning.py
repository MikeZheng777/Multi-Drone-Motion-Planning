from gym_pybullet_drones.planner.rrt_star import *


class Prioritized_Planing():

    def __init__(self, env, order, inital_xyz, final_xyz) -> None:
        self.env = env
        self.order = order
        self.inital_xyz = inital_xyz
        self.final_xyz = final_xyz

        
    def get_all_traj(self):

        dynamic_obstacles = []
        all_trajs = []
        for o in self.order:
            planner = BiRRT_Star(start_pos = self.inital_xyz[o], goal_pos=self.final_xyz[o], static_obstacles=self.env.static_obstacles, dynamic_obstacles=dynamic_obstacles)
            traj = planner.get_traj()
            dynamic_obstacles.append(smooth_trajectory(traj,num_samples=200))
            all_trajs.append(traj)

        return all_trajs


# if __name__ == '__main__':

