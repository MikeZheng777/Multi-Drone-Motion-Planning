# Multi-Drone Motion Planning using Bi-RRT* and Prioritized Planning

## Installation

```sh
git clone https://github.com/MikeZheng777/Multi-Drone-Motion-Planning.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . 

```

### Run Planning
```sh
python3 gym_pybullet_drones/run/plan_and_control_single.py # multi drone planning
python3 gym_pybullet_drones/run/plan_and_control_multi.py # multi drone planning without Prioritized Planning
python3 gym_pybullet_drones/run/plan_and_control_multi_pp.py # multi drone planning with Prioritized Planning

```

```sh
python3 gym_pybullet_drones/run/plan_and_control_multi_pp.py --debug True --reload True # plot previous saved planned path (saved as .npy without reload flag) in simulation.
```

### Video Demo
#### Single-drone planning
![Path founded offline by Bi-RRT*](gif_videos/single.gif)
#### Multi-drone planning
![Path founded offline by Bi-RRT*, without Prioritized Planning, drone to drone collision happens](gif_videos/bad_multi.gif)
![Path founded offline by Bi-RRT*, with Prioritized Planning](gif_videos/good_multi.gif)


