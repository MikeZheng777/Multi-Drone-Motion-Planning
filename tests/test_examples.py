def test_pid():
    from gym_pybullet_drones.run.pid import run
    run(gui=False, plot=False, output_folder='tmp')

def test_pid_velocity():
    from gym_pybullet_drones.run.pid_velocity import run
    run(gui=False, plot=False, output_folder='tmp')

def test_downwash():
    from gym_pybullet_drones.run.downwash import run
    run(gui=False, plot=False, output_folder='tmp')

def test_learn():
    from gym_pybullet_drones.run.learn import run
    run(gui=False, plot=False, output_folder='tmp', local=False)
