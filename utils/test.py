import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .mean_action import get_mean_action
from .drawer.berthing_trajectory_drawer import BerthingTrajectoryDrawer


def test(norm_init_coords, init_heading_angle, env, model, deterministic=False, use_recurrent_model=False):
    env.reset()

    LBP = env.get_attr("Berthing_env")[0].L
    
    init_coords = np.array(norm_init_coords) * LBP
    
    env.reset()
    obs = env.env_method("deterministic_reset", init_coords, init_heading_angle)
    action_option = "mean"  # single | mean  ('mean' performs better)

    n_steps = env.get_attr("Berthing_env")[0].simulationDuration
    state_hist, reward_hist = [], []
    #state_hist.append([env.get_attr("coord_x")[0], env.get_attr("coord_y")[0], env.get_attr("heading_angle")[0]])
    state = None  # initial states for LSTM
    done = [False for _ in range(env.num_envs)]  # When using VecEnv, done is a vector
    for i in range(n_steps):

        if action_option == "single":
            action = model.predict(obs, deterministic=deterministic)[0]
            obs, reward, done, _ = env.step(action)
        elif action_option == "mean":
            if use_recurrent_model:
                action, state = model.predict(obs, state=state, mask=done)
            else:
                action = model.predict(obs, deterministic=deterministic)[0]
            
            mean_action = get_mean_action(action)
            obs, reward, done, _ = env.step(mean_action)
            
        if True in np.array([done]).flatten():
            break
        
        rudder_angle, n = env.get_attr("Berthing_env")[0].rudder_angle, env.get_attr("Berthing_env")[0].n
        u, v, r = env.get_attr("Berthing_env")[0].u, env.get_attr("Berthing_env")[0].v, env.get_attr("Berthing_env")[0].r
        current_state = [env.get_attr("coord_x")[0], env.get_attr("coord_y")[0], env.get_attr("heading_angle")[0], n, rudder_angle, u, v, r]
        state_hist.append(current_state)
        reward_hist.append(reward)
        
    state_hist = np.array(state_hist)
    
    # plot
    pos_termination = env.get_attr("pos_termination")[0]
    terminal_circle_radius = env.get_attr("termination_tolerance")[0]
    drawer = BerthingTrajectoryDrawer(LBP, pos_termination, terminal_circle_radius, plot_lim=(-1, 20))

    x_hist, y_hist, heading_angle_hist, n_hist, rudder_angle_hist, u_hist, v_hist, r_hist = [state_hist[:, i] for i in range(len(current_state))]  # n [rps], rudder_angle [rad.]
    state_hist = pd.DataFrame(state_hist, columns=["x_hist", "y_hist", "heading_angle_hist", "n_hist", "rudder_angle_hist", "u_hist", "v_hist", "r_hist"])  # convert its type to pd.DataFrame
    
    # plot: ship trajectory
    drawing_period = 50
    idx = 0
    for x, y, heading_angle in zip(x_hist, y_hist, heading_angle_hist):
        if idx % drawing_period == 0:
            drawer.draw_a_ship_at_one_timestep(x, y, heading_angle)
        idx += 1
    drawer.show_plot()
    
    # plot: 'n','rudder-angle', 'u', 'v', 'r'
    plt.figure(figsize=(15, 3*5))
    t_hist = np.arange(0, len(n_hist), 1)  # RK4's step-size is 1 [s]
    
    plt.subplot(5, 1, 1)
    plt.plot(t_hist, n_hist)
    plt.ylabel("n [rps]")
    plt.grid()
    
    plt.subplot(5, 1, 2)
    plt.plot(t_hist, rudder_angle_hist * (180/np.pi))
    plt.ylabel("rudder angle [deg]")
    plt.grid()
    
    plt.subplot(5, 1, 3)
    plt.plot(t_hist, u_hist)
    plt.ylabel("u [m/s]")
    plt.grid()

    plt.subplot(5, 1, 4)
    plt.plot(t_hist, reward_hist)
    plt.ylabel("reward")
    plt.xlabel("time [s]")
    plt.grid()
    
    return state_hist
    
    
def set_init_heading_angle(norm_init_coords, norm_pos_termination=(1.5, 1.5), extra_angle_deg=0):
    x = norm_init_coords[0] - norm_pos_termination[0]
    y = norm_init_coords[1] - norm_pos_termination[1]
    theta = np.arctan( y/x )  # [rad]
    init_heading_angle = np.pi + (np.pi/2 - theta)  # [rad]

    # put extra to heading ang
    extra = extra_angle_deg * (np.pi/180)  # [rad]
    init_heading_angle += extra
    return init_heading_angle
