import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    to make the custom env, I referred to "https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e"
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, Berthing_env):
        super(CustomEnv, self).__init__()

        self.Berthing_env = Berthing_env
        self.LBP = self.Berthing_env.L
        
        levels = {1: {'X': np.array([7, 12]), 'Y': np.array([2, 9])}}
        #levels = {1: {'X': np.array([1.6, 1.6]), 'Y': np.array([1.6, 1.6])}}
        # ===================================================
        self.specified_level = 1
        
        self.init_Xcoord_range = levels[self.specified_level]['X'] * self.LBP  # [m]  # np.array([7, 12])
        self.init_Ycoord_range = levels[self.specified_level]['Y'] * self.LBP  # [m]  # np.array([2, 9])
        self.heading_noise_range = np.array([-15, 15]) * (np.pi / 180)  # [rad.]
        # ===================================================
        self.norm_init_Xcoord_range = self.init_Xcoord_range / self.LBP
        self.norm_init_Ycoord_range = self.init_Ycoord_range / self.LBP

        # define action space
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)  # rudder_angle, n

        # define observation space
        n_obs = 7  # [norm_x, norm_y, init_u, init_v, init_r, init_rudder_angle]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)  # [coord_x, coord_y, goal_coord_x, goal_coord_y, abs(distance_left)]

        # termination condition
        self.heading_angle_termination = np.array([240, 300]) * (np.pi / 180)  # [rad.]  # he relative heading angle to the berthing facility should be smaller than 30deg
        self.u_termination = 0.2  # [m/s]
        self.pos_termination = np.array([1.5 * self.LBP, 1.5 * self.LBP])
        self.norm_pos_termination = self.pos_termination / self.LBP
        self.termination_tolerance = 0.4 * self.LBP
        self.norm_termination_tolerance = self.termination_tolerance / self.LBP

        # count
        self.count = 0
        self.hit_zone_count = 0

        # init. variables
        self.coord_x = 0.0  # [m]
        self.coord_y = 0.0  # [m]
        self.heading_angle = 0.0  # [rad.]
        self.prev_norm_distance_to_terminal_pos = 0.
        
    def sample_in_norm_dist(self, x, mu, var):
        norm_dist = 1/np.sqrt(2*3.14*var) * np.exp(-(x-mu)**2/(2*var))
        return norm_dist
    
    def scale_up(self, y, max_val):
        return y * (1/max_val)

    def get_reward(self, norm_x, norm_y, heading_angle, u, n, rudder_angle, heading_limit_deg=(180, 300), cutoff_u_to_prevent_excessive_negative_u=-0.5):
        heading_angle_deg = heading_angle * (180/np.pi)
        rudder_angle_deg = rudder_angle * (180/np.pi)
        
        bonus_reward = 0
        terminal_condition_check = {"heading": False, "u": False}

        norm_distance_to_terminal_posx = np.abs(norm_x - self.norm_pos_termination[0])
        norm_distance_to_terminal_posy = np.abs(norm_y - self.norm_pos_termination[1])
        norm_distance_to_terminal_pos = np.sqrt((norm_x - self.norm_pos_termination[0])**2 + (norm_y - self.norm_pos_termination[1])**2)
        
        if self.count == 0:
            self.prev_norm_distance_to_terminal_pos = norm_distance_to_terminal_pos  # initialize prev_norm_distance_to_terminal_pos
        
        dist_rw = (self.prev_norm_distance_to_terminal_pos - norm_distance_to_terminal_pos) * 300
              
        # update 'self.prev_norm_distance_to_terminal_pos'
        self.prev_norm_distance_to_terminal_pos = norm_distance_to_terminal_pos
        
        # if a ship is inside of the terminal circle
        if norm_distance_to_terminal_pos <= self.norm_termination_tolerance:
            print("* got into the goal point circle.")
            bonus_reward += 10
            
            if self.heading_angle_termination[0] <= heading_angle <= self.heading_angle_termination[1]:
                bonus_reward += 2
                
            # early stop when the ship is in the zone
            if self.hit_zone_count >= 60:
                terminal_condition_check = {"heading": True, "u": True}
            self.hit_zone_count += 1
        
        bonus_reward -= np.abs(rudder_angle_deg) / 500
        
        if u < 0.0:
            bonus_reward += u / 10
        
        if (heading_angle_deg < heading_limit_deg[0]) or (heading_angle_deg > heading_limit_deg[1]):
            terminal_condition_check = {"heading": True, "u": True}  # equal to 'done=True'
    
        reward = dist_rw + bonus_reward
        reward /= 10  # reward normalization
        return reward, terminal_condition_check, norm_distance_to_terminal_pos

    def step(self, action):
        # get rudder_angle, n
        rudder_angle = action[0]
        n = action[1]

        # step
        self.Berthing_env.step(rudder_angle, n, step_size=1.0, verbose=False)  # step_size: 1.0 [s]
        
        x, y = self.Berthing_env.coordinate_x, self.Berthing_env.coordinate_y
        norm_x, norm_y = x / self.LBP, y / self.LBP
        u, v, r = self.Berthing_env.u, self.Berthing_env.v, self.Berthing_env.r
        heading_angle = self.Berthing_env.heading_angle
        
        norm_dist = np.sqrt((norm_x-self.norm_pos_termination[0])**2 + (norm_y-self.norm_pos_termination[1])**2)

        # update coord_x, coord_y, heading_angle
        self.coord_x = x
        self.coord_y = y
        self.heading_angle = heading_angle

        # get next_obs
        next_obs = np.array([norm_x, norm_y, norm_dist, u, v, r, heading_angle])

        # reward function =====================================================
        reward, terminal_condition_check, norm_distance_to_terminal_pos = self.get_reward(norm_x, norm_y, heading_angle, u, n, rudder_angle)

        # define 'done' =====================================================
        if False not in list(terminal_condition_check.values()):
            done = True
        else:
            done = False

        # 너무 멀리나가면 종료
        if (norm_x < self.norm_pos_termination[0]-self.norm_termination_tolerance*2) or (norm_y < self.norm_pos_termination[1]-self.norm_termination_tolerance*2):
            done = True

        # 너무 많은 스텝지나면 종료
        if self.count >= self.Berthing_env.simulationDuration:
            done = True
            
        # info =====================================================
        # info 의 초기값은 {} 으로 설정해야될것
        info = {}

        self.count += 1

        return next_obs, reward, done, info  # (obs, reward, done, step)

    def get_init_heading_angle(self, init_coordinate_x, init_coordinate_y, noise_range):
        x = init_coordinate_x - self.pos_termination[0]
        y = init_coordinate_y - self.pos_termination[1]
        theta = np.arctan( y/x )  # [rad.]
        appropriate_heading_angle = np.pi + (np.pi/2 - theta)  # [rad]

        noise = np.random.uniform(low=noise_range[0], high=noise_range[1])
        appropriate_heading_angle_with_noise = appropriate_heading_angle + noise
        return appropriate_heading_angle_with_noise

    def reset(self):
        # define initial conditions
        self.init_coordinate_x = np.random.uniform(low=self.init_Xcoord_range[0], high=self.init_Xcoord_range[1])
        self.init_coordinate_y = np.random.uniform(low=self.init_Ycoord_range[0], high=self.init_Ycoord_range[1])
        init_u = 1.
        init_v, init_r = 0., 0.

        init_heading_angle = self.get_init_heading_angle(self.init_coordinate_x, self.init_coordinate_y, self.heading_noise_range)  # [rad.]

        init_rudder_angle, init_n = 0., 0.5

        # update coord_x, coord_y, heading_angle
        self.coord_x = self.init_coordinate_x
        self.coord_y = self.init_coordinate_y
        self.heading_angle = init_heading_angle

        # reset
        self.Berthing_env.reset(self.init_coordinate_x, self.init_coordinate_y, init_heading_angle, init_u=init_u, init_n=init_n, init_v=init_v, init_r=init_r, init_rudder_angle=init_rudder_angle)

        # get the init_obs
        norm_x = self.init_coordinate_x / self.LBP  # eta
        norm_y = self.init_coordinate_y / self.LBP  # ksi
        
        norm_dist = np.sqrt((norm_x-self.norm_pos_termination[0])**2 + (norm_y-self.norm_pos_termination[1])**2)
        
        init_obs = np.array([norm_x, norm_y, norm_dist, init_u, init_v, init_r, init_rudder_angle])

        # update self.count
        self.count = 0
        self.hit_zone_count = 0

        return init_obs

    def deterministic_reset(self, init_coords, init_heading_angle):
        # define initial conditions
        self.init_coordinate_x = init_coords[0]
        self.init_coordinate_y = init_coords[1]
        init_u = 1.
        init_v, init_r = 0., 0.
        init_heading_angle = init_heading_angle  # [rad.]
        init_rudder_angle, init_n = 0., 0.5

        # update coord_x, coord_y, heading_angle
        self.coord_x = self.init_coordinate_x
        self.coord_y = self.init_coordinate_y
        self.heading_angle = init_heading_angle

        # reset
        self.Berthing_env.reset(self.init_coordinate_x, self.init_coordinate_y, init_heading_angle, init_u=init_u, init_n=init_n,
                                init_v=init_v, init_r=init_r, init_rudder_angle=init_rudder_angle)
        # get the init_obs
        norm_x = self.init_coordinate_x / self.LBP  # eta
        norm_y = self.init_coordinate_y / self.LBP  # ksi
        
        norm_dist = np.sqrt((norm_x-self.norm_pos_termination[0])**2 + (norm_y-self.norm_pos_termination[1])**2)

        init_obs = np.array([norm_x, norm_y, norm_dist, init_u, init_v, init_r, init_rudder_angle])

        # update self.count
        self.count = 0
        self.hit_zone_count = 0

        return init_obs

    def render(self, mode='human', close=False):
        ...