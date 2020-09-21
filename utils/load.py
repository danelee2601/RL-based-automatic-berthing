import os

from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines import PPO2

from .declare_berthing_env import decalre_Berthing_env
from .custom_gym_env import CustomEnv

def load_trained_env_n_model(loading_section_num=12):
    # params
    save_dirname_model = "models"
    save_dirname_env = "envs"
    
    # load env and model
    section_num = loading_section_num

    save_dirname_model2 = os.path.join('.', save_dirname_model, f"section{section_num}")
    save_dirname_env2 = os.path.join('.', save_dirname_env, f"section{section_num}.pkl")

    # set env
    Berthing_env = decalre_Berthing_env()
    n_cpu = 1  
    loaded_env = SubprocVecEnv([lambda: CustomEnv(Berthing_env) for i in range(n_cpu)])  # n_cpu=1 during test
    loaded_env = VecNormalize(loaded_env, norm_obs=False, norm_reward=False)  # to load VecNormalize object, norm_obs and norm_reward must be False
    loaded_env = loaded_env.load(save_dirname_env2, loaded_env)

    # set model1
    loaded_model = PPO2.load(save_dirname_model2, env=loaded_env)
    return loaded_env, loaded_model