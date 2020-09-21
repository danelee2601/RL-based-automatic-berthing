import numpy as np

def get_mean_action(action):
    mean_action = np.mean(action, keepdims=True, axis=0)
    stacked_mean_action = np.copy(mean_action)
    for i in range(action.shape[0] - 1):
        stacked_mean_action = np.vstack((stacked_mean_action, mean_action))
    return stacked_mean_action