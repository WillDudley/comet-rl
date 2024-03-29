from comet_logging import CometLogger
from comet_ml import Experiment
import gymnasium as gym
import os


experiment = Experiment(
    api_key=os.environ['COMET_API_KEY'],
    project_name="gym-test",
    workspace="willdudley",
)


env = gym.make('Acrobot-v1', render_mode="rgb_array")
env = CometLogger(env, experiment)

for x in range(20):

    obs, info = env.reset()
    truncated = False
    terminated = False 
    while not (truncated or terminated):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

env.close()