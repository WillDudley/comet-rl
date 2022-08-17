# comet-rl
Auto Logging Episode Rewards and Episode Lengths for Gym environments during RL training on Comet

## Install

Step 1: Clone this repo (ideally to replaced with pip)

```console
git clone https://github.com/sherpan/comet-rl.git && cd comet-rl
```

Step 2: Install the comet_rl package
```console
pip install . 
```
## Minumum Exmaple Usage

Step 1: Install a RL Agent/Algorithm Library. 

```console
pip install stablebaselines3
```

Step 2: Sign-up for [Comet](comet.com) and retrieve your API Key (free for individuals)

Step 3: Copy and paste code to your editor and populate the api_key, project_name, workspace

```python
from comet_ml import Experiment
from comet_rl import CometLogger
from stable_baselines3 import A2C
import gym

# Create an experiment with your api key
experiment = Experiment(
    api_key="YOUR_API_KEY",
    project_name="YOUR_PROJECT_NAME",
    workspace="YOUR_WORK_SPACE",
)

env = gym.make('Acrobot-v1')

env = CometLogger(env, experiment)

model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=100000)
```

Step 4: Click on the url given by Comet and view the experiment metric. View the episode_reward and episode_length metrics graphs in the panels/chart panes. Make sure to select 'epochs' as the x-axis for your graphs

## Logging Hyper-Parameters Example 
RL training is sensitive to hyper-parameters. It is imperative to save hyper-parameters for each experiment to gain insights on how the parameters affect are affecting the reward metric. Saving the agent after each experiment run is also a good practice. Users can come to a particalur run, access the "Assets & Artifacts" tab and download a model from a previous run at any point in time. 

Copy and paste the following code and run it! Then change some of the hyperparemeters to see if they made the agent learn better, worse, slower or faster!
