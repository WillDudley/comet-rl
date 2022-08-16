from comet_ml import Experiment
import gym 

class CometLogger(gym.Wrapper):

    def __init__(self, env: gym.Env , experiment: Experiment):
        super().__init__(env)
        self.env = env
        self.experiment = experiment  

        self.episode_counter = 0
        self._cumulative_episode_reward = 0
        self._episode_length = 0
        self.step_counter = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        
        self._cumulative_episode_reward += reward 
        self._episode_length += 1
        self.step_counter +=1

        if done:
            self.episode_counter +=1 
            self.experiment.log_epoch_end(self.episode_counter)
            self.experiment.log_metric('episode_reward', self._cumulative_episode_reward, epoch=self.episode_counter)
            self.experiment.log_metric('episode_length', self._episode_length, epoch=self.episode_counter)
            self._cumulative_episode_reward = 0
            self._episode_length = 1

        return next_state, reward, done, info