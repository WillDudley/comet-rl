import comet_ml
import gymnasium as gym 

class CometLogger(gym.Wrapper):

    def __init__(self, env: gym.Env, experiment: comet_ml.Experiment):
        super().__init__(env)

        self.experiment = experiment 


        self._recorded_video = False
        if self.render_mode == 'rgb_array' or self.render_mode == 'rgb_array_list':
            self.env = gym.wrappers.RecordVideo(env, 'gym_videos')
            self._recorded_video = True

        

        self.episode_counter = 0
        self.step_counter = 0
        if isinstance(self.experiment, comet_ml.ExistingExperiment):
            exp_key = self.experiment.get_key()
            api_exp = comet_ml.APIExperiment(previous_experiment=exp_key)
            self.episode_counter = api_exp.get_metrics('episode_reward')[-1]['epoch']
            self.step_counter = api_exp.get_metrics_summary('episode_reward')['stepCurrent']
            
            

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self._cumulative_episode_reward = 0
        self._episode_length = 0
        self.episode_counter +=1 
        return obs, info
    
    def step(self, action):
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self._cumulative_episode_reward += reward 
        self._episode_length += 1
        self.step_counter +=1

        if terminated or truncated:
            
            self.experiment.log_metric('episode_reward', self._cumulative_episode_reward,  step= self.step_counter, epoch=self.episode_counter)
            self.experiment.log_metric('episode_length', self._episode_length, step= self.step_counter, epoch=self.episode_counter)

        return observation, reward, terminated, truncated, info
    
    def close(self):
        if self._recorded_video:
            self.experiment.log_asset_folder('gym_videos')
        return super().close()