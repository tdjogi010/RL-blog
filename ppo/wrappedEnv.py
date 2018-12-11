from gym.spaces.box import Box
import numpy as np
import cv2
from collections import deque

class WrappedEnv:
    def __init__(self, env, stack_frames_size):
        self.env = env
        self.action_space = self.env.action_space
        # need to manually preprocess # need a better way
        self.observation_space = Box(0,255,(83, 75, stack_frames_size)) # (low, high, (height, width, frames))
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self.stack_frame = deque([], maxlen=stack_frames_size) #np.zeros((83, 75, k))
    
    def seed(self, arg):
        return self.env.seed(arg)
    
    def step(self, a):
        obs, *_ = self.env.step(a)
        # preprocess: crop, downsample and convert to greyscale
        obs = obs[30:-15:2,5:-5:2,:]
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs[:,:,None]
        
        self.stack_frame.append(obs) # here, (frame, height, width)
        obss = np.asarray(self.stack_frame)
        obss = np.concatenate(obss, axis=-1) # merge on last axis ie (height, width, frames)
        return obss, _ # return a list
    
    def reset(self):
        # preprocess: crop, downsample and convert to greyscale
        obs = self.env.reset()[30:-15:2,5:-5:2,:]
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs[:,:,None]
        
        # append to stack_frame
        for i in range(self.stack_frame.maxlen):
            self.stack_frame.append(obs) # here, (frame, height, width)
        obss = np.asarray(self.stack_frame)
        obss = np.concatenate(obss, axis=-1) # merge on last axis ie (height, width, frames)
        return obss
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()