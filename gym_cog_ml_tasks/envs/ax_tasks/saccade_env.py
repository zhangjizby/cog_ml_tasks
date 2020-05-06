## SACCADE-ANTISACCADE TASK 
## The Saccade-Antisaccade task consists in a sequence of multi-step trials where the final goal is to direct the eye movement according to previous cues projected on a screen.
## The cues are essentialy of two types:
##	- fixation mark : squaree at the center of the screen; if it is black it is a pro-saccade trial (P), if it is white it is an anti-saccade one (A);
##	- location cue: circle at the side of the screen; it can be either at the left side (L) or at the right side (R) of the screen
## The test-taker has 3 possible activities: F=front,L=left, R=right.
## In case of a pro-saccade trial, the eye movement has to be in the same direction as the location cue (PL or PR); otherwise, it has to be in the opposite direction (AL or AR).
## Each trial is composed by different phases:
##	- START: empty screen
##	- FIX: fixation mark appears; the test-taker has to fix it for two consecutive timesteps to have a first reward r_f (F selected twice)
##	- CUE: location cue appears together with fixation mark
##	- DELAY: location cue disappears for two timesteps to test the memory delay
##      - GO: fixation mark disappears as well (empty screen) and the subject has to solve the task (it has up to 8 timesteps to answer L or R)


from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys

class Saccade_ENV(Env):
    FIX_MARK = ['P','A']
    LOC_MARK = ['L','R']
    ACTIONS = ['Front', 'Left', 'Right']
    POS = ['Start', 'Fix', 'Cue', 'Delay', 'Go']

    def __init__(self, go_reward=10):

        # go reward: last reward of the episode, default 10
        self.go_reward = go_reward
        # observation: Empty, P, PR, PL, A, AL, AR
        self.observation_space = Discrete(7)
        # action: F, L, R
        self.action_space = Discrete(len(self.ACTIONS))
        # state of an episode
        self.pos = None
        self.time = 0
        self.screen = None
        self.total_reward = None
        self.target = None
        self.input_data = None
        self.last_action = None
        self.last_reward = None
        self.output_act = None

        self.np_random =None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _generate_input_data(self):
        input_data = np.random.choice(self.FIX_MARK) + np.random.choice(self.LOC_MARK)
        return input_data

    def _generate_target(self, pos=None):
        if pos is None:
            pos = self.POS[0]
        if pos == self.POS[0]:
            return None
        if pos == self.POS[1]:
            fix_target = self.ACTIONS[0]
            return fix_target
        if pos == self.POS[2]:
            cue_target = self.ACTIONS[0]
            return cue_target
        if pos == self.POS[3]:
            delay_target = self.ACTIONS[0]
            return delay_target
        if pos == self.POS[4]:
            if self.input_data[1] == self.LOC_MARK[0]:
                go_target = self.ACTIONS[1]
            if self.input_data[1] == self.LOC_MARK[1]:
                go_target = self.ACTIONS[2]
            return go_target

    def _get_obs(self):
        if self.pos is None:
            self.pos = self.POS[0]
        if self.pos == self.POS[0]:
            self.screen = None
        if self.pos == self.POS[1]:
            self.screen = self.input_data[0]
        if self.pos == self.POS[2]:
            self.screen = self.input_data
        if self.pos == self.POS[3]:
            self.screen == self.input_data[0]
        if self.pos == self.POS[4]:
            self.screen = None
        return self.screen
    
        
    def reset(self):
        self.time = 0
        self.pos = self.POS[0]
        self.input_data = self._generate_input_data()
        self.target = None
        self.total_reward = 0
        self.last_action = None
        self.last_reward = None
        self.screen = self._get_obs()
        self.output_act = ''
        return self.screen

    
    def step(self, action):
        assert self.action_space.contains(action)
        assert 0 <= self.time <= 7
        self.last_action = action
        reward = 0
        done = False
        output = self.ACTIONS[action]
        # get pos according to time
        if self.time == 0: pos = self.POS[0]
        if 1<= self.time <=2: pos = self.POS[1]
        if self.time == 3: pos = self.POS[2]
        if 4<= self.time <=5: pos = self.POS[3]
        if self.time == 6: pos = self.POS[4]
        self.pos = pos
        # get target according to pos
        self.target = self._generate_target(pos)
        self.time += 1
        # get reaward according to position, target and output
        if pos == self.POS[1]:
            if output == self.target: 
                reward = 1 
        if pos == self.POS[2]:
            if output != self.target: reward = -1
        if pos == self.POS[3]:
            if output != self.target: reward = -1
        if pos == self.POS[4]:
            if output == self.target:
                reward = self.go_reward
            done = True

        self.last_action = output
        self.output_act += ''+output
        self.last_reward = reward
        self.total_reward += reward
        obs = self._get_obs()
        self.screen = obs
            
        return obs, reward, done, {self.target}

    def render(self, mode='human'):
        outfile = sys.stdout
        screen = self.screen
        target = self.target
        pos = self.pos
        current_act = self.last_action
        output = self.output_act
        last_reward = self.last_reward
        total_reward = self.total_reward

        if screen == None: screen = 'None'
        if target == None: target = 'None'
        if pos == None: pos = self.POS[0]
        if current_act == None: current_act = 'None'
        if output == None: output = 'None'
        if last_reward == None: last_reward = 0
        if total_reward == None: total_reward = 0

        color = 'green' if target == current_act else 'red'
        output += colorize(current_act, color, highlight=True)

        outfile.write("="*20 + "\n")
        outfile.write("Time: %d\n" %self.time )
        outfile.write("Position: " + pos + "\n")
        outfile.write("Screen: " + screen +"\n")
        outfile.write("Target: " + target +"\n")
        outfile.write("Current Action: " + current_act + "\n")
        outfile.write("Cumulative Actions: " + output +"\n")
        outfile.write("-"*20 + "\n")
        outfile.write("Current reward: %d\n" %last_reward)
        outfile.write("Cumulative reward: %d\n" %total_reward)
        outfile.write("\n")
        return
        







