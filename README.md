# Cognitive and ML Tasks
This repo includes several custom tasks/environments built on [openai-gym](https://gym.openai.com/) for training RL agents. 

# Install
```
$ git clone https://github.com/CgnRLAgent/cog_ml_tasks.git
$ cd cog_ml_tasks
$ pip install -e .
```
**Requirements:** see the [setup.py](https://github.com/CgnRLAgent/cog_ml_tasks/blob/master/setup.py).

# Envs/Tasks Descriptions
## AX_12
The AX_12 task consists in the presentation to the subject of six possible stimuli/cues '1' - '2', 'A' - 'B', 'X' - 'Y'.

The tester has 2 possible responses which depend on the temporal order of previous and current stimuli:
he has to answer 'R' when
* the last stored digit is '1' AND the previous stimulus is 'A' AND the current one is 'X',
* the last stored digit is '2' AND the previous stimulus is 'B' AND the current one is 'Y';
in any other case , reply 'L'.

**actions:**  'L', 'R'

**predefined rewards:** output correctly(1.0), not correct(-1.0)

e.g.

Input: 1AXBZ

Target: LLRLL

Input: 2CXCYBY

Target: LLLLLLR

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('AX_12-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# custom params
env = gym.make('AX_12-v0', min_size=3, prob_r=0.5)
```

## AX_CPT
The AX_CPT task consists in the presentation to the subject of four possible stimuli/cues: two context cues 'A' - 'B' and 2 target cues 'X' - 'Y'.

The tester has 2 possible responses which depend on the temporal order of previous and current stimuli: 
This task must start with context cues. Context cues and target cues take turns to appear. 
he has to answer 'R' when
* the current stimulus is 'X' AND the previous stimulus is 'A' ,
in any other case , reply 'L'.

**actions:**  'L', 'R'

**predefined rewards:** output correctly(1.0), not correct(-1.0)

e.g.

Input: AXBY

Target: LRLL

Input: XXYAX

Target: LLLLR

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('AX_CPT-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# custom params
env = gym.make('AX_CPT-v0', size=500)
```