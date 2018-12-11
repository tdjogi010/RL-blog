import tensorflow as tf
import gym
import numpy as np

from network import get_mlp_network
from wrappedEnv import WrappedEnv

#common components
fc = tf.layers.dense #fully connected network
relu = tf.nn.relu
tanh = tf.tanh
conv2d = tf.layers.conv2d
xavier_init = tf.contrib.layers.xavier_initializer # same as tf.contrib.layers.xavier_initializer_conv2d
sess = tf.Session()

#contants
clip = 0.1
entropy_coefficient = 0.01
vf_coefficient = 0.5
lr = 2.5e-4
epsilon = 1e-5
gamma = 0.99
lambdA = 0.95

T = 256#1e6 # total number of timesteps ie total number of actions taken in env by agent before the training is completed

# Times 2 to have our agent run (as good as) twice (not really the same thing as having two different agents) to collect the samples
steps = batch_size = 128 #128*2 # number of steps taken by one agent to make a batch (to prepare for updates). Here, it is also = batch size, since using only one agent
number_of_mini_batch = 4

assert batch_size % number_of_mini_batch == 0 # no uneven mini_batch_size
mini_batch_size = batch_size // number_of_mini_batch

number_of_epochs = 4 # number of epochs (after creating a batch)

env_id = 'Breakout-v0'
save_path = './tmp1/agent'
load_path = './tmp/agent-256'

class Agent:
    def __init__(self, env):
        # define placeholders for feed forward/stepping
        OBS = tf.placeholder(tf.float32, (None,) + env.observation_space.shape, 'obs')

        # network
        network_out = get_mlp_network(OBS)# get_cnn_network(OBS)
        # policy outputing probability distribution of actions
        pi = fc(network_out, env.action_space.n, activation=tf.nn.softmax, kernel_initializer=xavier_init(), name='pi')
        sample_action = tf.multinomial(tf.log(pi), 1, name='sample_action')
        sample_action = sample_action[:,0]

        # get pi[:,sample_action]
        one_hot_mask = tf.one_hot(sample_action, pi.shape[1], on_value = True, off_value = False, dtype = tf.bool, name='one_hot')
        pi_action = tf.boolean_mask(pi, one_hot_mask, name='boolean_mask') # causes a "Converting sparse IndexedSlices to a dense Tensor of unknown shape" warning
        
        # state value function 
        vf = fc(network_out, 1, activation=None, kernel_initializer=xavier_init())
        
        ## Calculate losses ##
        # placeholder for old policy
        PI_ACTION_OLD = tf.placeholder(tf.float32, (None,), name='pi_action_old')
        VF_OLD = tf.placeholder(tf.float32, (None,), name='vf_old')
        RETURNS = tf.placeholder(tf.float32, (None,), name='returns')
        ADV = tf.placeholder(tf.float32, (None,), name='advs')
        ACTION = tf.placeholder(tf.int32, (None,), 'action') # need to choose the same action from the new policy

        # get pi[:,ACTION]
        one_hot_mask_train = tf.one_hot(ACTION, pi.shape[1], on_value = True, off_value = False, dtype = tf.bool, name='one_hot_train')
        pi_action_train = tf.boolean_mask(pi, one_hot_mask_train, name='boolean_mask_train') # causes a "Converting sparse IndexedSlices to a dense Tensor of unknown shape" warning
        
        ratio =  pi_action_train / PI_ACTION_OLD 
        
        # Essentially we want to maximise rewards with objective fucntion J.
        # Since we are constructing loss, L and we are using reward as metrics for(or in) objective function J to maximise it,
        # so we minimize loss, L = -J so to maximise J
        pg_loss_unclipped = - ADV * ratio
        pg_loss_clipped = - ADV * tf.clip_by_value(ratio, 1.0 - clip, 1.0 + clip)
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))
        
        # clipping the difference i.e clipping around zero
        vf_clipped = VF_OLD + tf.clip_by_value(vf - VF_OLD, - clip, clip)
        # vf loss is mean squared loss
        vf_loss_unclipped = tf.square(vf - RETURNS)
        vf_loss_clipped = tf.square(vf_clipped - RETURNS)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_loss_unclipped, vf_loss_clipped))

        # subtract entropy from total_loss so that policy does not become deterministics or to encourage exploration as well
        # tf does not function for entropy but for cross_entropy ... hmmm
        entropy = - tf.reduce_sum(pi * tf.log(pi))
        
        # Total loss
        loss = pg_loss - entropy * entropy_coefficient + vf_loss * vf_coefficient
        train_op = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon).minimize(loss)

        # agent taking a step gives u (ie sampling an) action along with value estimate V(s) and pi(a,s) given the observation 
        def step(obs):
            return sess.run([sample_action, pi_action, vf], feed_dict={OBS: obs})
        
        # Strain the agent. gives u losses (pg_loss, vf_loss, policy_entropy)
        def train(obs, actions, returns, pi_action_old, vf_old,  advs) :
            advs = (advs - advs.mean()) / (advs.std())
            return sess.run([train_op, loss, pg_loss, vf_loss, entropy],{OBS: obs, ACTION: actions, RETURNS: returns,
                PI_ACTION_OLD: pi_action_old, VF_OLD: vf_old, ADV: advs})[1:]
            
        self.step = step
        self.train = train
        self.saver = tf.train.Saver()
    
    def save(self, timestep):
        self.saver.save(sess, save_path, global_step=timestep)
    
    def restore(self):
        self.saver.restore(sess, load_path)

env = gym.make(env_id)
env = WrappedEnv(env)
agent = Agent(env)
if load_path == None or load_path == '':
    sess.run(tf.global_variables_initializer())
else:
    agent.restore()
    print("agent restored!")

obs = env.reset()
# obs1,*_ = env.step(3)
# print(agent.step([obs,obs1,obs1]))
timesteps = 0
while True:
    if timesteps >= T:
        break

    mb_obs, mb_rewards, mb_actions, mb_pi_actions, mb_vfs, mb_dones = [], [], [], [], [], []
    done = False
    for _ in range(steps):
        # sample action from agent and get pi_action and vf of given state
        action, pi_action, vf = agent.step([obs])
        mb_obs.append(obs)
        mb_actions.append(action)
        mb_pi_actions.append(pi_action)
        mb_vfs.append(vf)
        mb_dones.append(done)

        # step the env and get reward and next state
        obs, reward, done, info = env.step(action) # NOTE: done is for(or appended with) next state along with next action ie next state is terminal or not
        mb_rewards.append(reward) # current reward for taking sampled action

        if done:
            obs = env.reset()
        timesteps += 1

    
    mb_obs = np.asarray(mb_obs, dtype=obs.dtype)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions).flatten()
    mb_vfs = np.asarray(mb_vfs, dtype=np.float32).flatten()
    mb_pi_actions = np.asarray(mb_pi_actions, dtype=np.float32).flatten()
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    
    # need last_vf for A(s,a) = R + yV(s+1) - V(s) for s = steps-1 (here, s as state with timestep)
    _,_, last_vf = agent.step([obs])
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)

    # calculate R(t) = rew + gamma*rew + gamma^2*rew + ... + gamma^(nsteps-t)*vf(nsteps) 
    for t in reversed(range(steps)):
        if t == steps - 1:
            # ie we dont have t + 1 in the array
            mb_returns[t] = mb_rewards[t] + gamma * last_vf * (1 - done)
        else:
            mb_returns[t] = mb_rewards[t] + gamma * mb_returns[t+1] * (1 - mb_dones[t])
    
    mb_advs = mb_returns - mb_vfs

    mb_lossvals = []
    inds = np.arange(batch_size)
    print("Updating after {}".format(timesteps))
    for _ in range(number_of_epochs):
        # Randomize the indexes
        np.random.shuffle(inds)
        # 0 to batch_size with mini_batch_size number of data
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mbinds = inds[start:end]
            slices = (arr[mbinds] for arr in (mb_obs, mb_actions, mb_returns, mb_pi_actions, mb_vfs, mb_advs))
            mb_lossvals.append(agent.train(*slices))
        
        print("Updated")
        print(mb_lossvals)
    
    agent.save(timesteps)