# Student version

# Revision History
#
# 10/09/19    Tim Liu    changed env to BitFlipEnv
# 10/09/19    Tim Liu    moved running in environment to solve_enviornment
# 10/09/19    Tim Liu    created separate update_replay_buffer()
# 10/09/19    Tim Liu    removed DDQN option
# 10/09/19    Tim Liu    combined cycles and epochs into single term epochs
# 10/09/19    Tim Liu    modified to call plain DQN and with HER flag each
#                        time program is called
# 10/09/19    Tim Liu    renamed main() to flip_bits; type of HER passed as
#                        function argument
# 10/09/19    Tim Liu    changed variable bit_size to num_bits
# 10/16/19    Tim Liu    reset the Q-networks and replay buffer between calls to flip_bits
# 10/16/19    Tim Liu    removed sections for students

import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from BitFlip import BitFlipEnv
from buffers import Buffer
from matplotlib import pyplot as plt

flags = tf.app.flags
flags.DEFINE_string("HER", "None", "different strategies of choosing goal. Possible values are :- future, final, episode or None. If None HER is not used")
flags.DEFINE_integer("num_bits", 15, "number of bits in the bit-flipping environment")
flags.DEFINE_integer("num_epochs", 250, "Number of epochs to run training for")
flags.DEFINE_integer("log_interval", 5, "Epochs between printing log info")
flags.DEFINE_integer("opt_steps", 40, "Optimization steps in each epoch")



FLAGS = flags.FLAGS

class Model(object) :
    '''Define Q-model'''

    def __init__(self, num_bits, scope, reuse) :

        # initialize model
        hidden_dim = 256

        with tf.variable_scope(scope, reuse=reuse) :

            self.inp = tf.placeholder(shape = [None, num_bits*2],dtype = tf.float32)

            net = self.inp
            net = slim.fully_connected(net,hidden_dim,activation_fn = tf.nn.relu)

            # running policy: argmax_a Q(s,a)
            # self.predict = action to take
            self.out = slim.fully_connected(net,num_bits,activation_fn = None)
            self.predict = tf.argmax(self.out, axis=1)

            # actual taken action
            self.action_taken = tf.placeholder(shape = [None],dtype = tf.int32)
            action_one_hot = tf.one_hot(self.action_taken, num_bits)

            # predicted q value for that action
            Q_val = tf.reduce_sum(self.out*action_one_hot,axis = 1)

            # actual q value for that action
            self.Q_target = tf.placeholder(shape = [None],dtype = tf.float32)

            self.loss = tf.reduce_mean(tf.square(Q_val - self.Q_target))
            self.train_step = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(self.loss)


def update_target_graph(from_scope,to_scope,tau):
    '''update the target network by copying over the weights from the policy
    network to the target network'''
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    ops = []
    for (var1,var2) in zip(from_vars,to_vars) :
        ops.append(var2.assign(var2*tau + (1-tau)*var1))

    return ops  

def updateTarget(ops,sess) :
    for op in ops :
        sess.run(op)


# ************   Define global variables and initialize    ************ #

num_bits = FLAGS.num_bits        # number of bits in the bit_flipping environment
tau = 0.95                       # Polyak averaging parameter
buffer_size = 1e6                # maximum number of elements in the replay buffer
batch_size = 128                 # number of samples to draw from the replay_buffer

num_epochs = FLAGS.num_epochs    # epochs to run training for
num_episodes = 16                # episodes to run in the environment per epoch
num_relabeled = 4                # relabeled experiences to add to replay_buffer each pass                           
gamma = 0.98                     # weighting past and future rewards

# create bit flipping environment and replay buffer
bit_env = BitFlipEnv(num_bits)
replay_buffer = Buffer(buffer_size,batch_size)

# set up Q-policy (model) and Q-target (target_model)
model = Model(num_bits,scope = 'model',reuse = False)
target_model = Model(num_bits,scope = 'target_model',reuse = False)

update_ops_initial = update_target_graph('model','target_model',tau = 0.0)
update_ops = update_target_graph('model','target_model',tau = tau)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# start by making Q-target and Q-policy the same
updateTarget(update_ops_initial,sess)



# ************   Helper functions    ************ #

def solve_environment(state, goal_state, total_reward):
    '''attempt to solve the bit flipping environment using the current policy'''
    
    # list for recording what happened in the episode
    episode_experience = []
    succeeded = False

    state = np.concatenate([state, goal_state])

    for t in range(num_bits):
        # attempt to solve the state - number of steps given to solve the
        # state is equal to the size of the vector

        # forward pass to find action
        action = sess.run(model.predict, feed_dict = {model.inp : [state]})[0]

        # take the action
        next_state, reward, done, _ = bit_env.step(action)
        next_state = np.concatenate([next_state, goal_state])

        # add to the episode experience (what happened)
        episode_experience.append((state,action,reward,next_state,goal_state))

        # calculate total reward
        total_reward+=reward
        # update state
        state = next_state
        # mark that we've finished the episode and succeeded with training

        if done:
            if succeeded:
                continue
            else:
                succeeded = True

    return succeeded, episode_experience, total_reward


def update_replay_buffer(episode_experience, HER):
    '''adds past experience to the replay buffer. Training is done with episodes from the replay
    buffer. When HER is used, relabeled experiences are also added to the replay buffer

    inputs: epsidode_experience - list of transitions from the last episode
    modifies: replay_buffer
    outputs: None'''

    for t in range(num_bits) :
        # copy actual experience from episode_experience to replay_buffer

        s,a,r,s_,g = episode_experience[t]

        # state
        inputs = s
        # next state
        inputs_ = s_
        # add to the replay buffer

        replay_buffer.add(inputs,a,r,inputs_)

        # when HER is used, each call to update_replay_buffer should add num_relabeled
        # relabeled points to the replay buffer

        if HER == 'None':
            # HER not being used, so do nothing
            pass

        elif HER == 'final':
            new_goal = episode_experience[-1][3][:num_bits]
            for i in range(num_relabeled):
                s, a, r, s_, g = episode_experience[i]
                s[-num_bits:] = new_goal
                s_[-num_bits:] = new_goal
                r = 0
                replay_buffer.add(s, a, r, s_)

        elif HER == 'future':
            # future - relabel based on future state. At each timestep t, relabel the
            # goal with a randomly select timestep between t and the end of the
            # episode

            random_int = np.random.choice(np.random.permutation(num_bits-t))
            new_goal = episode_experience[-random_int][3][:num_bits]

            for i in range(num_relabeled):
                try:
                    s, a, r, s_, g = episode_experience[i]
                    s[-num_bits:] = new_goal
                    s_[-num_bits:] = new_goal
                    r = 0
                    replay_buffer.add(s, a, r, s_)
                except:
                    pass

        elif HER == 'random':
            random_int = np.random.choice(np.random.permutation(num_bits))
            new_goal = episode_experience[-random_int][3][:num_bits]

            for i in range(num_relabeled):
                try:
                    s, a, r, s_, g = episode_experience[i]
                    s[-num_bits:] = new_goal
                    s_[-num_bits:] = new_goal
                    g = new_goal
                    r = 0
                    replay_buffer.add(s, a, r, s_)
                except:
                    pass

        else:
            print("Invalid value for Her flag - HER not used")
    return



def plot_success_rate(success_rates, labels):
    '''This function plots the success rate as a function of the number of cycles.
    The results are averaged over num_epochs epochs.

    inputs: success_rates - list with each element a list of success rates for
                            a epochs of running flip_bits
            labels - list of labels for each success_rate line'''

    for i in range(len(success_rates)):
        plt.plot(success_rates[i], label=labels[i])

    plt.xlabel('Epochs')
    plt.ylabel('Success rate')
    plt.title('Success rate with %d bits' %num_bits)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    return


# ************   Main training loop    ************ #

def flip_bits(HER = "None"):
    '''Main loop for running in the bit flipping environment. The DQN is
    trained over num_epochs. In each epoch, the agent runs in the environment
    num_episodes number of times. The Q-target and Q-policy networks are
    updated at the end of each epoch. Within one episode, Q-policy attempts
    to solve the environment and is limited to the same number as steps as the
    size of the environment

    inputs: HER - string specifying whether to use HER'''

    print("Running bit flip environment with %d bits and HER policy: %s" %(num_bits, HER))


    total_loss = []                  # training loss for each epoch
    success_rate = []                # success rate for each epoch
    
    for i in range(num_epochs):
        # Run for a fixed number of epochs

        total_reward = 0.0           # total reward for the epoch
        successes = []               # record success rate for each episode of the epoch
        losses = []                  # loss at the end of each epoch

        for k in range(num_episodes):
            # Run in the environment for num_episodes  

            state, goal_state = bit_env.reset()             # reset the environment

            # attempt to solve the environment
            succeeded, episode_experience, total_reward = solve_environment(state, goal_state, total_reward)

            successes.append(succeeded)                     # track whether we succeeded in environment 
            update_replay_buffer(episode_experience, HER)   # add to the replay buffer; use specified  HER policy
             
        for k in range(FLAGS.opt_steps):
            # optimize the Q-policy network

            # sample from the replay buffer
            state,action,reward,next_state = replay_buffer.sample()

            # forward pass through target network   
            target_net_Q = sess.run(target_model.out,feed_dict = {target_model.inp : next_state})

            # calculate target q value
            target_reward = np.clip(np.reshape(reward,[-1]) + gamma * np.reshape(np.max(target_net_Q,axis = -1),[-1]),-1. / (1 - gamma), 0)

            # calculate loss
            _,loss = sess.run([model.train_step,model.loss],feed_dict = {model.inp : state, model.action_taken : np.reshape(action,[-1]), model.Q_target : target_reward})

            # append loss from this optimization step to the list of losses
            losses.append(loss)

        
        updateTarget(update_ops,sess)                 # update target model by copying Q-policy to Q-target      
        success_rate.append(np.mean(successes))       # append mean success rate for this epoch

        if i % FLAGS.log_interval == 0:
            print('Epoch: %d  Cumulative reward: %f  Success rate: %.4f Mean loss: %.4f' % (i, total_reward, np.mean(successes), np.mean(losses)))
                
    return success_rate


if __name__ == "__main__":
    success_rate   = flip_bits(HER = FLAGS.HER)        # run again with type of HER specified
    # pass success rate for each run as first argument and labels as second list
    plot_success_rate([success_rate], [FLAGS.HER])






