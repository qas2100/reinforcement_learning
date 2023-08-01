# this is environment for training purposes -- for real trading use trading_environment_REAL
from trading_environment_PPO_basic import TradingEnvironment
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import scipy.signal
from scipy.signal import lfilter
import time
import pickle
import os

training_mode = True

# Define the directory where you want to save the file
directory = r"your\Python\project\folder\training_sessions\PPO"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Define the file path
file_path = os.path.join(directory, 'saved_state.pkl')


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.5, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=None, kernel_initializer='glorot_normal')(x)
        x = layers.BatchNormalization()(x)
        x = activation(x)
    return layers.Dense(units=sizes[-1], activation=output_activation, kernel_initializer='glorot_normal')(x)



def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maximizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)

    # Apply gradient clipping
    clip_value = 1.0
    clipped_policy_grads = [tf.clip_by_norm(grad, clip_value) for grad in policy_grads]
    policy_optimizer.apply_gradients(zip(clipped_policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl

@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
        print('value loss', value_loss)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)

    # Apply gradient clipping
    clip_value = 1.0
    clipped_value_grads = [tf.clip_by_norm(grad, clip_value) for grad in value_grads]
    value_optimizer.apply_gradients(zip(clipped_value_grads, critic.trainable_variables))


# Hyperparameters of the PPO algorithm
steps_per_epoch = 300
epochs = 10000
gamma = 0.5
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (1000, 1000, 1000, 500, 500, 500)


# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = TradingEnvironment()
observation_dimensions = 21
num_actions = 3

# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length
observation, episode_return, episode_length = env.reset(), 0, 0

# Set folders and files to save weights
training_folder = "training_sessions/PPO"
weights_folder = os.path.join(training_folder, "weights")

os.makedirs(training_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)

actor_weights_file = os.path.join(weights_folder, "actor_model_weights.h5")
critic_weights_file = os.path.join(weights_folder, "critic_model_weights.h5")

# Load model (actor/critic) weights (if files exist)
if os.path.exists(actor_weights_file):
    actor.load_weights(actor_weights_file)

if os.path.exists(critic_weights_file):
    critic.load_weights(critic_weights_file)

# Load the variables saved earlier
try:
    with open(file_path, 'rb') as f:
        saved_state = pickle.load(f)
        sum_return = saved_state['sum_return']
        sum_length = saved_state['sum_length']
        num_episodes = saved_state['num_episodes']
        buffer = saved_state['buffer']
        epoch_start = saved_state['epoch']
        observation = saved_state['observation']
        episode_return = saved_state['episode_return']
        episode_length = saved_state['episode_length']
except FileNotFoundError:
    epoch_start = 0  # or other appropriate values

# Iterate over the number of epochs
for epoch in range(epoch_start, epochs):
    if training_mode:
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        # Iterate over the steps of each epoch
        for t in range(steps_per_epoch):

            # Get the logits, action, and take one step in the environment
            observation = np.array(observation).reshape(1, -1)
            logits, action = sample_action(observation)
            observation_new, reward, done, _ = env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = critic(observation)
            logprobability_t = logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, action, reward, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else critic(np.array(observation).reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(), 0, 0

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

        # Save the variables below
        with open(file_path, 'wb') as f:
            pickle.dump({
                'sum_return': sum_return,
                'sum_length': sum_length,
                'num_episodes': num_episodes,
                'buffer': buffer,
                'epoch': epoch,
                'observation': observation,
                'episode_return': episode_return,
                'episode_length': episode_length
            }, f)

        # Save the model (actor/critic) weights
        actor.save_weights(actor_weights_file)
        critic.save_weights(critic_weights_file)

    else:
        # Only inference, no training or storing
        for t in range(steps_per_epoch):
            observation = np.array(observation).reshape(1, -1)
            logits, action = sample_action(observation)
            observation, reward, done, _ = env.step(action[0].numpy())

            if done or (t == steps_per_epoch - 1):
                observation = env.reset()
