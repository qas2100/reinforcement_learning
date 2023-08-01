# this is environment for training - for real trading use trading_environment_REAL !!!!
from trading_environment import TradingEnvironment
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import pickle

training_mode = True

# Set folders and files to save weights (online and target), replay buffer and reward history, epsilon, frame and episode counts and optimizer config
training_folder = "training_sessions/q_learning_full_alpha"
weights_folder = os.path.join(training_folder, "weights")
buffer_file = os.path.join(training_folder, "replay_buffer.pkl")
reward_history_file = os.path.join(training_folder, "reward_history.pkl")
epsilon_file = os.path.join(training_folder, "epsilon.pkl")
count_file = os.path.join(training_folder, "count.pkl")
optimizer_config_file = os.path.join(training_folder, "optimizer_config.pkl")

os.makedirs(training_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)

online_weights_file = os.path.join(weights_folder, "online_model_weights.h5")
target_weights_file = os.path.join(weights_folder, "target_model_weights.h5")

save_interval = 50  # Save every 100 steps

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.5  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 1000

# Definition of the environment
env = TradingEnvironment()

num_actions = 3
state_dim = 21

np.random.seed(seed)
tf.random.set_seed(seed)

# Create an initializer with defined seed and He Normal kernel initializer
initializer = tf.keras.initializers.HeNormal(seed=seed)


def create_q_model():
    # Use a simple MLP architecture
    inputs = layers.Input(shape=(state_dim,))

    layer1 = layers.Dense(1024, kernel_initializer=initializer)(inputs)
    batch_norm1 = layers.BatchNormalization()(layer1)
    activation1 = layers.Activation("relu")(batch_norm1)

    layer2 = layers.Dense(1024, kernel_initializer=initializer)(activation1)
    batch_norm2 = layers.BatchNormalization()(layer2)
    activation2 = layers.Activation("relu")(batch_norm2)

    layer3 = layers.Dense(1024, kernel_initializer=initializer)(activation2)
    batch_norm3 = layers.BatchNormalization()(layer3)
    activation3 = layers.Activation("relu")(batch_norm3)

    layer4 = layers.Dense(1024, kernel_initializer=initializer)(activation3)
    batch_norm4 = layers.BatchNormalization()(layer4)
    activation4 = layers.Activation("relu")(batch_norm4)

    layer5 = layers.Dense(1024, kernel_initializer=initializer)(activation4)
    batch_norm5 = layers.BatchNormalization()(layer5)
    activation5 = layers.Activation("relu")(batch_norm5)

    layer6 = layers.Dense(1024, kernel_initializer=initializer)(activation5)
    batch_norm6 = layers.BatchNormalization()(layer6)
    activation6 = layers.Activation("relu")(batch_norm6)

    action_layer = layers.Dense(num_actions, activation="linear", kernel_initializer=initializer)(activation6)

    return keras.Model(inputs=inputs, outputs=action_layer)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 10000
# Number of frames for exploration
epsilon_greedy_frames = 100000
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 500
# Using huber loss for stability
loss_function = keras.losses.Huber()

# Load (if files exist) online weights, target weights, replay buffer, reward history, epsilon, counts and optimizer config
if os.path.exists(online_weights_file):
    model.load_weights(online_weights_file)

if os.path.exists(target_weights_file):
    model_target.load_weights(target_weights_file)

if os.path.exists(buffer_file):
    with open(buffer_file, "rb") as f:
        action_history, state_history, state_next_history, rewards_history, done_history = pickle.load(f)

if os.path.exists(reward_history_file):
    with open(reward_history_file, "rb") as f:
        episode_reward_history, running_reward = pickle.load(f)

if os.path.exists(epsilon_file):
    with open(epsilon_file, "rb") as f:
        epsilon = pickle.load(f)

if os.path.exists(count_file):
    with open(count_file, "rb") as f:
        frame_count, episode_count = pickle.load(f)

if os.path.exists(optimizer_config_file):
    with open(optimizer_config_file, "rb") as f:
        optimizer_config = pickle.load(f)
        optimizer = keras.optimizers.Adam.from_config(optimizer_config)

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Frame: {frame_count}")

        # Save model weights (online and target) and replay buffer, epsilon, counts and optimizer config
        if training_mode and frame_count % save_interval == 0:
            model.save_weights(online_weights_file)
            model_target.save_weights(target_weights_file)

            with open(buffer_file, "wb") as f:
                pickle.dump((action_history, state_history, state_next_history, rewards_history, done_history), f)

            with open(epsilon_file, "wb") as f:
                pickle.dump(epsilon, f)

            with open(count_file, "wb") as f:
                pickle.dump((frame_count, episode_count), f)

            with open(optimizer_config_file, "wb") as f:
                pickle.dump(optimizer.get_config(), f)

        # Use epsilon-greedy for exploration (take random action only when training_mode = True)
        if training_mode:
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
                print('randomly chosen action', action)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                print('q-value chosen action', action)

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
        else:
            # Predict action Q-values from environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            print('q-value chosen action', action)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)
        print('state_next', state_next)

        episode_reward += reward

        # Save actions and states in replay buffer (only when training_mode = True)
        if training_mode:
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

        # Update every fourth frame and once batch size is over 32 (only when training_mode = True):
        if training_mode and frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability and double DQN to reduce value overestimations
            # Use the online network to select the best action for the next state
            next_actions = np.argmax(model.predict(state_next_sample), axis=1)

            # Use the target network to estimate the Q-value for the selected actions
            next_q_values = model_target.predict(state_next_sample)
            future_rewards = np.array([next_q_values[i, action] for i, action in enumerate(next_actions)])

            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * future_rewards

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update target network only when training_mode = True
        if training_mode and frame_count % update_target_network == 0:
            # update the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # Save the episode_reward_history and running_reward after each episode
    with open(reward_history_file, "wb") as f:
        pickle.dump((episode_reward_history, running_reward), f)

    episode_count += 1

    if running_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
