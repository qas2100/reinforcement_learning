## Acknowledgments
Parts of the code were adapted from:
author(s): Jacob Chapman and Mathias Lechner
code: https://github.com/keras-team/keras-io/blob/master/examples/rl/deep_q_network_breakout.py

as well as generated using GPT transformer model 3.5 and 4

## Key features of the code -- this one is used for KuCoin and VET/USD, VET3L/USD, VET3S/USD pairs but it may be changed to incorporate others pairs/exchanges.

## MAIN TRADING ALGORITHM: double deep q-learning new.py

Environment: The code uses a custom environment for training, specifically for a trading task (TradingEnvironment). The environment should comply with the OpenAI's Gym interface, having methods like reset() and step().

Network Architecture: A Deep Q-Network (DQN) with six layers of 1024 neurons each is used, with batch normalization after each layer and ReLU activation function. The last layer uses linear activation function.

Optimizer: The Adam optimizer is used with a learning rate of 0.00025. The gradient norm is also clipped to 1.0 to avoid exploding gradients.

Epsilon-Greedy Strategy: The exploration strategy used is epsilon-greedy. This means that with a decreasing probability epsilon, a random action is chosen. The value of epsilon decreases linearly from 1.0 to 0.1 over 100,000 frames. This ensures that the model explores the environment initially and exploits learned knowledge later.

Replay Buffer: To avoid correlation between consecutive experiences and to improve sample efficiency, experience replay is used. The last 100,000 experiences (state, action, reward, next state, done flag) are saved and a batch of 32 experiences is sampled to train the network every 4 steps.

Target Network: A separate target network is used to generate the Q-values for the next state while computing the target for training the primary Q-network. The weights of the target network are updated to the primary network's weights every 500 steps.

Double DQN: The network uses the double DQN technique. The max operator in standard Q-learning and DQN uses the same values both to select and to evalate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates. To
prevent this, we can decouple the selection from the evaluation. This is the idea behind Double Q-learning (van Hasselt,
2010). Notice that the selection of the action, in the argmax, is
still due to the online weights θt. This means that, as in Q-learning, we are still estimating the value of the greedy policy according to the current values, as defined by θt. However, we use the second set of weights θ't
to fairly evaluate the value of this policy. This second set of weights can be updated symmetrically by switching the roles of θ and θ'

Loss Function: The Huber loss is used as it is less sensitive to outliers compared to the mean squared error loss. It combines the advantages of L1-loss (being less sensitive to outliers) and L2-loss (being smooth near the minimum).

Save & Load: The model parameters (weights), replay buffer, epsilon, frame and episode count, and optimizer configuration are periodically saved to disk. This allows to stop and resume training.

Monitoring: The code also computes and prints out the mean reward over the last 100 episodes. Training is considered done when this running reward exceeds 40.

Kernel Initializer: A He normal initializer with a specific seed is used for the initialization of the DQN.

Training mode: There's a flag named training_mode which when set to True enables features like epsilon-greedy exploration, storing experiences in the replay buffer, and network updates. If False, the network will simply execute its policy on the environment without any exploration or learning.

Gradient Clipping: This feature is implemented through the optimizer (Adam in this case), preventing the gradients from becoming too large and causing numerical issues.

Batch Normalization: It's used after every layer of the DQN, improving the training process by normalizing the activations of each layer.

## TRADING ENVIRONMENT (there are two versions of this file: training-wise trading_environment.py and inference-wise trading_environment_REAL.py)
Trading Environment Python Script
This Python script is designed to interact with the Kucoin API for the purpose of tracking and analyzing cryptocurrency trading prices. It calculates the angles of the trend lines in a given time frame and returns the current prices of the VET, VET3L and VET3S symbols. This information can be used for automated trading decision making.

Overview
At the start, the script initializes API key, API secret and API passphrase variables, which are required to communicate with the Kucoin API.

The TradingEnvironment class is defined to encapsulate the trading environment. It includes methods to get the current state of the environment (get_state), reset the environment (reset), and get the balance of the account (get_balances).

Initialization
The TradingEnvironment class is initialized with the following attributes:

num_actions: This refers to the number of possible actions. In this case, it is set to 3.
state_dim: This represents the dimension of the state space. Here, it is set to (21,).
previous_total_balance: This stores the total balance of the previous time step.
step_count: This counter keeps track of the steps in each episode.
max_steps: The maximum number of steps allowed in an episode, which is set to 300.
Key Methods
get_state
This method is used to fetch the current state of the market. It sends a GET request to the Kucoin API to retrieve the latest price information for the symbols VET, VET3L and VET3S. The function then calculates the logarithm of the price and returns it along with the price itself.

reset
This method is used to reset the environment. It returns the current state and sets the previous_total_balance and step_count back to 0.

get_balances
This method retrieves the current balances of USDT, VET3L, and VET3S. It sends a GET request to the Kucoin API and fetches the account balance information.

sell_vet3l_buy_vet3s
This method is used to perform a trading operation where the maximum available amount of VET3L is sold, and with the funds, the maximum available amount of VET3S is bought. It fetches the account balance, calculates the size of the VET3L that can be sold, sends a POST request to the Kucoin API to place the sell order, then fetches the account balance again, calculates the amount of USDT available for buying VET3S, and finally sends a POST request to place the buy order.

sell_vet3s_buy_vet3l
This method is similar to the previous one, but in this case, it sells the maximum available amount of VET3S and with the funds, buys the maximum available amount of VET3L.

do_nothing
As the name suggests, this method does not perform any action.

take_action
This method executes the action associated with a given action index. The actions are represented in the actions_set list, and the method associated with the given action index is executed.

get_reward
This method calculates and returns the reward. The reward is based on the change in total account balance, which is the sum of USDT balance and the dollar value of VET3L and VET3S holdings. If the total balance increases, the reward is 1; if it decreases, the reward is -1.

step
This is a key method in the trading environment. It takes an action index as an argument, performs the corresponding action, updates the environment state, calculates the reward, and determines whether the episode has ended. An episode ends when the number of steps reaches the maximum allowed steps, max_steps. The method returns the next state, the reward, a boolean indicating whether the episode has ended, and an empty dictionary for additional information.

** In case of trading environment used for training the following methods are changed in relation to the inference, real trading environment:

get_state: This method is used to fetch the current state of the environment. The state is fetched from CSV files that have pricing information and using those it calculates normalized angles for VET, VET3L, and VET3S. It then returns the current state consisting of prices, normalized angles, and logs of prices.

get_balances: This method updates the current balances of VET3L, VET3S, and USDT based on fake balances mechanism. It checks for signals from the sell_vet3l_buy_vet3s and sell_vet3s_buy_vet3l methods, and if there are any, it performs the respective selling and buying actions. It then returns the current balances for VET3L, VET3S, and USDT.


Usage
To use this script, you'll need to insert your Kucoin API key, secret, and passphrase into the respective variables at the top of the script. You will then be able to create an instance of the TradingEnvironment class and call its methods to interact with the Kucoin API.

Please note that this script is designed to be used in a learning context and as such may not be suitable for real-world trading purposes without further modifications or error handling.

The script also includes some print statements for debugging and verification purposes. They print out the current prices, logarithmic prices, and the computed angles.

Note that this bot operates in a specific mode where it only holds one type of asset at a time. That is, if it holds VET3L, it does not hold any VET3S, and vice versa. This is why when a trade signal is triggered, the bot sells all of its current asset and uses the proceeds to buy the other asset.

Additionally the training trading_environment code has the following features:

Fake Balances: The bot starts with some "fake balances" for training purposes. These balances represent the amounts of each asset the bot currently holds.

Balance Calculation: The get_balances method calculates the current balances of the assets based on fake balances created from data included in .csv files. If there is a signal to sell VET3L and buy VET3S, or vice versa, this method calculates the new balance after the transaction.

CSV Reading and Indexing: The bot uses CSV files for backtesting purposes. Each CSV file contains historical price data for a different asset. The bot then determines the starting index in the CSV file to begin reading data based on a target value.

## data retrieval from KuCoin 1.py
This Python script is designed to retrieve trading data for a specific cryptocurrency (in this case, 'VET' or VeChain) from the Kucoin API and write that data into a CSV file. In this case 30 minute intervals -- but one may choose different ones. The specific type of trading data being retrieved is 'Klines' data, which is a type of data used in technical analysis that describes price movements over time (also known as a candlestick chart). Here's a breakdown of what the script does:

Imports necessary libraries: Libraries such as requests, hmac, hashlib, base64, time, and csv are imported. These libraries will be used for making HTTP requests, creating HMAC signatures, hashing, base64 encoding, keeping track of time, and working with CSV files, respectively.

Declares API credentials: Three variables api_key, api_secret, and api_passphrase are declared for storing the Kucoin API credentials. These are left as empty strings and should be filled in with valid Kucoin API credentials.

Set trading pair and interval: The symbol variable is set to 'VET' (indicating VeChain) and the interval variable is set to '30min', which is the interval of the Klines data.

Get current timestamp: The now_prices variable gets the current time in milliseconds.

Set start times: A list of start_times is provided. These are UNIX timestamps that will be used to specify the start and end times for the data retrieval in the Kucoin API.

Create CSV file: A CSV file (klines_vet_data.csv) is created with headers for storing the data. The headers include: time, open, close, high, low, volume, and turnover.

Data retrieval and storage: The script enters a loop over the range of start times. In each iteration, it:

Forms the Kucoin API URL for retrieving Klines data for the 'VET-USDT' trading pair for the specified start and end times.

Creates HMAC signatures for the request using the API secret and the request parameters.

Sends the HTTP GET request to the Kucoin API with the necessary headers.

If the request is successful (status code 200), it prints out the retrieved data and appends it to the CSV file. The data is reversed before being written to the file, as the Kucoin API returns data in descending order of time.

If the request is not successful, it prints an error message including the status code and response text.

