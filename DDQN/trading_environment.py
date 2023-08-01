import time
import math
import csv
import decimal


training_mode = True  # Set to True during training, and False during real-life trading

# Set state_changed to False in order to assure that get_balances function moves rows only when get_state does
state_changed = False


api_key = ""
api_secret = ""
api_passphrase = ""

# Global variables for signals
sell_vet3l_buy_vet3s_signal = False
sell_vet3s_buy_vet3l_signal = False

# Initialize point zero fake balances
fake_balances = {
    'USDT': 1000,
    'VET3L': 0,
    'VET3S': 0
}


def find_initial_index(csv_file_path, target_value):
    with open(csv_file_path, 'r', newline='') as file_training:
        reader = csv.reader(file_training)
        next(reader)  # Skip the header row
        for i, row in enumerate(reader):
            if int(row[0]) >= target_value:
                return i
    return 0


csv_file_training_paths = {
    'VET': r"your_file_path.klines_vet_data.csv",
    'VET3L': r"your_file_path.klines_vet3L_data.csv",
    'VET3S': r"your_file_path.klines_vet3S_data.csv",
}

last_indices = {
    'VET': find_initial_index(csv_file_training_paths['VET'], 1622709000),
    'VET3L': 0,
    'VET3S': 0,
}

first_step = {
    'VET': True,
    'VET3L': True,
    'VET3S': True
}


class TradingEnvironment:

    # Assuming these variables are defined globally and initialized with some default values
    vet3l_price_previous = 1
    vet3s_price_previous = 1

    def __init__(self):
        self.num_actions = 3
        self.state_dim = (21,)
        self.previous_total_balance = 0.0
        self.step_count = 0
        self.max_steps = 1000  # Set the maximum number of steps allowed in an episode

    def reset(self):
        state_values = self.get_state()
        state = state_values[:21]
        print('state', state)
        balances = self.get_balances()
        self.previous_total_balance = float(balances['USDT']) + float(balances['VET3L']) * float(
            state_values[-2]) + float(balances['VET3S']) * float(state_values[-1])
        self.step_count = 0  # Reset the step counter when a new episode starts
        return state

    def get_state(self):
        global state_changed
        state_changed = True

        def get_price_and_log_price(symbol, time_offset):  # time offset used as a default value of 1

            # Get the appropriate CSV file path for the symbol
            csv_file_path = csv_file_training_paths[symbol]

            # Read the rows in the CSV file and store them in a list
            rows = []
            with open(csv_file_path, 'r', newline='') as file_training:
                reader = csv.reader(file_training)
                next(reader)  # Skip the header row
                for row in reader:
                    rows.append(row)

            # Update the last index for the symbol and get the required price and log_price
            global last_indices, first_step
            if first_step[symbol]:
                last_indices[symbol] += time_offset
                first_step[symbol] = False
            else:
                last_indices[symbol] += 1
            # when the function reaches the last row[0], it will reset the last_indices[symbol] to the initial index value
            if last_indices[symbol] >= len(rows):
                last_indices[symbol] = find_initial_index(csv_file_training_paths[symbol],
                                                          1622709000) if symbol == 'VET' else 0

            current_index = last_indices[symbol]

            price = (float(rows[current_index][1]) + float(rows[current_index][2])) / 2
            log_price = math.log(float(price))

            return price, log_price

        # Set time offsets for price timepoints and create a prices_and_logs list every time function get_state() is called

        time_offsets = [0, 0.5, 0.5, 0.5]
        prices_and_logs = []

        for offset in time_offsets:
            time.sleep(offset)  # Wait for the specified offset (if it's 0, it won't wait)

            price_vet, log_price_vet = get_price_and_log_price('VET', offset)
            price_vet3l, log_price_vet3l = get_price_and_log_price('VET3L', offset)
            price_vet3s, log_price_vet3s = get_price_and_log_price('VET3S', offset)
            prices_and_logs.append(
                (price_vet, log_price_vet, price_vet3l, log_price_vet3l, price_vet3s, log_price_vet3s))

        def calculate_normalized_angles(prices):
            # calculate alpha2 and alpha1 angle values in degrees
            x = [decimal.Decimal(0), decimal.Decimal(100), decimal.Decimal(200), decimal.Decimal(300)]
            y = [decimal.Decimal(price) for price in prices]

            nx = len(x)
            ny = len(y)

            def linear_mean(x_values, y_values):

                mean_x = sum(x_values) / len(x_values)
                mean_y = sum(y_values) / len(y_values)
                numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(len(x_values)))
                denominator = sum((x_values[i] - mean_x) ** 2 for i in range(len(y_values)))
                slope = numerator / denominator
                y_intercept = mean_y - slope * mean_x
                return slope, y_intercept

            def angle_between_linear_mean_and_x_axis(slope):
                angle = math.atan(float(slope))
                angle = angle * 180 / math.pi
                return angle

            x1 = [x[0], x[1]]
            y1 = [y[0], y[1]]
            slope1, y_intercept1 = linear_mean(x1, y1)
            alpha1 = angle_between_linear_mean_and_x_axis(slope1)
            if y_intercept1 < 0:
                alpha1 *= -1

            x2 = [x[1], x[2]]
            y2 = [y[1], y[2]]
            slope2, y_intercept2 = linear_mean(x2, y2)
            alpha2 = angle_between_linear_mean_and_x_axis(slope2)
            if y_intercept2 < 0:
                alpha2 *= -1

            x3 = [x[2], x[3]]
            y3 = [y[2], y[3]]
            slope3, y_intercept3 = linear_mean(x3, y3)
            alpha3 = angle_between_linear_mean_and_x_axis(slope3)
            if y_intercept3 < 0:
                alpha3 *= -1

            # Normalize angles (alphas)
            min_value = -90
            max_value = 90
            alpha1_angle_normalized = (alpha1 - min_value) / (max_value - min_value)
            alpha2_angle_normalized = (alpha2 - min_value) / (max_value - min_value)
            alpha3_angle_normalized = (alpha3 - min_value) / (max_value - min_value)

            return alpha1_angle_normalized, alpha2_angle_normalized, alpha3_angle_normalized

        previous_price, log_previous_price, previous_vet3l_price, log_previous_vet3l_price, previous_vet3s_price, log_previous_vet3s_price = prices_and_logs[0]
        mid_price, log_mid_price, mid_vet3l_price, log_mid_vet3l_price, mid_vet3s_price, log_mid_vet3s_price = prices_and_logs[1]
        penultimate_price, log_penultimate_price, penultimate_vet3l_price, log_penultimate_vet3l_price, penultimate_vet3s_price, log_penultimate_vet3s_price = prices_and_logs[2]
        current_price, log_current_price, current_vet3l_price, log_current_vet3l_price, current_vet3s_price, log_current_vet3s_price = prices_and_logs[3]

        # Define price lists
        prices_vet = [previous_price, mid_price, penultimate_price, current_price]
        prices_vet3l = [previous_vet3l_price, mid_vet3l_price, penultimate_vet3l_price, current_vet3l_price]
        prices_vet3s = [previous_vet3s_price, mid_vet3s_price, penultimate_vet3s_price, current_vet3s_price]

        # Call the function with the price lists and unpack the returned tuples - angles calculated from normal prices
        alpha1_vet_normalized, alpha2_vet_normalized, alpha3_vet_normalized = calculate_normalized_angles(prices_vet)
        alpha1_vet3l_normalized, alpha2_vet3l_normalized, alpha3_vet3l_normalized = calculate_normalized_angles(
            prices_vet3l)
        alpha1_vet3s_normalized, alpha2_vet3s_normalized, alpha3_vet3s_normalized = calculate_normalized_angles(
            prices_vet3s)

        # wait 2 seconds before continuing
        time.sleep(0)

        print('current_vet3l_price', current_vet3l_price)
        print('current_vet3s_price', current_vet3s_price)

        return (
            log_previous_price, log_mid_price, log_penultimate_price, log_current_price,
            alpha1_vet_normalized, alpha2_vet_normalized, alpha3_vet_normalized, alpha1_vet3l_normalized,
            alpha2_vet3l_normalized,
            alpha3_vet3l_normalized, alpha1_vet3s_normalized, alpha2_vet3s_normalized, alpha3_vet3s_normalized,
            log_previous_vet3l_price, log_mid_vet3l_price, log_penultimate_vet3l_price, log_current_vet3l_price,
            log_previous_vet3s_price, log_mid_vet3s_price, log_penultimate_vet3s_price, log_current_vet3s_price,
            current_vet3l_price, current_vet3s_price
        )

    def get_balances(self):
        global fake_balances, sell_vet3l_buy_vet3s_signal, sell_vet3s_buy_vet3l_signal

        def get_price_interior(symbol, time_offset=0):

            # Get the appropriate CSV file path for the symbol
            csv_file_path = csv_file_training_paths[symbol]

            # Read the rows in the CSV file and store them in a list
            rows = []
            with open(csv_file_path, 'r', newline='') as file_training:
                reader = csv.reader(file_training)
                next(reader)  # Skip the header row
                for row in reader:
                    rows.append(row)

            # Update the last index for the symbol and get the required price (only move forward into next row if get_state called)
            global state_changed, last_indices, first_step
            if state_changed:
                if first_step[symbol]:
                    last_indices[symbol] += time_offset
                    first_step[symbol] = False
                else:
                    last_indices[symbol] += 0

                # Set the state_changed flag back to False
                state_changed = False
            # when the function reaches the last row[0], it will reset the last_indices[symbol] to the initial index value
            if last_indices[symbol] >= len(rows):
                last_indices[symbol] = find_initial_index(csv_file_training_paths[symbol],
                                                          1622709000) if symbol == 'VET' else 0

            current_index = last_indices[symbol]

            price = (float(rows[current_index][1]) + float(rows[current_index][2])) / 2

            return price

        # Update balances based on signals from sell_vet3l_buy_vet3s and sell_vet3s_buy_vet3l functions
        if sell_vet3l_buy_vet3s_signal:
            # Sell VET3L and Buy VET3S
            vet3l_price = get_price_interior("VET3L")
            if vet3l_price / self.__class__.vet3l_price_previous > 100:
                fold_increase = vet3l_price / self.__class__.vet3l_price_previous
                vet3l_price /= fold_increase

            usdt_received = fake_balances['VET3L'] * vet3l_price * 0.999
            fake_balances['USDT'] += usdt_received
            fake_balances['VET3L'] = 0
            self.__class__.vet3l_price_previous = vet3l_price

            vet3s_price = get_price_interior("VET3S")
            if vet3s_price / self.__class__.vet3s_price_previous > 100:
                fold_increase = vet3s_price / self.__class__.vet3s_price_previous
                vet3s_price /= fold_increase

            vet3s_bought = (fake_balances['USDT'] * 0.999) / vet3s_price
            fake_balances['VET3S'] += vet3s_bought
            fake_balances['USDT'] = 0
            sell_vet3l_buy_vet3s_signal = False
            self.__class__.vet3s_price_previous = vet3s_price

        if sell_vet3s_buy_vet3l_signal:
            # Sell VET3S and Buy VET3L
            vet3s_price = get_price_interior("VET3S")
            if vet3s_price / self.__class__.vet3s_price_previous > 100:
                fold_increase = vet3s_price / self.__class__.vet3s_price_previous
                vet3s_price /= fold_increase

            usdt_received = fake_balances['VET3S'] * vet3s_price * 0.999
            fake_balances['USDT'] += usdt_received
            fake_balances['VET3S'] = 0
            self.__class__.vet3s_price_previous = vet3s_price

            vet3l_price = get_price_interior("VET3L")
            if vet3l_price / self.__class__.vet3l_price_previous > 100:
                fold_increase = vet3l_price / self.__class__.vet3l_price_previous
                vet3l_price /= fold_increase

            vet3l_bought = (fake_balances['USDT'] * 0.999) / vet3l_price
            fake_balances['VET3L'] += vet3l_bought
            fake_balances['USDT'] = 0
            sell_vet3s_buy_vet3l_signal = False
            self.__class__.vet3l_price_previous = vet3l_price

        print('USDT balance', fake_balances['USDT'])
        print('VET3L balance', fake_balances['VET3L'])
        print('VET3S balance', fake_balances['VET3S'])

        usdt_balance2 = fake_balances['USDT']
        vet3l_balance2 = fake_balances['VET3L']
        vet3s_balance2 = fake_balances['VET3S']

        return {
            'USDT': usdt_balance2,
            'VET3L': vet3l_balance2,
            'VET3S': vet3s_balance2
        }

    def sell_vet3l_buy_vet3s(self):
        # sell maximum available amount of VET3L and buy maximum available amount VET3S
        global sell_vet3l_buy_vet3s_signal
        sell_vet3l_buy_vet3s_signal = True

        return

    def sell_vet3s_buy_vet3l(self):
        # sell maximum available amount of VET3S and buy maximum available amount VET3L
        global sell_vet3s_buy_vet3l_signal
        sell_vet3s_buy_vet3l_signal = True

        return

    def do_nothing(self):
        pass

    def take_action(self, action_idx):
        actions_set = [self.sell_vet3l_buy_vet3s, self.sell_vet3s_buy_vet3l, self.do_nothing]
        actions_set[action_idx]()

    def get_reward(self, current_vet3l_price_reward, current_vet3s_price_reward):
        balances = self.get_balances()
        current_total_balance = float(balances['USDT']) + float(balances['VET3L']) * float(
            current_vet3l_price_reward) + float(balances['VET3S']) * float(current_vet3s_price_reward)

        # If the previous_total_balance is not zero and the current_total_balance is more than 100 times bigger
        if self.previous_total_balance > 0 and current_total_balance / self.previous_total_balance > 100:
            fold_increase = current_total_balance / self.previous_total_balance
            current_total_balance /= fold_increase

        reward_value = 0
        if current_total_balance > self.previous_total_balance:
            reward_value = 1
        elif current_total_balance < self.previous_total_balance:
            reward_value = -1

        self.previous_total_balance = current_total_balance

        print('current_total_balance', current_total_balance)

        return reward_value

    def step(self, action_idx):
        self.take_action(action_idx)
        next_state_values = self.get_state()
        next_state = next_state_values[:21]
        print('next state', next_state)
        current_vet3l_price_reward = next_state_values[-2]
        current_vet3s_price_reward = next_state_values[-1]
        reward = self.get_reward(current_vet3l_price_reward, current_vet3s_price_reward)

        # Update the step counter
        self.step_count += 1

        # Check if the maximum number of steps has been reached
        done = self.step_count >= self.max_steps

        return next_state, reward, done, {}


