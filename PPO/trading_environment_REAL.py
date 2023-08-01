import time
import math
import decimal
import requests
import hmac
import hashlib
import uuid
import base64
import json


api_key = ""
api_secret = ""
api_passphrase = ""


# Initialize point zero fake balances
fake_balances = {
    'USDT': 1000,
    'VET3L': 0,
    'VET3S': 0
}


class TradingEnvironment:
    def __init__(self):
        self.num_actions = 3
        self.state_dim = (21,)
        self.previous_total_balance = 0.0
        self.step_count = 0
        self.max_steps = 300  # Set the maximum number of steps allowed in an episode

    def reset(self):
        state_values = self.get_state()
        state = state_values[:21]
        print('state', state)
        self.previous_total_balance = 0.0  # Reset the previous_total_balance when a new episode starts
        self.step_count = 0  # Reset the step counter when a new episode starts
        return state

    def get_state(self):

        # Get orderbook info at time point + normalize by logarithmic transformation
        def get_price_and_log_price(symbol, time_offset):

            # Get the current time and add the time_offset
            now_prices = int((time.time() + time_offset) * 1000)
            url_orderbook = f'https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}-USDT'
            str_to_sign_orderbook = str(now_prices) + 'GET' + f'/api/v1/market/orderbook/level1?symbol={symbol}-USDT'
            signature_orderbook = base64.b64encode(
                hmac.new(api_secret.encode('utf-8'), str_to_sign_orderbook.encode('utf-8'), hashlib.sha256).digest())
            passphrase_orderbook = base64.b64encode(
                hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
            headers_orderbook = {
                "KC-API-SIGN": signature_orderbook,
                "KC-API-TIMESTAMP": str(now_prices),
                "KC-API-KEY": api_key,
                "KC-API-PASSPHRASE": passphrase_orderbook,
                "KC-API-KEY-VERSION": "2"
            }
            response_orderbook = requests.request('get', url_orderbook, headers=headers_orderbook)
            price = response_orderbook.json()['data']['price']
            log_price = math.log(float(price))
            print(f'{symbol}_price =', price)
            print(f'log_{symbol}_price =', log_price)

            return price, log_price

        # Set time offsets for price timepoints and create a prices_and_logs list every time function get_state() is called
        time_offsets = [0, 1800, 1800, 1800]
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
                print('mean_x =', mean_x)
                print('mean_y =', mean_y)

                numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(len(x_values)))
                print('numerator =', numerator)
                denominator = sum((x_values[i] - mean_x) ** 2 for i in range(len(y_values)))
                print('denominator =', denominator)
                slope = numerator / denominator
                print('slope =', slope)
                y_intercept = mean_y - slope * mean_x
                print('y_intercept =', y_intercept)
                return slope, y_intercept

            def angle_between_linear_mean_and_x_axis(slope):
                angle = math.atan(float(slope))
                angle = angle * 180 / math.pi
                print('angle =', angle)
                return angle

            x1 = [x[0], x[1]]
            y1 = [y[0], y[1]]
            slope1, y_intercept1 = linear_mean(x1, y1)
            print('slope1 =', slope1)
            alpha1 = angle_between_linear_mean_and_x_axis(slope1)
            if y_intercept1 < 0:
                alpha1 *= -1

            x2 = [x[1], x[2]]
            y2 = [y[1], y[2]]
            slope2, y_intercept2 = linear_mean(x2, y2)
            print('slope2 =', slope2)
            alpha2 = angle_between_linear_mean_and_x_axis(slope2)
            if y_intercept2 < 0:
                alpha2 *= -1

            x3 = [x[2], x[3]]
            y3 = [y[2], y[3]]
            slope3, y_intercept3 = linear_mean(x3, y3)
            print('slope3 =', slope3)
            alpha3 = angle_between_linear_mean_and_x_axis(slope3)
            if y_intercept3 < 0:
                alpha3 *= -1

            # Normalize angles (alphas)
            min_value = -90
            max_value = 90
            alpha1_angle_normalized = (alpha1 - min_value) / (max_value - min_value)
            alpha2_angle_normalized = (alpha2 - min_value) / (max_value - min_value)
            alpha3_angle_normalized = (alpha3 - min_value) / (max_value - min_value)

            print('alpha1 =', alpha1)
            print('alpha2 =', alpha2)
            print('alpha3 =', alpha3)

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

        print("VET Normalized Angles:")
        print("alpha1_vet_normalized:", alpha1_vet_normalized)
        print("alpha2_vet_normalized:", alpha2_vet_normalized)
        print("alpha3_vet_normalized:", alpha3_vet_normalized)

        print("\nVET3L Normalized Angles:")
        print("alpha1_vet3l_normalized:", alpha1_vet3l_normalized)
        print("alpha2_vet3l_normalized:", alpha2_vet3l_normalized)
        print("alpha3_vet3l_normalized:", alpha3_vet3l_normalized)

        print("\nVET3S Normalized Angles:")
        print("alpha1_vet3s_normalized:", alpha1_vet3s_normalized)
        print("alpha2_vet3s_normalized:", alpha2_vet3s_normalized)
        print("alpha3_vet3s_normalized:", alpha3_vet3s_normalized)

        # wait 2 seconds before continuing
        time.sleep(2)

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
        # Get account info
        now3 = int(time.time() * 1000)
        url_accounts = 'https://api.kucoin.com/api/v1/accounts'
        str_to_sign_accounts = str(now3) + 'GET' + '/api/v1/accounts'
        signature_accounts = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), str_to_sign_accounts.encode('utf-8'), hashlib.sha256).digest())
        passphrase_accounts = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
        headers_accounts = {
            "KC-API-SIGN": signature_accounts,
            "KC-API-TIMESTAMP": str(now3),
            "KC-API-KEY": api_key,
            "KC-API-PASSPHRASE": passphrase_accounts,
            "KC-API-KEY-VERSION": "2"
        }
        response_accounts = requests.request('get', url_accounts, headers=headers_accounts)
        print(response_accounts.status_code)
        print(response_accounts.json())

        # Get account balance of USDT
        usdt_balance2 = 0
        for account in response_accounts.json()['data']:
            if account['currency'] == 'USDT' and account['type'] == 'trade':
                usdt_balance2 = account['available']
        print('usdt_balance2 =', usdt_balance2)

        # Get account balance of VET3L
        vet3l_balance2 = 0
        for account in response_accounts.json()['data']:
            if account['currency'] == 'VET3L' and account['type'] == 'trade':
                vet3l_balance2 = account['available']
        print('vet3l_balance2 =', vet3l_balance2)

        # Get account balance of VET3S
        vet3s_balance2 = 0
        for account in response_accounts.json()['data']:
            if account['currency'] == 'VET3S' and account['type'] == 'trade':
                vet3s_balance2 = account['available']
        print('vet3s_balance2 =', vet3s_balance2)

        return {
            'USDT': usdt_balance2,
            'VET3L': vet3l_balance2,
            'VET3S': vet3s_balance2
        }

    def sell_vet3l_buy_vet3s(self):

        # sell maximum available amount of VET3L and buy maximum available amount VET3S

        # Call the function to get the account balances
        balances = self.get_balances()
        usdt_balance = balances['USDT']
        vet3l_balance = balances['VET3L']
        vet3s_balance = balances['VET3S']
        now1 = int(time.time() * 1000)
        url_sell = 'https://api.kucoin.com/api/v1/orders'
        # Generate a unique clientOid
        client0id = str(uuid.uuid4())

        # Calculate the size based on the baseIncrement and baseMinSize
        baseMinSize = 0.1
        baseIncrement = 0.0001
        a = vet3l_balance
        print('vet3l_balance_before_sell = ', vet3l_balance)
        decimal_val = decimal.Decimal(str(a)).quantize(
            decimal.Decimal('.000001'),
            rounding=decimal.ROUND_DOWN
        )
        size_vet3l = float(decimal_val)

        data_sell = {
            "clientOid": client0id,
            "side": "sell",
            "symbol": "VET3L-USDT",
            "size": size_vet3l,
            "type": "market",
            "price": ""
        }
        str_to_sign_sell = str(now1) + 'POST' + '/api/v1/orders' + json.dumps(data_sell)
        signature_sell = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), str_to_sign_sell.encode('utf-8'), hashlib.sha256).digest())
        passphrase_sell = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
        headers_sell = {
            "Content-Type": "application/json",
            "KC-API-SIGN": signature_sell,
            "KC-API-TIMESTAMP": str(now1),
            "KC-API-KEY": api_key,
            "KC-API-PASSPHRASE": passphrase_sell,
            "KC-API-KEY-VERSION": "2"
        }
        response_sell = requests.request('post', url_sell, headers=headers_sell, data=json.dumps(data_sell))
        print(response_sell.status_code)
        print(response_sell.content)
        # wait 5 seconds before looping again
        while True:
            time.sleep(10)
            break

        # Call the function to get the account balances
        balances = self.get_balances()
        usdt_balance_second = balances['USDT']
        vet3l_balance_second = balances['VET3L']
        vet3s_balance_second = balances['VET3S']

        now2 = int(time.time() * 1000)
        url_buy = 'https://api.kucoin.com/api/v1/orders'
        # Generate a unique clientOid
        client0id = str(uuid.uuid4())

        # Calculate the funds based on the quoteIncrement and quoteMinSize
        quoteMinSize = 0.1
        quoteIncrement = 0.1
        usdt_balance_second = float(usdt_balance_second)
        c = usdt_balance_second - (0.001 * usdt_balance_second)
        print('usdt_balance_second_before_buy = ', c)
        decimal_val = decimal.Decimal(str(c)).quantize(
            decimal.Decimal('.1'),
            rounding=decimal.ROUND_DOWN
        )
        funds_usdt = float(decimal_val)

        data_buy = {
            "clientOid": client0id,
            "side": "buy",
            "symbol": "VET3S-USDT",
            "funds": funds_usdt,
            "type": "market",
            "price": ""
        }
        str_to_sign_buy = str(now2) + 'POST' + '/api/v1/orders' + json.dumps(data_buy)
        signature_buy = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), str_to_sign_buy.encode('utf-8'), hashlib.sha256).digest())
        passphrase_buy = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
        headers_buy = {
            "Content-Type": "application/json",
            "KC-API-SIGN": signature_buy,
            "KC-API-TIMESTAMP": str(now2),
            "KC-API-KEY": api_key,
            "KC-API-PASSPHRASE": passphrase_buy,
            "KC-API-KEY-VERSION": "2"
        }
        response_buy = requests.request('post', url_buy, headers=headers_buy, data=json.dumps(data_buy))
        print(response_buy.status_code)
        print(response_buy.content)

    def sell_vet3s_buy_vet3l(self):
        # sell maximum available amount of VET3S and buy maximum available amount VET3L

        # Call the function to get the account balances
        balances = self.get_balances()
        usdt_balance_third = balances['USDT']
        vet3l_balance_third = balances['VET3L']
        vet3s_balance_third = balances['VET3S']
        now1 = int(time.time() * 1000)
        url_sell = 'https://api.kucoin.com/api/v1/orders'
        # Generate a unique clientOid
        client0id = str(uuid.uuid4())

        # Calculate the size based on the baseIncrement and baseMinSize
        baseMinSize = 0.1
        baseIncrement = 0.0001
        b = vet3s_balance_third
        print('vet3s_balance_before_sell = ', vet3s_balance_third)
        decimal_val = decimal.Decimal(str(b)).quantize(
            decimal.Decimal('.00001'),
            rounding=decimal.ROUND_DOWN
        )
        size_vet3s = float(decimal_val)

        data_sell = {
            "clientOid": client0id,
            "side": "sell",
            "symbol": "VET3S-USDT",
            "size": size_vet3s,
            "type": "market",
            "price": ""
        }
        str_to_sign_sell = str(now1) + 'POST' + '/api/v1/orders' + json.dumps(data_sell)
        signature_sell = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), str_to_sign_sell.encode('utf-8'), hashlib.sha256).digest())
        passphrase_sell = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
        headers_sell = {
            "Content-Type": "application/json",
            "KC-API-SIGN": signature_sell,
            "KC-API-TIMESTAMP": str(now1),
            "KC-API-KEY": api_key,
            "KC-API-PASSPHRASE": passphrase_sell,
            "KC-API-KEY-VERSION": "2"
        }
        response_sell = requests.request('post', url_sell, headers=headers_sell, data=json.dumps(data_sell))
        print(response_sell.status_code)
        print(response_sell.content)
        # wait 5 seconds before looping again
        while True:
            time.sleep(10)
            break

        # Call the function to get the account balances
        balances = self.get_balances()
        usdt_balance_fourth = balances['USDT']
        vet3l_balance_fourth = balances['VET3L']
        vet3s_balance_fourth = balances['VET3S']

        now2 = int(time.time() * 1000)
        url_buy = 'https://api.kucoin.com/api/v1/orders'
        # Generate a unique clientOid
        client0id = str(uuid.uuid4())

        # Calculate the funds based on the quoteIncrement and quoteMinSize
        quoteMinSize = 0.1
        quoteIncrement = 0.1
        usdt_balance_fourth = float(usdt_balance_fourth)
        d = usdt_balance_fourth - (0.001 * usdt_balance_fourth)
        print('usdt_balance_fourth_before_buy = ', d)
        decimal_val = decimal.Decimal(str(d)).quantize(
            decimal.Decimal('.1'),
            rounding=decimal.ROUND_DOWN
        )
        funds_usdt = float(decimal_val)

        data_buy = {
            "clientOid": client0id,
            "side": "buy",
            "symbol": "VET3L-USDT",
            "funds": funds_usdt,
            "type": "market",
            "price": ""
        }
        str_to_sign_buy = str(now2) + 'POST' + '/api/v1/orders' + json.dumps(data_buy)
        signature_buy = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), str_to_sign_buy.encode('utf-8'), hashlib.sha256).digest())
        passphrase_buy = base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
        headers_buy = {
            "Content-Type": "application/json",
            "KC-API-SIGN": signature_buy,
            "KC-API-TIMESTAMP": str(now2),
            "KC-API-KEY": api_key,
            "KC-API-PASSPHRASE": passphrase_buy,
            "KC-API-KEY-VERSION": "2"
        }
        response_buy = requests.request('post', url_buy, headers=headers_buy, data=json.dumps(data_buy))
        print(response_buy.status_code)
        print(response_buy.content)

    def do_nothing(self):
        pass

    def take_action(self, action_idx):
        actions_set = [self.sell_vet3l_buy_vet3s, self.sell_vet3s_buy_vet3l, self.do_nothing]
        actions_set[action_idx]()

    def get_reward(self, current_vet3l_price_reward, current_vet3s_price_reward):
        balances = self.get_balances()
        current_total_balance = float(balances['USDT']) + float(balances['VET3L']) * float(current_vet3l_price_reward) + float(balances['VET3S']) * float(current_vet3s_price_reward)

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


