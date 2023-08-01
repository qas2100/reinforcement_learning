import csv
import math
import time


def find_initial_index(csv_file_path, target_value):
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for i, row in enumerate(reader):
            if int(row[0]) >= target_value:
                return i
    return 0


csv_file_paths = {
    'VET': r"your_file_path.klines_vet_data.csv",
    'VET3L': r"your_file_path.klines_vet3l_data.csv",
    'VET3S': r"your_file_path.klines_vet3S_data.csv",
}

last_indices = {
    'VET': find_initial_index(csv_file_paths['VET'], 1622707200),
    'VET3L': 0,
    'VET3S': 0,
}


def get_state():

    def get_price_and_log_price(symbol, time_offset):

        # Get the appropriate CSV file path for the symbol
        csv_file_path = csv_file_paths[symbol]

        # Read the rows in the CSV file and store them in a list
        rows = []
        with open(csv_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                rows.append(row)

        # Update the last index for the symbol and get the required price and log_price
        global last_indices
        last_indices[symbol] += time_offset
        # when the function reaches the last row[0], it will reset the last_indices[symbol] to the initial index value
        if last_indices[symbol] >= len(rows):
            last_indices[symbol] = find_initial_index(csv_file_paths[symbol], 1622707200) if symbol == 'VET' else 0

        current_index = last_indices[symbol]

        # Print the content of row[0] used
        print(f'{symbol}_row[0] =', rows[current_index][0])

        price = (float(rows[current_index][1]) + float(rows[current_index][2])) / 2
        log_price = math.log(float(price))
        print(f'{symbol}_price =', price)
        print(f'log_{symbol}_price =', log_price)

        return price, log_price

    # Set time offsets for price timepoints and create a prices_and_logs list every time function get_state() is called

    time_offsets = [0, 1, 1, 1]
    prices_and_logs = []

    for offset in time_offsets:
        time.sleep(offset)  # Wait for the specified offset (if it's 0, it won't wait)

        price_vet, log_price_vet = get_price_and_log_price('VET', offset)
        price_vet3l, log_price_vet3l = get_price_and_log_price('VET3L', offset)
        price_vet3s, log_price_vet3s = get_price_and_log_price('VET3S', offset)
        prices_and_logs.append((price_vet, log_price_vet, price_vet3l, log_price_vet3l, price_vet3s, log_price_vet3s))

    previous_price, log_previous_price, previous_vet3l_price, log_previous_vet3l_price, previous_vet3s_price, log_previous_vet3s_price = prices_and_logs[0]
    mid_price, log_mid_price, mid_vet3l_price, log_mid_vet3l_price, mid_vet3s_price, log_mid_vet3s_price = prices_and_logs[1]
    penultimate_price, log_penultimate_price, penultimate_vet3l_price, log_penultimate_vet3l_price, penultimate_vet3s_price, log_penultimate_vet3s_price = prices_and_logs[2]
    current_price, log_current_price, current_vet3l_price, log_current_vet3l_price, current_vet3s_price, log_current_vet3s_price = prices_and_logs[3]

    # wait  seconds before continuing
    time.sleep(1)

    return (
        log_previous_price, log_mid_price, log_penultimate_price, log_current_price,
        log_previous_vet3l_price, log_mid_vet3l_price, log_penultimate_vet3l_price, log_current_vet3l_price,
        log_previous_vet3s_price, log_mid_vet3s_price, log_penultimate_vet3s_price, log_current_vet3s_price
    )


# !!!!!!!!!!!!!!!! THIS while True loop is JUST A PLACEHOLDER -- remove it for the final code implementation !!!!!!!!!!!
while True:
    test = get_state()
