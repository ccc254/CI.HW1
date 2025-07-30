import random

def load_data(file_path):
    """
    Loads data from a CSV file without using external libraries.
    Returns data as a list of lists of floats.
    Assumes the first row is a header and skips it.
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            header_skipped = False
            for line in f:
                if not header_skipped:
                    header_skipped = True
                    continue # Skip the header line
                
                line = line.strip() # Remove leading/trailing whitespace
                if not line: # Skip empty lines
                    continue
                
                try:
                    # Split by comma and convert each part to float
                    row = [float(val.strip()) for val in line.split(',')]
                    data.append(row)
                except ValueError as ve:
                    raise ValueError(f"Error parsing line: '{line}'. All values must be numerical. Details: {ve}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading data from {file_path}: {e}")
    return data

def calculate_min_max(data_list):
    """
    Calculates min and max values for each feature (column) in a list of lists.
    Returns two lists: [min_val_feature1, min_val_feature2, ...] and [max_val_feature1, max_val_feature2, ...]
    """
    if not data_list:
        return [], []

    num_features = len(data_list[0])
    min_vals = [float('inf')] * num_features
    max_vals = [float('-inf')] * num_features

    for row in data_list:
        for i in range(num_features):
            if row[i] < min_vals[i]:
                min_vals[i] = row[i]
            if row[i] > max_vals[i]:
                max_vals[i] = row[i]
    return min_vals, max_vals


def normalize_data(data_list):
    """
    Normalizes data (list of lists) to a range of [0, 1] using Min-Max scaling.
    Returns normalized data along with min and max values for denormalization.
    """
    if not data_list:
        return [], [], []

    min_vals, max_vals = calculate_min_max(data_list)
    normalized_data = []

    for row in data_list:
        new_row = []
        for i, val in enumerate(row):
            range_val = max_vals[i] - min_vals[i]
            if range_val == 0:
                # Handle cases where all values in a feature are the same (e.g., if a column is constant)
                new_row.append(0.5) # Normalize to 0.5 (mid-point)
            else:
                new_row.append((val - min_vals[i]) / range_val)
        normalized_data.append(new_row)
        
    return normalized_data, min_vals, max_vals

def denormalize_data(normalized_data_input, min_val_feature, max_val_feature):
    """
    Denormalizes data from [0, 1] range back to its original scale.
    normalized_data_input can be a single float, a list of floats (flat list), or a list of lists (matrix).
    min_val_feature, max_val_feature are single float values (since we denormalize one feature at a time, e.g., T_plus_7).
    """
    original_data = []
    
    # Calculate the range for denormalization
    range_val = max_val_feature - min_val_feature

    if isinstance(normalized_data_input, list):
        if not normalized_data_input:
            return []
        
        # Check if it's a list of lists (matrix) or a flat list
        if isinstance(normalized_data_input[0], list):
            # It's a list of lists (matrix), usually for X
            for row in normalized_data_input:
                new_row = []
                for val in row: # Assumes min_val_feature/max_val_feature are lists for each feature
                    original_val = val * range_val + min_val_feature
                    new_row.append(original_val)
                original_data.append(new_row)
        else:
            # It's a flat list (vector), usually for y (predictions/actuals)
            for val in normalized_data_input:
                original_val = val * range_val + min_val_feature
                original_data.append(original_val)
    else: # It's a single float value
        original_data = normalized_data_input * range_val + min_val_feature
    
    return original_data


def split_data(X, y, test_size=0.15, val_size=0.15, random_seed=None):
    """
    Splits data into training, validation, and test sets.
    X, y are lists of lists.
    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    if random_seed is not None:
        random.seed(random_seed)

    # Combine X and y for shuffling
    combined = list(zip(X, y))
    random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)
    
    total_samples = len(X_shuffled)
    
    # Calculate sizes for each split
    test_samples = int(total_samples * test_size)
    val_samples = int(total_samples * val_size)
    train_samples = total_samples - test_samples - val_samples

    # Splitting into train, validation, test
    X_train = list(X_shuffled[:train_samples])
    y_train = list(y_shuffled[:train_samples])

    X_val = list(X_shuffled[train_samples : train_samples + val_samples])
    y_val = list(y_shuffled[train_samples : train_samples + val_samples])
    
    X_test = list(X_shuffled[train_samples + val_samples :])
    y_test = list(y_shuffled[train_samples + val_samples :])

    return X_train, y_train, X_val, y_val, X_test, y_test
