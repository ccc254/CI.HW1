# C:\CI.HW1\utils\data_preprocessing.py

import random

def load_data_pat(file_path):
    """
    Loads data from the specific cross.pat file format.
    Assumes format:
    pX
    [input_feature1] [input_feature2]
    [output_class0] [output_class1] (one-hot encoded)
    
    Returns X (list of lists) and y (list of lists, where inner list is single element for binary class 0 or 1).
    """
    X = []
    y = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line: # Skip empty lines
                    i += 1
                    continue
                
                if line.startswith('p'): # It's a pattern header line
                    # Next line should be input features
                    i += 1
                    if i >= len(lines): break # End of file
                    input_line = lines[i].strip()
                    input_values = [float(val) for val in input_line.split()]
                    X.append(input_values)

                    # Next line should be output classes (one-hot)
                    i += 1
                    if i >= len(lines): break # End of file
                    output_line = lines[i].strip()
                    output_values = [int(val) for val in output_line.split()]
                    
                    # Convert one-hot to single binary label (0 or 1)
                    # If [1 0] -> 0, if [0 1] -> 1
                    if output_values == [1, 0]:
                        y.append([0])
                    elif output_values == [0, 1]:
                        y.append([1])
                    else:
                        raise ValueError(f"Unexpected output format: {output_values} in line '{output_line}'")
                
                i += 1 # Move to the next line
                
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}")
    except ValueError as ve:
        raise ValueError(f"Error parsing .pat file format. Details: {ve}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading data from {file_path}: {e}")
    return X, y

def calculate_min_max(data_list):
    """Calculates min and max for each feature across all samples."""
    if not data_list:
        return [], []
    num_features = len(data_list[0])
    min_vals = [float('inf')] * num_features
    max_vals = [float('-inf')] * num_features
    for sample in data_list:
        for i, val in enumerate(sample):
            if val < min_vals[i]:
                min_vals[i] = val
            if val > max_vals[i]:
                max_vals[i] = val
    return min_vals, max_vals

def normalize_data(data_list):
    """Normalizes data to a [0, 1] range based on min/max of each feature."""
    if not data_list:
        return [], [], []
    min_vals, max_vals = calculate_min_max(data_list)
    normalized_data = []
    for sample in data_list:
        normalized_sample = []
        for i, val in enumerate(sample):
            if max_vals[i] - min_vals[i] == 0:
                normalized_sample.append(0.0) # Handle constant features
            else:
                normalized_sample.append((val - min_vals[i]) / (max_vals[i] - min_vals[i]))
        normalized_data.append(normalized_sample)
    return normalized_data, min_vals, max_vals

def denormalize_data(normalized_data_input, min_val_feature, max_val_feature):
    """Denormalizes a single feature value back to its original scale."""
    # This function might not be directly used for classification outputs (0 or 1)
    # but is kept for consistency if needed for other regression tasks.
    if isinstance(normalized_data_input, list) and isinstance(normalized_data_input[0], list):
        # If it's a matrix of samples
        denormalized_matrix = []
        for sample in normalized_data_input:
            denormalized_sample = []
            for i, val in enumerate(sample):
                denormalized_sample.append(val * (max_val_feature[i] - min_val_feature[i]) + min_val_feature[i])
            denormalized_matrix.append(denormalized_sample)
        return denormalized_matrix
    else:
        # If it's a single value or list of single values for one feature
        return [val * (max_val_feature[0] - min_val_feature[0]) + min_val_feature[0] for val in normalized_data_input]

# split_data is not directly used for K-Fold, as K-Fold handles its own splitting.
# def split_data(X, y, test_size=0.15, val_size=0.15, random_seed=None):
#     ...