import random
import math # math library is allowed for basic math operations like exp
from mlp_model.mlp import MLP
from utils.data_preprocessing import load_data, normalize_data, denormalize_data, split_data
from utils.metrics import calculate_mse

def main():
    # --- Configuration Parameters ---
    file_path = 'data/flood_data.csv'
    
    # MLP Model Hyperparameters
    input_size = 8  # S1_t-3 to S2_t-0
    hidden_size = 10 # You can adjust this
    output_size = 1 # T_plus_7
    learning_rate = 0.01
    epochs = 10000
    
    # --- 1. Load Data ---
    print(f"Loading data from {file_path}...")
    try:
        data = load_data(file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to load data. Exiting.")
        return

    # Separate input features (X) and target output (y)
    # X contains all columns except the last one (S1_t-3 to S2_t-0)
    # y is the last column (T_plus_7), reshaped to be a list of lists (for consistency as a 1x1 matrix)
    X = [row[:-1] for row in data]
    y = [[row[-1]] for row in data] # Keep y as list of lists (e.g., [[val1], [val2]])

    # --- 2. Normalize Data ---
    # Normalize X and y separately using Min-Max Scaling
    print("Normalizing input data...")
    X_normalized, x_min_vals, x_max_vals = normalize_data(X)
    print("Input data normalized.")

    print("Normalizing target output data...")
    y_normalized, y_min_vals, y_max_vals = normalize_data(y)
    print("Target output data normalized.")
    
    # Store min/max values for denormalization
    # Since y is a list of 1-element lists, y_min_vals/y_max_vals will be a list of 1 element.
    # We extract the single float value from them.
    y_min_max = (y_min_vals[0], y_max_vals[0]) 

    # --- 3. Split Data into Training, Validation, and Test Sets ---
    print("Splitting data into training, validation, and test sets...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_normalized, y_normalized, test_size=0.15, val_size=0.15, random_seed=42)
    print(f"Data split: Train={len(X_train)} samples, Validation={len(X_val)} samples, Test={len(X_test)} samples.")

    # --- 4. Initialize and Train the MLP Model ---
    print("Initializing MLP model...")
    mlp = MLP(input_size, hidden_size, output_size)
    print(f"MLP model initialized with {input_size} input, {hidden_size} hidden, {output_size} output layers.")

    print("Starting training...")
    train_losses, val_losses = mlp.train(X_train, y_train, X_val, y_val, learning_rate, epochs)
    print("Training complete.")

    # --- 5. Evaluate the Model on Test Set ---
    print("Evaluating model on test set...")
    
    # Make predictions on the normalized test set
    # mlp.forward returns [[float_value]]. We want to flatten it to [float_value] before denormalization.
    y_pred_normalized_flat = [mlp.forward(sample)[0][0] for sample in X_test]

    # Denormalize predictions back to original scale
    # denormalize_data now accepts and returns flat lists if given flat lists.
    y_pred_original = denormalize_data(y_pred_normalized_flat, y_min_max[0], y_min_max[1])
    
    # Flatten y_test (which is list of 1-element lists) to a simple list for denormalization
    y_test_flat_normalized = [val[0] for val in y_test]
    y_test_original = denormalize_data(y_test_flat_normalized, y_min_max[0], y_min_max[1])

    # Calculate Mean Squared Error (MSE)
    # Both y_test_original and y_pred_original should now be flat lists of floats.
    mse = calculate_mse(y_test_original, y_pred_original)
    print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")

    # Print first few actual vs. predicted values
    print("\nFirst 10 Actual vs. Predicted values (on original scale):")
    for i in range(min(10, len(y_test_original))):
        print(f"Actual: {y_test_original[i]:.2f}, Predicted: {y_pred_original[i]:.2f}")

if __name__ == "__main__":
    main()
