def calculate_mse(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) between true and predicted values.
    Inputs are assumed to be flat lists of numerical values.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples.")

    squared_errors_sum = 0
    for i in range(len(y_true)):
        error = y_true[i] - y_pred[i]
        squared_errors_sum += error * error
    
    mse = squared_errors_sum / len(y_true)
    return mse
