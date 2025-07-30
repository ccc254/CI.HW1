# C:\CI.HW1\utils\metrics.py

def predict_class(output_values, threshold=0.5):
    """
    Converts a list of output probabilities/activations (from sigmoid) to binary class predictions (0 or 1).
    output_values: A flat list of floats (predictions for each sample).
    threshold: The cutoff value to classify.
    Returns a flat list of integers (0 or 1).
    """
    return [1 if val >= threshold else 0 for val in output_values]

def calculate_confusion_matrix(y_true, y_pred_classes):
    """
    Calculates the confusion matrix for binary classification.
    y_true: A flat list of true class labels (0 or 1).
    y_pred_classes: A flat list of predicted class labels (0 or 1).
    Returns a dictionary with 'TP', 'TN', 'FP', 'FN'.
    """
    if len(y_true) != len(y_pred_classes):
        raise ValueError("y_true and y_pred_classes must have the same number of samples.")

    TP = 0  # True Positives: Actual 1, Predicted 1
    TN = 0  # True Negatives: Actual 0, Predicted 0
    FP = 0  # False Positives: Actual 0, Predicted 1
    FN = 0  # False Negatives: Actual 1, Predicted 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred_classes[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred_classes[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred_classes[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred_classes[i] == 0:
            FN += 1
    
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

def calculate_accuracy(y_true, y_pred_classes):
    """
    Calculates the accuracy score.
    y_true: A flat list of true class labels (0 or 1).
    y_pred_classes: A flat list of predicted class labels (0 or 1).
    """
    if len(y_true) != len(y_pred_classes):
        raise ValueError("y_true and y_pred_classes must have the same number of samples.")
    
    correct_predictions = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred_classes[i]:
            correct_predictions += 1
            
    return correct_predictions / len(y_true) if len(y_true) > 0 else 0