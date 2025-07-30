# C:\CI.HW1\main.py

import random
import math
from mlp_model.mlp import MLP
from utils.data_preprocessing import load_data_pat, normalize_data
from utils.metrics import predict_class, calculate_confusion_matrix, calculate_accuracy

def k_fold_split(X, y, k=10, random_seed=None):
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    if random_seed is not None:
        random.seed(random_seed)

    combined = list(zip(X, y))
    random.shuffle(combined)

    fold_size = len(combined) // k
    folds = []
    for i in range(k):
        test_start = i * fold_size
        test_end = test_start + fold_size
        
        test_fold = combined[test_start:test_end]
        train_fold = combined[:test_start] + combined[test_end:]
        
        X_train_fold, y_train_fold = zip(*train_fold)
        X_test_fold, y_test_fold = zip(*test_fold)
        
        folds.append({
            'X_train': list(X_train_fold),
            'y_train': list(y_train_fold),
            'X_test': list(X_test_fold),
            'y_test': list(y_test_fold)
        })
    return folds

def run_experiment(hidden_size, learning_rate, momentum, init_seed, k_folds=10):
    print(f"\n--- Running Experiment: Hidden={hidden_size}, LR={learning_rate}, Momentum={momentum}, Seed={init_seed} ---")
    
    file_path = 'data/cross.pat'
    
    try:
        X_raw, y_raw = load_data_pat(file_path)
        print("Data loaded successfully from cross.pat.")
    except Exception as e:
        print(f"Error loading data: {e}. Skipping experiment.")
        return

    X_normalized, x_min_vals, x_max_vals = normalize_data(X_raw)
    y_normalized, y_min_vals, y_max_vals = normalize_data(y_raw)

    input_size = len(X_normalized[0])
    output_size = len(y_normalized[0])

    all_fold_accuracies = []
    all_fold_confusion_matrices = []

    folds = k_fold_split(X_normalized, y_normalized, k=k_folds, random_seed=42)

    for i, fold in enumerate(folds):
        print(f"  --- Running Fold {i+1}/{k_folds} ---")
        
        random.seed(init_seed + i)
        
        mlp = MLP(input_size, hidden_size, output_size, momentum=momentum)
        
        epochs = 5000
        
        mlp.train(fold['X_train'], fold['y_train'], fold['X_test'], fold['y_test'], learning_rate, epochs) 
        
        y_test_pred_probs = []
        for sample_x_test in fold['X_test']:
            y_test_pred_probs.append(mlp.forward(sample_x_test)[0][0])

        y_test_true_flat = [val[0] for val in fold['y_test']]

        y_test_pred_classes = predict_class(y_test_pred_probs, threshold=0.5)

        fold_accuracy = calculate_accuracy(y_test_true_flat, y_test_pred_classes)
        all_fold_accuracies.append(fold_accuracy)

        fold_cm = calculate_confusion_matrix(y_test_true_flat, y_test_pred_classes)
        all_fold_confusion_matrices.append(fold_cm)

        print(f"    Fold {i+1} Test Accuracy: {fold_accuracy:.4f}")
        print(f"    Fold {i+1} Confusion Matrix: TP={fold_cm['TP']}, TN={fold_cm['TN']}, FP={fold_cm['FP']}, FN={fold_cm['FN']}")

    avg_accuracy = sum(all_fold_accuracies) / len(all_fold_accuracies)

    avg_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for cm in all_fold_confusion_matrices:
        avg_cm['TP'] += cm['TP']
        avg_cm['TN'] += cm['TN']
        avg_cm['FP'] += cm['FP']
        avg_cm['FN'] += cm['FN']
    
    print(f"\n--- Experiment Summary (Hidden={hidden_size}, LR={learning_rate}, Momentum={momentum}, Seed={init_seed}) ---")
    print(f"  Average Test Accuracy across {k_folds} folds: {avg_accuracy:.4f}")
    print(f"  Total Confusion Matrix across {k_folds} folds: TP={avg_cm['TP']}, TN={avg_cm['TN']}, FP={avg_cm['FP']}, FN={avg_cm['FN']}")

    return avg_accuracy, avg_cm, all_fold_accuracies

if __name__ == "__main__":
    hidden_sizes = [2, 5, 8]
    learning_rates = [0.01, 0.05, 0.1]
    momentums = [0.0, 0.5, 0.9]
    initialization_seeds = [10, 20, 30] 

    results = []
    for hs in hidden_sizes:
        for lr in learning_rates:
            for mom in momentums:
                for seed in initialization_seeds:
                    avg_acc, avg_cm, fold_accs = run_experiment(hs, lr, mom, seed)
                    results.append({
                        'hidden_size': hs,
                        'learning_rate': lr,
                        'momentum': mom,
                        'init_seed': seed,
                        'avg_accuracy': avg_acc,
                        'total_confusion_matrix': avg_cm
                    })

    print("\n--- All Experiment Results Summary ---")
    for res in results:
        print(f"Config: H={res['hidden_size']}, LR={res['learning_rate']:.3f}, Mom={res['momentum']:.1f}, Seed={res['init_seed']} -> Avg Acc: {res['avg_accuracy']:.4f}, CM: {res['total_confusion_matrix']}")

    best_result = max(results, key=lambda x: x['avg_accuracy'])
    print(f"\nBEST CONFIGURATION: ")
    print(f"  Hidden Size: {best_result['hidden_size']}")
    print(f"  Learning Rate: {best_result['learning_rate']:.3f}")
    print(f"  Momentum: {best_result['momentum']:.1f}")
    print(f"  Initialization Seed: {best_result['init_seed']}")
    print(f"  Average Accuracy: {best_result['avg_accuracy']:.4f}")
    print(f"  Total Confusion Matrix: {best_result['total_confusion_matrix']}")