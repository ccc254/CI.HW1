# C:\CI.HW1\mlp_model\mlp.py

import random
import math

class MLP:
    def __init__(self, input_size, hidden_size, output_size, momentum=0.0):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.momentum = momentum # New: Momentum parameter

        # Weights from input layer to hidden layer
        self.W1 = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [[0.0 for _ in range(hidden_size)]]
        
        # Weights from hidden layer to output layer
        self.W2 = [[random.uniform(-0.01, 0.01) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [[0.0 for _ in range(output_size)]]

        # New: Store previous weight/bias changes for momentum
        self.dW1_prev = [[0.0 for _ in range(hidden_size)] for _ in range(input_size)]
        self.db1_prev = [[0.0 for _ in range(hidden_size)]]
        self.dW2_prev = [[0.0 for _ in range(output_size)] for _ in range(hidden_size)]
        self.db2_prev = [[0.0 for _ in range(output_size)]]

    def sigmoid(self, x):
        if isinstance(x, list) and isinstance(x[0], list): # If it's a matrix
            return [[1 / (1 + math.exp(-val)) for val in row] for row in x]
        else: 
            return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        if isinstance(x, list) and isinstance(x[0], list): # If it's a matrix
            return [[val * (1 - val) for val in row] for row in x]
        else:
            return x * (1 - x)

    def dot_product(self, matrix1, matrix2):
        rows1 = len(matrix1)
        cols1 = len(matrix1[0])
        rows2 = len(matrix2)
        cols2 = len(matrix2[0])
        if cols1 != rows2:
            raise ValueError(f"Matrices dimensions mismatch for dot product: {rows1}x{cols1} and {rows2}x{cols2}")
        result = [[0.0 for _ in range(cols2)] for _ in range(rows1)]
        for i in range(rows1):
            for j in range(cols2):
                for k in range(cols1):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result

    def transpose(self, matrix):
        if not matrix: return []
        rows = len(matrix)
        cols = len(matrix[0])
        transposed = [[0.0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
        return transposed
    
    def add_matrices(self, matrix1, matrix2):
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                result[r][c] = matrix1[r][c] + matrix2[r][c]
        return result
    
    def subtract_matrices(self, matrix1, matrix2):
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                result[r][c] = matrix1[r][c] - matrix2[r][c]
        return result
        
    def multiply_scalar_matrix(self, scalar, matrix):
        return [[scalar * val for val in row] for row in matrix]

    def element_wise_multiply(self, matrix1, matrix2):
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                result[r][c] = matrix1[r][c] * matrix2[r][c]
        return result

    def sum_matrix_rows(self, matrix):
        if not matrix: return []
        cols = len(matrix[0])
        result = [[0.0 for _ in range(cols)]]
        for row in matrix:
            for c in range(cols):
                result[0][c] += row[c]
        return result

    def forward(self, X_sample):
        X_matrix = [X_sample]
        self.z1 = self.add_matrices(self.dot_product(X_matrix, self.W1), self.b1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.add_matrices(self.dot_product(self.a1, self.W2), self.b2)
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X_sample, y_true_sample, y_pred_sample, learning_rate):
        X_matrix = [X_sample]
        y_true_matrix = [[y_true_sample[0]]] 
        y_pred_matrix = y_pred_sample 

        error_output = self.subtract_matrices(y_pred_matrix, y_true_matrix)
        delta2 = self.element_wise_multiply(error_output, self.sigmoid_derivative(y_pred_matrix))

        dW2 = self.dot_product(self.transpose(self.a1), delta2)
        db2 = self.sum_matrix_rows(delta2)

        delta1 = self.element_wise_multiply(self.dot_product(delta2, self.transpose(self.W2)), self.sigmoid_derivative(self.a1))

        dW1 = self.dot_product(self.transpose(X_matrix), delta1)
        db1 = self.sum_matrix_rows(delta1)

        dW2_current_change = self.multiply_scalar_matrix(learning_rate, dW2)
        db2_current_change = self.multiply_scalar_matrix(learning_rate, db2)
        dW1_current_change = self.multiply_scalar_matrix(learning_rate, dW1)
        db1_current_change = self.multiply_scalar_matrix(learning_rate, db1)

        dW2_momentum_term = self.multiply_scalar_matrix(self.momentum, self.dW2_prev)
        db2_momentum_term = self.multiply_scalar_matrix(self.momentum, self.db2_prev)
        dW1_momentum_term = self.multiply_scalar_matrix(self.momentum, self.dW1_prev)
        db1_momentum_term = self.multiply_scalar_matrix(self.momentum, self.db1_prev)

        dW2_update = self.add_matrices(dW2_current_change, dW2_momentum_term)
        db2_update = self.add_matrices(db2_current_change, db2_momentum_term)
        dW1_update = self.add_matrices(dW1_current_change, dW1_momentum_term)
        db1_update = self.add_matrices(db1_current_change, db1_momentum_term)

        self.W2 = self.subtract_matrices(self.W2, dW2_update)
        self.b2 = self.subtract_matrices(self.b2, db2_update)
        self.W1 = self.subtract_matrices(self.W1, dW1_update)
        self.b1 = self.subtract_matrices(self.b1, db1_update)

        self.dW2_prev = dW2_update
        self.db2_prev = db2_update
        self.dW1_prev = dW1_update
        self.db1_prev = db1_update

    def train(self, X_train, y_train, X_val, y_val, learning_rate, epochs):
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            combined_train = list(zip(X_train, y_train))
            random.shuffle(combined_train)
            X_shuffled_train, y_shuffled_train = zip(*combined_train)
            X_shuffled_train = list(X_shuffled_train)
            y_shuffled_train = list(y_shuffled_train)

            for i in range(len(X_shuffled_train)):
                x_sample = X_shuffled_train[i]
                y_true_sample = y_shuffled_train[i] 
                y_pred_sample = self.forward(x_sample)
                self.backward(x_sample, y_true_sample, y_pred_sample, learning_rate)

            y_train_pred_probs = []
            for sample_x in X_train:
                y_train_pred_probs.append(self.forward(sample_x)[0][0]) 

            y_train_true_flat = [val[0] for val in y_train] 
            y_train_pred_classes = [1 if p >= 0.5 else 0 for p in y_train_pred_probs] 

            train_accuracy = sum(1 for i in range(len(y_train_true_flat)) if y_train_true_flat[i] == y_train_pred_classes[i]) / len(y_train_true_flat)
            train_accuracies.append(train_accuracy)

            y_val_pred_probs = []
            for sample_x_val in X_val:
                y_val_pred_probs.append(self.forward(sample_x_val)[0][0])
            
            y_val_true_flat = [val[0] for val in y_val]
            y_val_pred_classes = [1 if p >= 0.5 else 0 for p in y_val_pred_probs]

            val_accuracy = sum(1 for i in range(len(y_val_true_flat)) if y_val_true_flat[i] == y_val_pred_classes[i]) / len(y_val_true_flat)
            val_accuracies.append(val_accuracy)

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return train_accuracies, val_accuracies