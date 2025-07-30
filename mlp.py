import random
import math

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        # Weights from input layer to hidden layer
        # W1 is a list of lists (matrix) of size input_size x hidden_size
        self.W1 = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(input_size)]
        # b1 is a list (vector) of size 1 x hidden_size
        self.b1 = [[0.0 for _ in range(hidden_size)]]
        
        # Weights from hidden layer to output layer
        # W2 is a list of lists (matrix) of size hidden_size x output_size
        self.W2 = [[random.uniform(-0.01, 0.01) for _ in range(output_size)] for _ in range(hidden_size)]
        # b2 is a list (vector) of size 1 x output_size
        self.b2 = [[0.0 for _ in range(output_size)]]

    def sigmoid(self, x):
        """Sigmoid activation function. Handles single value or list of lists (matrix)."""
        if isinstance(x, list) and isinstance(x[0], list): # If it's a matrix
            return [[1 / (1 + math.exp(-val)) for val in row] for row in x]
        else: # If it's a single value (this case might not be hit if all internal calculations are matrix-based)
            return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function. x is assumed to be the output of sigmoid (a)."""
        if isinstance(x, list) and isinstance(x[0], list): # If it's a matrix
            return [[val * (1 - val) for val in row] for row in x]
        else: # If it's a single value
            return x * (1 - x)

    def dot_product(self, matrix1, matrix2):
        """
        Calculates the dot product (matrix multiplication) of two matrices.
        matrix1: list of lists (rows x cols1)
        matrix2: list of lists (cols1 x cols2)
        Returns: list of lists (rows x cols2)
        """
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
        """
        Transposes a matrix (list of lists).
        Returns: transposed list of lists.
        """
        if not matrix:
            return []
        rows = len(matrix)
        cols = len(matrix[0])
        transposed = [[0.0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
        return transposed
    
    def add_matrices(self, matrix1, matrix2):
        """Adds two matrices element-wise. Assumes same dimensions."""
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                result[r][c] = matrix1[r][c] + matrix2[r][c]
        return result
    
    def subtract_matrices(self, matrix1, matrix2):
        """Subtracts matrix2 from matrix1 element-wise. Assumes same dimensions."""
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                result[r][c] = matrix1[r][c] - matrix2[r][c]
        return result
        
    def multiply_scalar_matrix(self, scalar, matrix):
        """Multiplies a matrix by a scalar."""
        return [[scalar * val for val in row] for row in matrix]

    def element_wise_multiply(self, matrix1, matrix2):
        """Performs element-wise multiplication of two matrices. Assumes same dimensions."""
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                result[r][c] = matrix1[r][c] * matrix2[r][c]
        return result

    def sum_matrix_rows(self, matrix):
        """Sums elements across rows to get a 1xCols matrix (for biases)."""
        if not matrix: return []
        cols = len(matrix[0])
        result = [[0.0 for _ in range(cols)]]
        for row in matrix:
            for c in range(cols):
                result[0][c] += row[c]
        return result

    def forward(self, X_sample):
        """
        Performs the forward pass through the network for a single sample.
        X_sample: A single input sample (list). Will be converted to a 1xN matrix.
        Returns: The predicted output for the sample (list of lists, a 1x1 matrix).
        """
        # Convert X_sample to a 1xN matrix for matrix multiplication
        X_matrix = [X_sample]

        # Input to hidden layer
        self.z1 = self.add_matrices(self.dot_product(X_matrix, self.W1), self.b1)
        self.a1 = self.sigmoid(self.z1) # Activation of hidden layer

        # Hidden to output layer
        self.z2 = self.add_matrices(self.dot_product(self.a1, self.W2), self.b2)
        self.a2 = self.sigmoid(self.z2) # Activation of output layer
        return self.a2

    def backward(self, X_sample, y_true_sample, y_pred_sample, learning_rate):
        """
        Performs the backward pass (backpropagation) to update weights and biases
        for a single sample.
        X_sample: A single input sample (list).
        y_true_sample: True output value for the sample (list, e.g., [float_value]).
                       This will be converted to [[float_value]].
        y_pred_sample: Predicted output value for the sample (list of lists, e.g., [[float_value]]).
        learning_rate: Learning rate for updates
        """
        # Convert X_sample to a 1xN matrix for matrix multiplication
        X_matrix = [X_sample]
        
        # FIX: Ensure y_true_sample is a 1x1 matrix ([[float_value]]) for consistency with y_pred_sample.
        # y_true_sample comes in as a 1-element list, e.g., [153.0]. Convert it to [[153.0]].
        y_true_matrix = [[y_true_sample[0]]] 
        
        # y_pred_sample from forward is already a 1x1 matrix [[float_value]]
        y_pred_matrix = y_pred_sample 

        # Calculate error for output layer
        error_output = self.subtract_matrices(y_pred_matrix, y_true_matrix)
        delta2 = self.element_wise_multiply(error_output, self.sigmoid_derivative(y_pred_matrix))

        # Gradients for W2 and b2
        dW2 = self.dot_product(self.transpose(self.a1), delta2)
        db2 = self.sum_matrix_rows(delta2)

        # Calculate error for hidden layer
        delta1 = self.element_wise_multiply(self.dot_product(delta2, self.transpose(self.W2)), self.sigmoid_derivative(self.a1))

        # Gradients for W1 and b1
        dW1 = self.dot_product(self.transpose(X_matrix), delta1)
        db1 = self.sum_matrix_rows(delta1)

        # Update weights and biases
        self.W2 = self.subtract_matrices(self.W2, self.multiply_scalar_matrix(learning_rate, dW2))
        self.b2 = self.subtract_matrices(self.b2, self.multiply_scalar_matrix(learning_rate, db2))
        self.W1 = self.subtract_matrices(self.W1, self.multiply_scalar_matrix(learning_rate, dW1))
        self.b1 = self.subtract_matrices(self.b1, self.multiply_scalar_matrix(learning_rate, db1))

    def train(self, X_train, y_train, X_val, y_val, learning_rate, epochs):
        """
        Trains the MLP model using stochastic gradient descent (sample by sample).
        X_train, y_train: Training data (lists of lists)
        X_val, y_val: Validation data for monitoring (lists of lists)
        learning_rate: Learning rate
        epochs: Number of training epochs
        """
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Shuffle training data for each epoch (Stochastic Gradient Descent)
            combined_train = list(zip(X_train, y_train))
            random.shuffle(combined_train)
            X_shuffled_train, y_shuffled_train = zip(*combined_train)
            X_shuffled_train = list(X_shuffled_train)
            y_shuffled_train = list(y_shuffled_train)

            for i in range(len(X_shuffled_train)):
                x_sample = X_shuffled_train[i]
                y_true_sample = y_shuffled_train[i] # This is a 1-element list, e.g., [float_value]

                # Forward pass for a single sample
                y_pred_sample = self.forward(x_sample) # This is a 1x1 matrix, e.g., [[float_value]]
                
                # Backward pass to update weights for a single sample
                self.backward(x_sample, y_true_sample, y_pred_sample, learning_rate)

            # --- Calculate Loss after each epoch (on full training/validation sets) ---
            # Training Loss
            y_train_pred_full = []
            for sample_x in X_train:
                y_train_pred_full.append(self.forward(sample_x)[0][0]) # Get the single float prediction value

            # Flatten y_train to match y_train_pred_full for MSE calculation
            y_train_true_flat = [val[0] for val in y_train]
            
            # Calculate mean squared error using standard Python arithmetic
            train_loss_sum_sq_err = 0
            for i in range(len(y_train_true_flat)):
                train_loss_sum_sq_err += (y_train_true_flat[i] - y_train_pred_full[i])**2
            train_loss = train_loss_sum_sq_err / len(y_train_true_flat)
            train_losses.append(train_loss)

            # Validation Loss
            y_val_pred_full = []
            for sample_x_val in X_val:
                y_val_pred_full.append(self.forward(sample_x_val)[0][0]) # Get the single float prediction value
            
            # Flatten y_val for MSE calculation
            y_val_true_flat = [val[0] for val in y_val]

            val_loss_sum_sq_err = 0
            for i in range(len(y_val_true_flat)):
                val_loss_sum_sq_err += (y_val_true_flat[i] - y_val_pred_full[i])**2
            val_loss = val_loss_sum_sq_err / len(y_val_true_flat)
            val_losses.append(val_loss)

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return train_losses, val_losses
