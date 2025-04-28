import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def generate_dataset(num_samples=20):
    """Generate dataset with 3 features and 1 target"""
    X = np.random.uniform(-1, 1, (num_samples, 3))
    y = 0.5*X[:, 0] + 0.8*X[:, 1]**2 - 1.2*X[:, 2] + np.random.normal(0, 0.1, num_samples)
    return X, y.reshape(-1, 1)

# Activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))

# Loss functions
def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred)**2)
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize network with random weights and biases"""
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias1 = np.random.randn(1, hidden_size) * 0.1
        self.bias2 = np.random.randn(1, output_size) * 0.1
        
        # For tracking gradient magnitudes
        self.gradient_history = {
            'dW1': [], 'db1': [], 'dW2': [], 'db2': [],
            'gradient_loss': []  # To track gradient loss
        }
    
    def forward(self, X):
        """Forward pass through the network"""
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        return self.output_input, self.hidden_output
    
    def backward(self, X, y, output, hidden_output, learning_rate=0.01):
        """Backward pass with gradient tracking"""
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        output_error = output - y  # ∂L/∂output
        dW2 = np.dot(hidden_output.T, output_error) / m  # ∂L/∂W2
        db2 = np.sum(output_error, axis=0, keepdims=True) / m  # ∂L/∂b2
        
        # Hidden layer gradients
        hidden_error = np.dot(output_error, self.weights2.T) * sigmoid_derivative(self.hidden_input)
        dW1 = np.dot(X.T, hidden_error) / m  # ∂L/∂W1
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m  # ∂L/∂b1
        
        # Calculate gradient loss (magnitude of all gradients)
        gradient_magnitude = np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2))
        self.gradient_history['gradient_loss'].append(gradient_magnitude)
        
        # Store individual gradient magnitudes
        self.gradient_history['dW1'].append(np.mean(np.abs(dW1)))
        self.gradient_history['db1'].append(np.mean(np.abs(db1)))
        self.gradient_history['dW2'].append(np.mean(np.abs(dW2)))
        self.gradient_history['db2'].append(np.mean(np.abs(db2)))
        
        # Update parameters
        self.weights1 -= learning_rate * dW1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dW2
        self.bias2 -= learning_rate * db2
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=100):
        """Train the network with gradient tracking"""
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output, hidden_output = self.forward(X)
            
            # Calculate and store loss
            loss = mse_loss(y, output)
            losses.append(loss)
            
            # Backward pass and parameter update
            self.backward(X, y, output, hidden_output, learning_rate)
            
            if verbose and epoch % verbose == 0:
                acc = r_squared(y, output)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, R-squared: {acc:.4f}")
        
        return losses
    
    def plot_gradient_history(self):
        """Visualize gradient magnitudes during training"""
        plt.figure(figsize=(12, 5))
        
        # Plot gradient loss (overall magnitude)
        plt.subplot(1, 2, 1)
        plt.plot(self.gradient_history['gradient_loss'])
        plt.title("Gradient Loss (Overall Magnitude)")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Magnitude")
        
        # Plot individual gradient components
        plt.subplot(1, 2, 2)
        for param, values in self.gradient_history.items():
            if param != 'gradient_loss':
                plt.plot(values, label=param)
        plt.title("Individual Gradient Components")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Magnitude")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    X, y = generate_dataset(20)
    nn = NeuralNetwork(3, 3, 1)
    
    print("Initial R-squared:", r_squared(y, nn.forward(X)[0]))
    losses = nn.train(X, y, epochs=5000, learning_rate=0.1)
    print("Final R-squared:", r_squared(y, nn.forward(X)[0]))
    
    # Plot training loss
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    # Plot gradient history
    nn.plot_gradient_history()
    
    # Show sample predictions
    predictions, _ = nn.forward(X)
    print("\nSample predictions:")
    for i in range(5):
        print(f"True: {y[i][0]:.4f}, Predicted: {predictions[i][0]:.4f}")


"""

Key Explanations:
Why Non-Zero Biases Matter:
Breaks Symmetry: Starting with zero biases can lead to neurons learning the same features during initial training

Faster Learning: Non-zero initialization can help the network start learning patterns immediately

Avoids Dead Neurons: Helps prevent neurons from getting "stuck" in zero activation states

1. Partial Derivatives Calculation:
Output Layer Gradients:

output_error = output - y represents ∂L/∂output (partial derivative of loss w.r.t. output)

dW2 = np.dot(hidden_output.T, output_error) / m calculates ∂L/∂W2 using chain rule:

∂L/∂W2 = ∂L/∂output * ∂output/∂W2

Where ∂output/∂W2 = hidden_output (from forward pass equation)

Hidden Layer Gradients:

hidden_error = np.dot(output_error, self.weights2.T) calculates ∂L/∂hidden_output

Multiplying by sigmoid_derivative gives ∂L/∂hidden_input (applying chain rule again)

dW1 = np.dot(X.T, hidden_error) / m calculates ∂L/∂W1:

∂L/∂W1 = ∂L/∂hidden_input * ∂hidden_input/∂W1

Where ∂hidden_input/∂W1 = X (input features)

2. Chain Rule Applications:
The chain rule is applied at multiple points:

Output Layer Weights (W2):

To find how loss changes with W2, we chain:

How loss changes with output (∂L/∂output)

How output changes with W2 (∂output/∂W2)

Hidden Layer Weights (W1):

To find how loss changes with W1, we chain:

How loss changes with hidden output (∂L/∂hidden_output)

How hidden output changes with hidden input (sigmoid derivative)

How hidden input changes with W1 (∂hidden_input/∂W1)

3. Weight Update Mechanism:

self.weights1 -= learning_rate * dW1
self.bias1 -= learning_rate * db1
self.weights2 -= learning_rate * dW2 
self.bias2 -= learning_rate * db2


Gradient Tracking:

Added gradient_history dictionary to store gradient magnitudes

Tracks four components:

dW1: Gradients for input-to-hidden weights

db1: Gradients for hidden layer biases

dW2: Gradients for hidden-to-output weights

db2: Gradients for output layer biases

gradient_loss: Overall gradient magnitude (Euclidean norm)

Gradient Calculation:

In backward() method, added calculation of gradient magnitude:

python
gradient_magnitude = np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2))
Visualization Method:

Added plot_gradient_history() method that shows:

Left plot: Overall gradient magnitude (shows how gradients change during training)

Right plot: Individual gradient components (helps identify which parameters are changing most)

Interpretation:

The gradient plots help diagnose training problems:

If gradients vanish (approach 0), learning may stall

If gradients explode (become very large), learning may become unstable

Healthy training shows gradual decrease in gradient magnitudes

What to Look For in the Plots:
Gradient Loss Plot:

Shows whether gradients are vanishing (approaching 0) or exploding (growing very large)

Ideally should show a gradual decrease as the network learns

Component Plots:

Helps identify if specific layers have problematic gradients

All components should show similar patterns for balanced learning

"""