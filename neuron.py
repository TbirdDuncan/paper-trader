import math


def relu_neuron(inputs, weights, bias=0.0):
    """Compute one hidden neuron's output using the ReLU activation."""
    if len(inputs) != len(weights):
        raise ValueError("Each input must have exactly one weight.")

    weighted_sum = sum(value * weight for value, weight in zip(inputs, weights))
    return max(0.0, weighted_sum + bias)

def sigmoid_neuron(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))

    exp_z = math.exp(z)
    return exp_z / (1 + exp_z)


if __name__ == "__main__":
    # Example inputs: daily return as a decimal and volume relative to average.
    inputs = [0.02, 1.5]
    weights = [5.0, 0.4]
    output = relu_neuron(inputs, weights, bias=-0.5)
    sigmoid = sigmoid_neuron(output)
    print(f"Neuron output: {output:.2f}")
    print(f"Sigmoid output: {sigmoid:.2f}")
