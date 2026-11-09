from neuron import relu_neuron


def relu_layer(inputs, weight_rows, biases):
    """Return one activation per neuron for a single set of inputs."""
    if len(weight_rows) != len(biases):
        raise ValueError("Each neuron must have exactly one bias.")

    return [
        relu_neuron(inputs, weights, bias)
        for weights, bias in zip(weight_rows, biases)
    ]


if __name__ == "__main__":
    from ibm_data import IBM_FEATURE_NAMES, get_ibm_history, ibm_features

    inputs = ibm_features(get_ibm_history())
    # Open, high, low, close, adjusted close, volume, then the four derived features.
    weight_rows = [
        [0.01, 0.0, 0.0, -0.01, 0.0, 1e-8, 5.0, 1.0, -2.0, 0.4],
        [0.0, 0.01, -0.01, 0.0, 0.001, -1e-8, -2.0, 2.0, 1.0, 0.1],
        [-0.01, 0.0, 0.0, 0.01, -0.001, 2e-8, 3.0, -1.0, 2.0, -0.2],
    ]
    biases = [-0.5, 0.1, 0.0]
    activations = relu_layer(inputs, weight_rows, biases)
    print("Inputs:", dict(zip(IBM_FEATURE_NAMES, inputs)))
    print("Hidden activations:", ", ".join(f"{value:.2f}" for value in activations))
