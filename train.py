from copy import deepcopy
from math import exp, isfinite, log1p, sqrt
from random import Random
from statistics import mean, pstdev


import yfinance as yf


IBM_FEATURE_NAMES = (
    "open", "high", "low", "close", "adjusted_close", "volume",
    "return_1d", "return_5d", "volatility_20d", "relative_volume",
)


def get_ibm_history():
    """Return daily (date, open, high, low, close, adjusted_close, volume) tuples."""
    history = yf.Ticker("IBM").history(
        period="max", interval="1d", auto_adjust=False, actions=False
    )
    if history.empty:
        raise RuntimeError("Yahoo Finance returned no IBM daily history.")

    columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    rows = history.sort_index()[columns]
    return [
        (timestamp.date(), open_price, high, low, close, adjusted_close, volume)
        for timestamp, open_price, high, low, close, adjusted_close, volume in
        rows.itertuples(name=None)
    ]


def ibm_features(rows):
    """Build features in IBM_FEATURE_NAMES order from 21 or more oldest-first rows."""
    if len(rows) < 21:
        raise ValueError("At least 21 daily rows are required.")

    recent_rows = rows[-21:]
    closes = [row[4] for row in recent_rows]
    daily_returns = [today / yesterday - 1 for yesterday, today in zip(closes, closes[1:])]
    average_volume = mean(row[6] for row in recent_rows[:-1])
    if average_volume <= 0:
        raise ValueError("The prior 20 days must have positive average volume.")

    return (
        *recent_rows[-1][1:],
        daily_returns[-1],
        closes[-1] / closes[-6] - 1,
        pstdev(daily_returns),
        recent_rows[-1][6] / average_volume,
    )


def relu_neuron(inputs, weights, bias=0.0):
    """Compute one hidden neuron's output using the ReLU activation."""
    if len(inputs) != len(weights):
        raise ValueError("Each input must have exactly one weight.")

    weighted_sum = sum(value * weight for value, weight in zip(inputs, weights))
    return max(0.0, weighted_sum + bias)

def sigmoid_neuron(z):
    if z >= 0:
        return 1 / (1 + exp(-z))

    exp_z = exp(z)
    return exp_z / (1 + exp_z)


def relu_layer(inputs, weight_rows, biases):
    """Return one activation per neuron for a single set of inputs."""
    if len(weight_rows) != len(biases):
        raise ValueError("Each neuron must have exactly one bias.")

    return [
        relu_neuron(inputs, weights, bias)
        for weights, bias in zip(weight_rows, biases)
    ]


def examples(rows, start, stop):
    # Both the feature date and its next-day label belong to this split.
    for day in range(max(20, start), stop - 1):
        window = rows[day - 20:day + 1]
        if any(row[4] <= 0 or not isfinite(row[4]) for row in rows[day - 20:day + 2]):
            continue
        if sum(row[6] for row in window[:-1]) <= 0:
            continue
        features = ibm_features(window)
        if all(isfinite(value) for value in features):
            yield features, int(rows[day + 1][4] > rows[day][4])


def forward(inputs, model):
    weights, biases, output_weights, output_bias = model
    hidden = relu_layer(inputs, weights, biases)
    score = sum(value * weight for value, weight in zip(hidden, output_weights))
    return hidden, score + output_bias[0]


def evaluate(data, model):
    loss = correct = 0
    for inputs, target in data:
        _, score = forward(inputs, model)
        # Binary cross-entropy calculated directly from the output score.
        loss += max(score, 0) - score * target + log1p(exp(-abs(score)))
        correct += (score >= 0) == target
    return loss / len(data), correct / len(data)


def train_ibm(rows, epochs=50, learning_rate=0.001, patience=5):
    train_end = len(rows) * 7 // 10
    validation_end = len(rows) * 85 // 100
    training = list(examples(rows, 0, train_end))
    validation = list(examples(rows, train_end, validation_end))
    test = list(examples(rows, validation_end, len(rows)))
    if not all((training, validation, test)):
        raise ValueError("Each split needs usable IBM examples.")

    columns = list(zip(*(inputs for inputs, _ in training)))
    centers = [mean(column) for column in columns]
    scales = [pstdev(column) or 1.0 for column in columns]

    def scale(data):
        return [
            ([(value - center) / spread
              for value, center, spread in zip(inputs, centers, scales)], target)
            for inputs, target in data
        ]

    training, validation, test = map(scale, (training, validation, test))
    rng = Random(42)
    feature_count = len(IBM_FEATURE_NAMES)
    hidden_count = 3
    weights = [[rng.gauss(0, sqrt(2 / feature_count)) for _ in range(feature_count)]
               for _ in range(hidden_count)]
    biases = [0.0] * hidden_count
    output_weights = [rng.gauss(0, sqrt(1 / hidden_count)) for _ in biases]
    output_bias = [0.0]
    model = (weights, biases, output_weights, output_bias)
    best_loss, _ = evaluate(validation, model)
    best_model = deepcopy(model)
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        rng.shuffle(training)
        for inputs, target in training:
            hidden, score = forward(inputs, model)
            error = sigmoid_neuron(score) - target
            for i, activation in enumerate(hidden):
                if activation > 0:
                    step = learning_rate * error * output_weights[i]
                    for j, value in enumerate(inputs):
                        weights[i][j] -= step * value
                    biases[i] -= step
                output_weights[i] -= learning_rate * error * activation
            output_bias[0] -= learning_rate * error

        loss, _ = evaluate(validation, model)
        print(f"Epoch {epoch}: validation loss {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_model = deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    loss, accuracy = evaluate(test, best_model)
    baseline = mean(target for _, target in test)
    print(f"Test loss: {loss:.4f} | Accuracy: {accuracy:.1%} | Always-up accuracy: {baseline:.1%}")
    return best_model, centers, scales


if __name__ == "__main__":
    train_ibm(get_ibm_history())
