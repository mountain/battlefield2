import random
import math
import struct


BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def base62_encode(data: bytes) -> str:
    num = int.from_bytes(data, "big")
    if num == 0:
        return BASE62_ALPHABET[0]
    result = []
    while num > 0:
        num, rem = divmod(num, 62)
        result.append(BASE62_ALPHABET[rem])
    return ''.join(reversed(result))


def base62_decode(encoded: str) -> bytes:
    num = 0
    for char in encoded:
        num = num * 62 + BASE62_ALPHABET.index(char)
    byte_length = (num.bit_length() + 7) // 8
    return num.to_bytes(byte_length, "big")


def compress_floats(float_list):
    quantized = [int((f + 1) * 32767.5) for f in float_list]  # 转换为 0-65535 (16位)
    byte_data = struct.pack('>' + 'H' * len(quantized), *quantized)
    encoded_data = base62_encode(byte_data)
    return encoded_data


def decompress_floats(encoded_data):
    byte_data = base62_decode(encoded_data)
    quantized = struct.unpack('>' + 'H' * (len(byte_data) // 2), byte_data)
    return [(q / 32767.5) - 1 for q in quantized]


def relu(x):
    return max(0, x)


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = [[[random.uniform(-1, 1) for _ in range(kernel_size)] for _ in range(kernel_size)] for _ in
                        range(out_channels)]
        self.biases = [random.uniform(-1, 1) for _ in range(out_channels)]

    def get_weights(self):
        return [w for kernel in self.weights for row in kernel for w in row] + self.biases

    def set_weights(self, weights, biases):
        self.weights = [
            [
                [weights[i * self.kernel_size + j] for j in range(self.kernel_size)]
                for i in range(self.kernel_size)
            ]
            for _ in range(self.out_channels)
        ]
        self.biases = biases

    def forward(self, inputs):
        H, W, C = len(inputs), len(inputs[0]), len(inputs[0][0])
        out_H = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_W = (W - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = [[[0 for _ in range(self.out_channels)] for _ in range(out_W)] for _ in range(out_H)]

        padded_input = inputs
        if self.padding > 0:
            padded_input = [[[0 for _ in range(C)] for _ in range(W + 2 * self.padding)] for _ in
                            range(H + 2 * self.padding)]
            for i in range(H):
                for j in range(W):
                    for k in range(C):
                        padded_input[i + self.padding][j + self.padding][k] = inputs[i][j][k]

        for out_c in range(self.out_channels):
            for i in range(out_H):
                for j in range(out_W):
                    sum_value = self.biases[out_c]
                    for ki in range(self.kernel_size):
                        for kj in range(self.kernel_size):
                            for in_c in range(self.in_channels):
                                sum_value += (
                                        self.weights[out_c][ki][kj] *
                                        padded_input[i * self.stride + ki][j * self.stride + kj][in_c]
                                )
                    output[i][j][out_c] = relu(sum_value)
        return output


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs):
        H, W, C = len(inputs), len(inputs[0]), len(inputs[0][0])
        out_H = H // self.stride
        out_W = W // self.stride
        output = [[[0 for _ in range(C)] for _ in range(out_W)] for _ in range(out_H)]

        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    max_value = -float('inf')
                    for ki in range(self.kernel_size):
                        for kj in range(self.kernel_size):
                            max_value = max(max_value, inputs[i * self.stride + ki][j * self.stride + kj][c])
                    output[i][j][c] = max_value
        return output


class Linear:
    def __init__(self, input_size, output_size, activation):
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1, 1) for _ in range(output_size)]
        self.activation = activation

    def forward(self, inputs):
        return [
            self.activation(sum(w * inp for w, inp in zip(weights, inputs)) + bias)
            for weights, bias in zip(self.weights, self.biases)
        ]

    def get_weights(self):
        return [w for row in self.weights for w in row] + self.biases

    def set_weights(self, weights, biases):
        if len(weights) != len(self.weights) or len(biases) != len(self.biases):
            raise ValueError("Weights or biases dimensions do not match layer dimensions")
        self.weights = weights
        self.biases = biases


class GlobalCNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv2D(4, 4, kernel_size=3, stride=2, padding=1)
        self.fc = Linear(4 * 4 * 4, 4, relu)

    def get_parameters(self):
        return self.conv1.get_weights() + self.conv2.get_weights() + self.fc.get_weights()

    def set_parameters(self, weights):
        conv1_weights = weights[:40]
        self.conv1.set_weights(conv1_weights[:36], conv1_weights[36:40])

        conv2_weights = weights[40:148]
        self.conv2.set_weights(conv2_weights[:144], conv2_weights[144:148])

        fc_weights = weights[148:]
        self.fc.set_weights(fc_weights[:64 * 4], fc_weights[64 * 4:])

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = [item for sublist in x for subsublist in sublist for item in subsublist]
        return self.fc.forward(x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Linear(input_size, hidden_size, activation=lambda x: math.tanh(x))
        self.output_layer = Linear(hidden_size, output_size, activation=lambda x: math.tanh(x))

    def forward(self, inputs):
        return self.output_layer.forward(self.hidden_layer.forward(inputs))

    def get_parameters(self):
        return self.hidden_layer.get_weights() + self.output_layer.get_weights()

    def set_parameters(self, parameters):
        hidden_weight_size = len(self.hidden_layer.weights) * len(self.hidden_layer.weights[0])
        hidden_bias_size = len(self.hidden_layer.biases)

        hidden_weights = [parameters[i:i + len(self.hidden_layer.weights[0])] for i in
                          range(0, hidden_weight_size, len(self.hidden_layer.weights[0]))]
        hidden_biases = parameters[hidden_weight_size:hidden_weight_size + hidden_bias_size]

        self.hidden_layer.set_weights(hidden_weights, hidden_biases)

        output_weight_size = len(self.output_layer.weights) * len(self.output_layer.weights[0])
        output_bias_size = len(self.output_layer.biases)

        output_weights = [parameters[i:i + len(self.output_layer.weights[0])] for i in
                          range(hidden_weight_size + hidden_bias_size,
                                hidden_weight_size + hidden_bias_size + output_weight_size,
                                len(self.output_layer.weights[0]))]
        output_biases = parameters[-output_bias_size:]

        self.output_layer.set_weights(output_weights, output_biases)


policy = NeuralNetwork(input_size=12, hidden_size=24, output_size=2)
policy.set_parameters(decompress_floats(
    "%s"
))

encoder = GlobalCNN()
encoder.set_parameters(decompress_floats(
    "%s"
))


def health_by_coords(state, coord):
    obj = state.obj_by_coords(coord)
    if obj and obj.team is not None:
        return obj.health / 5 if obj.team == state.our_team else -obj.health / 5
    return 0


def encode_global(state):
    input_grid = [[[health_by_coords(state, Coords(x, y))] for y in range(19)] for x in range(19)]
    return encoder.forward(input_grid)


def robot(state, unit):
    inputs = [float(unit.coords.x) / 18, float(unit.coords.y) / 18] + [
        health_by_coords(state, coord) for coord in unit.coords.coords_around()
    ] + [
        math.tanh(x) for x in encode_global(state)
    ]

    output = policy.forward(inputs)
    action_value, direction_value = output[:2]
    action_value, direction_value = (1 + action_value) / 2, (1 + direction_value) / 2

    action_type = ActionType.Move if action_value > 0.5 else ActionType.Attack
    direction = (
        Direction.North if direction_value < 0.25 else
        Direction.South if direction_value < 0.5 else
        Direction.East if direction_value < 0.75 else
        Direction.West
    )

    return Action.move(direction) if action_type == ActionType.Move else Action.attack(direction)