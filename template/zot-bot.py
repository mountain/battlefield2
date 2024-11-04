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
    quantized = [int((f + 1) * 32767.5) for f in float_list]
    byte_data = struct.pack('>' + 'H' * len(quantized), *quantized)
    return base62_encode(byte_data)


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
        self.weights = [
            [
                [[random.uniform(-1, 1) for _ in range(kernel_size)] for _ in range(kernel_size)]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.biases = [random.uniform(-1, 1) for _ in range(out_channels)]
        self.output = None

    def get_parameters(self):
        return [w for kernel in self.weights for channel in kernel for row in channel for w in row] + self.biases

    def set_parameters(self, flat_parameters):
        weight_count = self.out_channels * self.in_channels * self.kernel_size * self.kernel_size
        weights_flat = flat_parameters[:weight_count]
        biases = flat_parameters[weight_count:weight_count + self.out_channels]

        self.weights = [
            [
                [
                    [weights_flat[i * self.kernel_size * self.kernel_size + j * self.kernel_size + k]
                     for k in range(self.kernel_size)]
                    for j in range(self.kernel_size)
                ]
                for i in range(self.in_channels)
            ]
            for _ in range(self.out_channels)
        ]

        if len(biases) != self.out_channels:
            raise ValueError(f"Biases length {len(biases)} does not match out_channels {self.out_channels}")
        self.biases = biases

    def forward(self, inputs):
        H, W, C = len(inputs), len(inputs[0]), len(inputs[0][0])
        out_H = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_W = (W - self.kernel_size + 2 * self.padding) // self.stride + 1

        padded_input = (
            [[[0 for _ in range(C)] for _ in range(W + 2 * self.padding)] for _ in range(H + 2 * self.padding)]
            if self.padding > 0
            else inputs
        )

        if self.padding > 0:
            for i in range(H):
                for j in range(W):
                    for k in range(C):
                        padded_input[i + self.padding][j + self.padding][k] = inputs[i][j][k]

        if self.output is None:
            self.output = [[[0 for _ in range(self.out_channels)] for _ in range(out_W)] for _ in range(out_H)]

        for out_c in range(self.out_channels):
            for i in range(out_H):
                for j in range(out_W):
                    self.output[i][j][out_c] = relu(
                        self.biases[out_c]
                        + sum(
                            self.weights[out_c][in_c][ki][kj] * padded_input[i * self.stride + ki][j * self.stride + kj][in_c]
                            for ki in range(self.kernel_size)
                            for kj in range(self.kernel_size)
                            for in_c in range(self.in_channels)
                        )
                    )

        return self.output



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

    def get_parameters(self):
        return [w for row in self.weights for w in row] + self.biases

    def set_parameters(self, flat_parameters):
        weight_count = len(self.weights) * len(self.weights[0])
        weights_flat = flat_parameters[:weight_count]
        biases = flat_parameters[weight_count:]

        self.weights = [
            weights_flat[i * len(self.weights[0]):(i + 1) * len(self.weights[0])]
            for i in range(len(self.weights))
        ]
        self.biases = biases


class GlobalCNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv2D(4, 4, kernel_size=3, stride=2, padding=1)
        self.fc = Linear(4 * 4 * 4, 4, relu)

    def get_parameters(self):
        return self.conv1.get_parameters() + self.conv2.get_parameters() + self.fc.get_parameters()

    def set_parameters(self, parameters):
        conv1_param_count = len(self.conv1.get_parameters())
        conv2_param_count = len(self.conv2.get_parameters())

        self.conv1.set_parameters(parameters[:conv1_param_count])
        self.conv2.set_parameters(parameters[conv1_param_count:conv1_param_count + conv2_param_count])
        self.fc.set_parameters(parameters[conv1_param_count + conv2_param_count:])

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
        return self.hidden_layer.get_parameters() + self.output_layer.get_parameters()

    def set_parameters(self, parameters):
        hidden_param_count = len(self.hidden_layer.get_parameters())
        self.hidden_layer.set_parameters(parameters[:hidden_param_count])
        self.output_layer.set_parameters(parameters[hidden_param_count:])


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


_cached_input_grid = [[[0] for _ in range(19)] for _ in range(19)]


def encode_global(state):
    global _cached_input_grid
    for x in range(19):
        for y in range(19):
            _cached_input_grid[x][y][0] = health_by_coords(state, Coords(x, y))

    return encoder.forward(_cached_input_grid)


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