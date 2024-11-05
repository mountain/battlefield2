import random
import math
import hashlib
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
    encoded_data = base62_encode(byte_data)
    return encoded_data


def decompress_floats(encoded_data):
    byte_data = base62_decode(encoded_data)
    quantized = struct.unpack('>' + 'H' * (len(byte_data) // 2), byte_data)
    return [(q / 32767.5) - 1 for q in quantized]


def fingerprint(float_list):
    byte_data = struct.pack('>' + 'f' * len(float_list), *float_list)
    hash_obj = hashlib.sha256(byte_data)
    fingerprint = hash_obj.hexdigest()
    return fingerprint


def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x


def softmax(values):
    exp_values = [math.exp(v) for v in values]
    sum_exp_values = sum(exp_values)
    probabilities = [v / sum_exp_values for v in exp_values]
    return probabilities


def argmax(values):
    max_index = 0
    max_value = values[0]
    for i in range(1, len(values)):
        if values[i] > max_value:
            max_value = values[i]
            max_index = i
    return max_index


class Linear:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, inputs):
        return [
            self.activation(sum(w * inp for w, inp in zip(weights, inputs)) + bias)
            for weights, bias in zip(self.weights, self.biases)
        ]

    def get_parameters(self):
        return [w for row in self.weights for w in row] + self.biases

    def set_parameters(self, parameters):
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weights[i][j] = parameters[i * self.input_size + j]
        self.biases = parameters[self.input_size * self.output_size:]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_layer = Linear(input_size, hidden_size, activation=leaky_relu)
        self.hidden_layer = Linear(hidden_size, hidden_size, activation=leaky_relu)
        self.output_layer = Linear(hidden_size, output_size, activation=leaky_relu)

    def forward(self, inputs):
        return self.output_layer.forward(self.hidden_layer.forward(self.input_layer.forward(inputs)))

    def get_parameters(self):
        return self.input_layer.get_parameters() + self.hidden_layer.get_parameters() + self.output_layer.get_parameters()

    def set_parameters(self, parameters):
        input_length = len(self.input_layer.get_parameters())
        hidden_length = len(self.hidden_layer.get_parameters())
        self.input_layer.set_parameters(parameters[:input_length])
        self.hidden_layer.set_parameters(parameters[input_length:input_length + hidden_length])
        self.output_layer.set_parameters(parameters[input_length + hidden_length:])


shared_state = [0.0, 0.0, 0.0, 0.0, 0.0]
network = NeuralNetwork(input_size=12, hidden_size=24, output_size=12)
network.set_parameters(decompress_floats(
    "%s"
))


def health_by_coords(state, coord):
    obj = state.obj_by_coords(coord)
    if obj and obj.team is not None:
        return obj.health / 5 if obj.team == state.our_team else -obj.health / 5
    return 0


def robot(state, unit):
    global shared_state

    x = (float(unit.coords.x) - 10) / 10
    y = (float(unit.coords.y) - 10) / 10
    r = math.sqrt(x * x + y * y)
    inputs = [x, y, r] + [
        health_by_coords(state, coord) for coord in unit.coords.coords_around()
    ] + shared_state

    output = network.forward(inputs)
    action_value, direction_value, decay_coeff, shared_updates = output[:2], output[2:6], output[7], output[7:]

    total_units = len(state.ids_by_team(state.our_team)) + len(state.ids_by_team(state.other_team))
    memory_decay = (1 + math.tanh(decay_coeff)) * (1 + math.tanh(total_units - 20)) / 4
    shared_state[:] = [
        math.tanh((1 - memory_decay) * s + memory_decay * u)
        for s, u in zip(shared_state, shared_updates)
    ]

    action_type = [ActionType.Move, ActionType.Attack][argmax(softmax(action_value))]
    direction = [
        Direction.North,
        Direction.South,
        Direction.East,
        Direction.West
    ][argmax(softmax(direction_value))]

    return Action.move(direction) if action_type == ActionType.Move else Action.attack(direction)