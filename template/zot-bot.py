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


class Layer:
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


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(input_size, hidden_size, activation=lambda x: math.tanh(x))
        self.output_layer = Layer(hidden_size, output_size, activation=lambda x: math.tanh(x))

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


shared_state = [0.0, 0.0, 0.0]
network = NeuralNetwork(input_size=12, hidden_size=24, output_size=6)
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
    action_value, direction_value = output[:2]
    action_value, direction_value = (1 + math.tanh(action_value)) / 2, (1 + math.tanh(direction_value)) / 2

    memory_decay = (1 + math.tanh(output[2])) / 2
    shared_updates = output[3:]
    shared_state[:] = [
        math.tanh(memory_decay * s + (1 - memory_decay) * u)
        for s, u in zip(shared_state, shared_updates)
    ]

    action_type = ActionType.Move if action_value > 0.5 else ActionType.Attack
    direction = (
        Direction.North if direction_value < 0.25 else
        Direction.South if direction_value < 0.5 else
        Direction.East if direction_value < 0.75 else
        Direction.West
    )
    return Action.move(direction) if action_type == ActionType.Move else Action.attack(direction)