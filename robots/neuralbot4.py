import random
import math
import struct
from typing import List, Generator


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


def compress_floats(float_list: List[float]) -> str:
    quantized = [int((f + 1) * 32767.5) for f in float_list]
    byte_data = struct.pack('>' + 'H' * len(quantized), *quantized)
    encoded_data = base62_encode(byte_data)
    return encoded_data


def decompress_floats(encoded_data: str) -> List[float]:
    encoded_data = encoded_data.strip()
    byte_data = base62_decode(encoded_data)
    quantized = struct.unpack('>' + 'H' * (len(byte_data) // 2), byte_data)
    return [(q / 32767.5) - 1 for q in quantized]


def identity(x: float) -> float:
    return x


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    return x if x > 0 else alpha * x


def sigmoid(x: float) -> float:
    return (1 + math.tanh(x)) / 2


def softmax(values: List[float]) -> List[float]:
    exp_values = [math.exp(v) for v in values]
    sum_exp_values = sum(exp_values) + 0.0001
    probabilities = [v / sum_exp_values for v in exp_values]
    return probabilities


def argmax(values: List[float]) -> int:
    max_index = 0
    max_value = values[0]
    for i in range(1, len(values)):
        if values[i] > max_value:
            max_value = values[i]
            max_index = i
    return max_index


class Linear:
    def __init__(self, input_size: int, output_size: int, activation) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, inputs: List[float]) -> List[float]:
        return [
            self.activation(sum(w * inp for w, inp in zip(weights, inputs)) + bias)
            for weights, bias in zip(self.weights, self.biases)
        ]

    def get_parameters(self) -> List[float]:
        return [w for row in self.weights for w in row] + self.biases

    def set_parameters(self, parameters: List[float]) -> None:
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weights[i][j] = parameters[i * self.input_size + j]
        self.biases = parameters[self.input_size * self.output_size:]


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.input_layer = Linear(input_size, hidden_size, activation=leaky_relu)
        self.hidden_layer = Linear(hidden_size, hidden_size, activation=leaky_relu)
        self.output_layer = Linear(hidden_size, output_size, activation=identity)

    def forward(self, inputs: List[float]) -> List[float]:
        return self.output_layer.forward(self.hidden_layer.forward(self.input_layer.forward(inputs)))

    def get_parameters(self) -> List[float]:
        return self.input_layer.get_parameters() + self.hidden_layer.get_parameters() + self.output_layer.get_parameters()

    def set_parameters(self, parameters: List[float]) -> None:
        input_length = len(self.input_layer.get_parameters())
        hidden_length = len(self.hidden_layer.get_parameters())
        self.input_layer.set_parameters(parameters[:input_length])
        self.hidden_layer.set_parameters(parameters[input_length:input_length + hidden_length])
        self.output_layer.set_parameters(parameters[input_length + hidden_length:])


shared_state: List[float] = [0.0, 0.0, 0.0, 0.0]
network = NeuralNetwork(input_size=18, hidden_size=36, output_size=9)
network.set_parameters(decompress_floats("""
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
"""))


def health_by_coords(state, coord) -> float:
    obj = state.obj_by_coords(coord)
    if obj and obj.team is not None:
        return obj.health / 5 if obj.team == state.our_team else -obj.health / 5
    return 0


def enterable(state, unit, direction) -> bool:
    x, y = unit.coords.x, unit.coords.y
    if direction == Direction.North:
        y -= 1
    elif direction == Direction.South:
        y += 1
    elif direction == Direction.East:
        x += 1
    elif direction == Direction.West:
        x -= 1

    board_limits = (1, 17)
    corner_limits = (6, 30)
    inaccessible = (
            x < board_limits[0] or x > board_limits[1] or
            y < board_limits[0] or y > board_limits[1] or
            x + y < corner_limits[0] or x + y > corner_limits[1] or
            x - y > 12 or y - x > 12
    )
    occupied = state.obj_by_coords(Coords(x, y))
    return not occupied and not inaccessible


def choose_direction_based_on_probability(direction_value: List[float]) -> Generator[Direction, None, None]:
    direction_table = [Direction.North, Direction.South, Direction.East, Direction.West]
    probabilities = softmax(direction_value)
    while sum(probabilities) > 0:
        ix = argmax(probabilities)
        direction = direction_table[ix]
        probabilities[ix] = 0.0  # Zero out used direction to avoid re-selecting
        yield direction


def robot(state, unit) -> Action:
    global shared_state

    upper_border = 5 <= unit.coords.x <= 13 and unit.coords.y == 1
    right_border = 5 <= unit.coords.y <= 13 and unit.coords.x == 17
    lower_border = 5 <= unit.coords.x <= 13 and unit.coords.y == 17
    left_border = 5 <= unit.coords.y <= 13 and unit.coords.x == 1
    upper_left = unit.coords.x + unit.coords.y == 6
    upper_right = unit.coords.x - unit.coords.y == 12
    lower_left = unit.coords.y - unit.coords.x == 12
    lower_right = unit.coords.x + unit.coords.y == 30

    if upper_border:
        return Action.move(Direction.South)
    if right_border:
        return Action.move(Direction.West)
    if lower_border:
        return Action.move(Direction.North)
    if left_border:
        return Action.move(Direction.East)
    if upper_left:
        return Action.move(random.choice([Direction.East, Direction.South]))
    if upper_right:
        return Action.move(random.choice([Direction.West, Direction.South]))
    if lower_left:
        return Action.move(random.choice([Direction.East, Direction.North]))
    if lower_right:
        return Action.move(random.choice([Direction.West, Direction.North]))

    enemies = state.objs_by_team(state.other_team)
    closest_enemy = min(enemies, key=lambda e: e.coords.distance_to(unit.coords), default=None)
    if closest_enemy:
        closest_distance = closest_enemy.coords.distance_to(unit.coords)
        closest_direction = unit.coords.direction_to(closest_enemy.coords)
        if closest_distance == 1 and (unit.health >= closest_enemy.health):
            return Action.attack(closest_direction)

    health_around_matrix = [
        health_by_coords(state, coord) for coord in unit.coords.coords_around()
    ]

    x = (float(unit.coords.x) - 10) / 10
    y = (float(unit.coords.y) - 10) / 10
    r = math.sqrt(x * x + y * y)

    ours = len(state.ids_by_team(state.our_team))
    others = len(state.ids_by_team(state.other_team))
    alpha = (ours + others) / 249
    beta = (ours + 1) / (others + 1)

    inputs = [x, y, r, alpha, beta] + health_around_matrix + shared_state
    output = network.forward(inputs)
    direction_value, shared_updates, decay = output[0:4], output[4:8], output[8]

    decay = sigmoid(decay)
    shared_state[:] = [
        math.tanh((1 - decay) * s + decay * u)
        for s, u in zip(shared_state, shared_updates)
    ]

    for direction in choose_direction_based_on_probability(direction_value):
        if enterable(state, unit, direction):
            return Action.move(direction)

    return Action.move(random.choice([
        Direction.North,
        Direction.South,
        Direction.East,
        Direction.West,
    ]))
