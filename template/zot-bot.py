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
network = NeuralNetwork(input_size=17, hidden_size=27, output_size=9)
network.set_parameters(decompress_floats("""
%s
"""))


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


def health_by_coords(state, coord) -> float:
    obj = state.obj_by_coords(coord)
    if obj and obj.team is not None:
        return obj.health / 5 if obj.team == state.our_team else -obj.health / 5
    return 0


def calculate_torque_vector(state, unit) -> List[float]:
    our_torque_x = 0.0
    our_torque_y = 0.0
    enemy_torque_x = 0.0
    enemy_torque_y = 0.0

    center_x, center_y = unit.coords.x, unit.coords.y

    for obj in state.objs_by_team(state.our_team):
        delta_x = obj.coords.x - center_x
        delta_y = obj.coords.y - center_y

        torque_x = obj.health * delta_x
        torque_y = obj.health * delta_y
        our_torque_x += torque_x / 9 / 5
        our_torque_y += torque_y / 9 / 5

    for obj in state.objs_by_team(state.other_team):
        delta_x = obj.coords.x - center_x
        delta_y = obj.coords.y - center_y

        torque_x = obj.health * delta_x
        torque_y = obj.health * delta_y
        enemy_torque_x += torque_x / 9 / 5
        enemy_torque_y += torque_y / 9 / 5

    return [-enemy_torque_x, -enemy_torque_y, our_torque_x, our_torque_y]


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

    around_matrix = [
        health_by_coords(state, coord) for coord in unit.coords.coords_around()
    ]
    torque = calculate_torque_vector(state, unit)

    x = (float(unit.coords.x) - 9) / 9
    y = (float(unit.coords.y) - 9) / 9
    r = math.sqrt(x * x + y * y)

    ours = len(state.ids_by_team(state.our_team))
    others = len(state.ids_by_team(state.other_team))
    alpha = (ours + others) / 249
    beta = (ours + 1) / (others + 1)

    inputs = [x, y, r, alpha, beta] + around_matrix + torque + shared_state
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
