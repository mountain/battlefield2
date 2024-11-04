import random
import struct


from zb.db import rc


BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LENGTH = 12 * 24 + 24 + 24 * 6 + 6


with open('template/zot-bot.py') as zs:
    tmpl = zs.read()


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


class Bot:
    @classmethod
    def fetch(cls, bid):
        try:
            bid = int(bid)
        except ValueError:
            pass
        if type(bid) is int:
            robo = rc.get('bot:%06d:robo' % int(bid)).decode('utf8')
            return cls(bid, robo)
        else:
            return cls(bid, None)

    @classmethod
    def create(cls, robo):
        if not rc.exists('serials:bot'):
            rc.set('serials:bot', 0)

        bid = rc.incr('serials:bot')
        if bid < 0:
            raise Exception(bid)

        rc.set('bot:%06d:robo' % int(bid), robo)
        rc.zadd('board', {bid: 20})
        return cls(bid, robo)

    @classmethod
    def next(cls):
        robo = compress_floats([random.uniform(-1, 1) for _ in range(LENGTH)])
        return Bot.create(robo)

    def __init__(self, bid, robo):
        self.bid = bid
        if type(bid) == int:
            self.robo = robo
            self.path = 'robots/%06d.py' % self.bid
        else:
            self.path = './robots/%s.py' % bid

    def dump(self):
        if type(self.bid) == int:
            with open(self.path, mode='w') as tf:
                tf.write(tmpl % self.robo)
                tf.flush()
        return self.path

    def unlink(self):
        import os
        if os.path.exists(self.path) and type(self.bid) == int:
            os.unlink(self.path)

    def mutate(self):
        def mutate_seq(seq):
            seq = decompress_floats(seq)
            idx = random.randint(0, len(seq) - 1)
            seq[idx] = random.uniform(-1, 1)
            return compress_floats(seq)

        robo = mutate_seq(self.robo)
        Bot.create(robo)

    def crossover(self, other):
        def crossover_seq(seq1, seq2):
            seq1 = decompress_floats(seq1)
            seq2 = decompress_floats(seq2)
            idx1 = random.randint(0, len(seq1) - 1)
            idx2 = random.randint(idx1, len(seq1) - 1)
            seq1[idx1:idx2] = seq2[idx1:idx2]
            return compress_floats(seq1)

        robo = crossover_seq(self.robo, other.robo)
        Bot.create(robo)

    def swallow(self, other):
        def swallow_seq(seq1, seq2):
            seq1 = decompress_floats(seq1)
            seq2 = decompress_floats(seq2)
            idx1 = random.randint(0, len(seq1) - 1)
            idx2 = random.randint(idx1, len(seq1) - 1)
            for i in range(idx1, idx2):
                seq1[i] = max(-1, min(1, seq1[i] + seq2[i]))
            return compress_floats(seq1)

        robo = swallow_seq(self.robo, other.robo)
        Bot.create(robo)
