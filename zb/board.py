from zb.db import rc
from rq.decorators import job


TIMEOUT = 120
diversity = int(rc.get('board:diversity')) if rc.exists('board:diversity') else 1000


def get_diversity():
    return int(rc.get("board:diversity"))


def init(d):
    from zb.bot import Bot
    rc.set('board:diversity', d)
    for _ in range(int(str(d))):
        Bot.next()


def add(delta):
    d = int(rc.get('board:diversity')) + delta
    from zb.bot import Bot
    rc.set('board:diversity', d)
    for _ in range(int(str(delta))):
        Bot.next()


def cleanup_bot(bid, archive=False):
    if archive:
        # archive_bot(bid)
        pass
    else:
        rc.delete(f'bot:{bid:06d}:policy')
        rc.delete(f'match:{bid:06d}')
        rc.delete(f'exam:{bid:06d}')


def maintain_board():
    d = get_diversity()
    rc.zremrangebyscore('board', -100, 0)
    card = rc.zcard('board')

    for key, _ in rc.zscan_iter('board', score_cast_func=int):
        if int(rc.zscore('board', key)) <= 0:
            cleanup_bot(int(key))

    if card > d:
        removed = rc.zpopmin('board', card - d)
        for bid, _ in removed:
            cleanup_bot(int(bid))

    elif card < d:
        for _ in range(d - card):
            Bot.next()


@job('match', connection=rc, timeout=TIMEOUT+1)
def match():
    import subprocess
    from random import random, sample
    from zb.bot import Bot
    from math import log1p

    try:
        seeds = []
        for key, score in rc.zscan_iter('board', score_cast_func=int):
            key = int(key)
            for _ in range(score):
                seeds.append(key)
        blue_id, red_id = sample(seeds, 2)
        blue, red = Bot.fetch(blue_id), Bot.fetch(red_id)
        rc.setnx('match:%06d:%06d' % (blue_id, red_id), 0)
        rc.setnx('match:%06d:%06d' % (red_id, blue_id), 0)

        bfile, rfile = blue.dump(), red.dump()
        p = subprocess.run(['./rumblebot', 'run', 'term', bfile, rfile], capture_output=True, text=True, timeout=TIMEOUT)
        if p.stdout:
            result = p.stdout

        else:
            raise Exception(p.stderr)
        if result.find('Done! it was a tie') > -1:
            rc.zincrby('board', 0, blue_id)
            rc.zincrby('board', 0, red_id)
        elif result.find('Done! Blue won') > -1:
            blue_score = max(int(rc.zscore('board', blue_id)), 1)
            red_score = max(int(rc.zscore('board', red_id)), 1)
            bscore = int(log1p(blue_score)) + 1
            rscore = int(log1p(red_score)) + 1
            rc.zincrby('board', rscore, blue_id)
            rc.zincrby('board', -bscore, red_id)
            rc.incr('match:%06d:%06d' % (blue_id, red_id), 1)
            rc.incr('match:%06d:%06d' % (red_id, blue_id), -1)
            if random() > 0.9:
                blue.mutate()
            if random() > 0.95:
                blue.crossover(red)
            if random() > 0.99:
                blue.swallow(red)
        elif result.find('Done! Red won') > -1:
            blue_score = max(int(rc.zscore('board', blue_id)), 1)
            red_score = max(int(rc.zscore('board', red_id)), 1)
            bscore = int(log1p(blue_score)) + 1
            rscore = int(log1p(red_score)) + 1
            rc.zincrby('board', -rscore, blue_id)
            rc.zincrby('board', bscore, red_id)
            rc.incr('match:%06d:%06d' % (red_id, blue_id), 1)
            rc.incr('match:%06d:%06d' % (blue_id, red_id), -1)
            if random() > 0.9:
                red.mutate()
            if random() > 0.95:
                red.crossover(blue)
            if random() > 0.99:
                red.swallow(blue)

        exam.delay(blue_id)
        exam.delay(red_id)
    except Exception as e:
        import traceback
        import sys
        traceback.print_exception(*sys.exc_info())
        try:
            rc.zincrby('board', -1, blue_id)
            rc.zincrby('board', -1, red_id)
        except:
            pass
    finally:
        blue.unlink()
        red.unlink()
        try:
            maintain_board()
        except:
            pass


def teach(student_id, teacher, teacher_file, reward=1, penalty=0):
    import subprocess
    from zb.bot import Bot

    try:
        blue = Bot.fetch(student_id)
        rc.setnx('exam:%06d:%s' % (student_id, teacher), 0)
        rc.setnx('exam:%s:%06d' % (teacher, student_id), 0)

        bfile, rfile = blue.dump(), teacher_file
        p = subprocess.run(['./rumblebot', 'run', 'term', bfile, rfile], capture_output=True, text=True, timeout=TIMEOUT)
        if p.stdout:
            result = p.stdout
        else:
            raise Exception(p.stderr)
        if result.find('Done! it was a tie') > -1:
            rc.zincrby('board', 0, student_id)
        elif result.find('Done! Blue won') > -1:
            rc.zincrby('board', reward, student_id)
            rc.incr('exam:%06d:%s' % (student_id, teacher), 1)
            for _ in range(reward):
                blue.mutate()
        elif result.find('Done! Red won') > -1:
            rc.zincrby('board', penalty, student_id)
            rc.incr('exam:%s:%06d' % (teacher, student_id), 1)
    except Exception as e:
        import traceback
        import sys
        traceback.print_exception(*sys.exc_info())
        try:
            rc.zincrby('board', -1, student_id)
        except:
            pass
    finally:
        blue.unlink()
        try:
            maintain_board()
        except:
            pass


@job('exam', connection=rc, timeout=4 * TIMEOUT+1)
def exam(student_id):
    # teach(student_id, 'simple-bot', 'robots/simple-bot.js', reward=1, penalty=-1)
    # teach(student_id, 'random-bot', 'robots/random-bot.js', reward=2, penalty=-1)
    # teach(student_id, 'flail', 'robots/flail.js', reward=1, penalty=-5)
    # teach(student_id, 'heuristic', 'robots/heuristic.js', reward=2, penalty=-4)
    teach(student_id, 'chaser', 'robots/chaser.js', reward=1, penalty=-6)
    teach(student_id, 'neuralbot4', 'robots/neuralbot4.py', reward=2, penalty=-5)
    teach(student_id, 'black-magic', 'robots/black_magic.js', reward=4, penalty=-4)
    teach(student_id, 'seven-of-nine', 'robots/seven_of_nine.py', reward=8, penalty=-3)
    teach(student_id, 'we-are-borg', 'robots/we_are_borg.py', reward=16, penalty=-2)
    teach(student_id, 'gigachad', 'robots/gigachad.py', reward=32, penalty=-1)


def peek(bid):
    from zb.bot import Bot
    bot = Bot.fetch(bid)
    print('-----------------------------------------')
    print('bid: %06d' % bid)
    print('-----------------------------------------')
    print('policy')
    print(bot.policy)
    print('-----------------------------------------')

def poke(bid):
    from zb.bot import Bot, tmpl
    bot = Bot.fetch(bid)
    code = (tmpl.replace('%', '%%').replace('%%s', '%s') % bot.policy).strip()
    print('-----------------------------------------')
    print('code                       length: %06d' % len(code))
    print('-----------------------------------------')
    print(code)
    print('-----------------------------------------')


def summary(num):
    def descr(row):
        return '%06d-%04d     ' % (int(row[0]), int(row[1]))

    card, population = 0, 0
    ranks = []
    for key, score in rc.zscan_iter('board', score_cast_func=int):
        ranks.append((key, score))
        card += 1
        population += score
    ranks = sorted(ranks, key=lambda x: -x[1])
    print('-----------------------------------------')
    print('report')
    print('-----------------------------------------')
    print('diversity: %03d         population: %05d' % (card, population))
    print('-----------------------------------------')
    for ix, row in enumerate(ranks[:num]):
        print('%02d: %s' % (ix + 1, descr(row)))
    print('-----------------------------------------')


def benchmark(bid, target, count):
    import sys
    import subprocess
    from zb.bot import Bot

    blue, red = Bot.fetch(bid), Bot.fetch(target)
    bscore, rscore = 0, 0
    print('---------------------------------------------------------')
    for _ in range(count):
        try:
            bfile, rfile = blue.dump(), red.dump()
            p = subprocess.run(['./rumblebot', 'run', 'term', bfile, rfile], capture_output=True, text=True,
                               timeout=TIMEOUT, cwd='.')
            if p.stdout:
                result = p.stdout
            else:
                raise Exception(p.stderr)
            if result.find('Done! it was a tie') > -1:
                bscore += 0.5
                rscore += 0.5
                print('.', sep='', end='')
                sys.stdout.flush()
            elif result.find('Done! Blue won') > -1:
                bscore += 1
                print('+', sep='', end='')
                sys.stdout.flush()
            elif result.find('Done! Red won') > -1:
                rscore += 1
                print('-', sep='', end='')
                sys.stdout.flush()
        except Exception as e:
            rscore += 1
            print('?', sep='', end='')
            sys.stdout.flush()
    print('\n---------------------------------------------------------')
    print('benchmark        total: %d      blue: %06d      red: %s' % (count, bid, target))
    print('---------------------------------------------------------')
    print('score: %0.04f' % ((bscore + 1) / (rscore + 1)))
    print('---------------------------------------------------------')


def play(bid, target, mode='web'):
    import subprocess
    from zb.bot import Bot

    blue = Bot.fetch(bid)
    print('---------------------------------------------------------')
    bfile, rfile = blue.dump(), 'robots/%s.js' % target
    if mode == 'term':
        p = subprocess.run(['./rumblebot', 'run', 'term', bfile, rfile], capture_output=True, text=True,
                           timeout=TIMEOUT, cwd='.')
        if p.stdout:
            result = p.stdout
        print(result)
    else:
        print('check your browser and press return to finish...')
        p = subprocess.run(['./rumblebot', 'run', 'web', bfile, rfile], capture_output=True, text=True,
                           cwd='.')
    print('---------------------------------------------------------')
