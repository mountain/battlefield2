import argparse
from asyncio import timeout

from redis import Redis
from rq import Queue

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

init_parser = subparsers.add_parser('init')
init_parser.add_argument('-n', '--diversity', default=1000, type=int)

run_parser = subparsers.add_parser('run')
run_parser.add_argument('-n', '--round', default=1000, type=int)

pek_parser = subparsers.add_parser('peek')
pek_parser.add_argument('-b', '--bid', default=1, type=int)

pok_parser = subparsers.add_parser('poke')
pok_parser.add_argument('-b', '--bid', default=1, type=int)

rep_parser = subparsers.add_parser('report')
rep_parser.add_argument('-n', '--num', default=10, type=int)

wrk_parser = subparsers.add_parser('worker')
wrk_parser.add_argument('-c', '--concurrency', default=10, type=int)
wrk_parser.add_argument('-n', '--num', default=14400, type=int)
wrk_parser.add_argument('-t', '--task', default='match', type=str)

dsh_parser = subparsers.add_parser('dashboard')

ply_parser = subparsers.add_parser('play')
ply_parser.add_argument('-b', '--bid', default=1, type=int)
ply_parser.add_argument('-t', '--target', default='random-bot', type=str)
ply_parser.add_argument('-m', '--mode', default='web', type=str)

bnm_parser = subparsers.add_parser('benchmark')
bnm_parser.add_argument('-b', '--bid', default=1, type=int)
bnm_parser.add_argument('-t', '--target', default='random-bot', type=str)
bnm_parser.add_argument('-n', '--num', default=300, type=int)


def pooled_proc(args):
    from zb.board import exam, match
    from zb.db import rc
    from rq import Worker, Queue

    ix, task_name = args
    queue = Queue(task_name, connection=rc)
    worker = Worker([queue], connection=rc)
    worker.work()
    return ix


if __name__ == '__main__':
    import sys
    from zb.board import init, match, summary, peek, poke, benchmark, play
    opts = parser.parse_args(sys.argv[1:])
    if opts.command == 'init':
        init(opts.diversity)
    if opts.command == 'run':
        for _ in range(int(opts.round)):
            match.delay()
    if opts.command == 'peek':
        peek(int(opts.bid))
    if opts.command == 'poke':
        poke(int(opts.bid))
    if opts.command == 'report':
        summary(int(opts.num))

    if opts.command == 'play':
        play(int(opts.bid), str(opts.target), str(opts.mode))

    if opts.command == 'benchmark':
        benchmark(int(opts.bid), str(opts.target), int(opts.num))

    if opts.command == 'worker':
        from multiprocessing import Pool

        sys.path += list('.')
        with Pool(opts.concurrency) as p:
            p.map(pooled_proc, [(ix, opts.task) for ix in range(opts.num)])

    if opts.command == 'dashboard':
        import logging
        import rq_dashboard
        from flask import Flask
        sys.path += list('.')
        app = Flask(__name__)
        app.config.from_object(rq_dashboard.default_settings)
        app.register_blueprint(rq_dashboard.blueprint, url_prefix="/dashboard")

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.error(" * Running on {}:{}".format('0.0.0.0', 4321))

        app.run(host='0.0.0.0', port=4321, debug=False)
