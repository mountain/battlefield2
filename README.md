# nn-bot: an evolutionary neural-network bot playing Robot Rumble games

The idea of zot-bot is to evolve the bot by itself.

To run the python code, you need creat a symbol link to rumblebot in the project root.

## how to run

install dependencies

```bash
brew install redis
pip3 install redis rq rq-dashboard sh
ln -s RUMBLEBOT PROJ_DIR
```

### commands

#### init 
initialize the genetic pool with size of 1000

```bash
zb init -n 1000
```

#### worker 
execute the worker asynchronously

```bash
zb worker -t match -n 10
```

currently only two buildin task(the -t paramater) are supported
* match
* exam

#### run 
execute the genetic evolution

```bash
zb run -n 10000
```

#### report 
report the first 10 seeds

```bash
zb report -n 10
```

#### peek 
given the detailed information of the seed with bid 1234

```bash
zb peek -b 1234
```

#### poke 
given the code of the seed with bid 1234

```bash
zb poke -b 1234
```

#### benchmark 
make a benchmark between bot with bid 1234 and the buildin random-bot

```bash
zb benchmark -b 1234 -t random-bot -n 200
```

currently only two buildin bot are supported
* random-bot
* chaser

#### play 
make a play between bot with bid 1234 and the buildin random-bot

```bash
zb play -b 1234 -t random-bot -m web
```

currently only two buildin bot are supported
* random-bot
* chaser

two mode are supported
* term
* web
