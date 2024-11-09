/**
 * Originally from https://github.com/robot-rumble/builtin-bots/blob/main/LICENSE
 * By Anton Outkine
 *
 * robot-rumble/builtin-bots is licensed under the
 * GNU General Public License v3.0
 *
 */

// credit: Grant Slatton

const ATTACK = "a";
const MOVE = "m";

const ALL_DIRECTIONS = [Direction.North, Direction.East, Direction.South, Direction.West];

const COORDS = {};
for(let x = 1; x <= 17; x++) {
  for(let y = 1; y <= 17; y++) {
    COORDS[new Coords(x, y)] = new Coords(x, y);
  }
}

const is_legal_coordinate = c => {
  if(c.x <= 0 || c.x > 17 || c.y <= 0 || c.y > 17) return false;
  if(c.y <= 5-c.x) return false;
  if(c.y <= c.x-13) return false;
  if(c.y >= c.x+13) return false;
  if(c.y >= 31-c.x) return false;
  return true;
};

const score = (friends, enemies) => {
  const unit_score = Object.keys(friends).length - Object.keys(enemies).length;

  let health_score = 0;
  for(const x in friends) {
    health_score += Math.pow(friends[x], 0.5);
  }
  for(const x in enemies) {
    health_score -= Math.pow(enemies[x], 0.5);
  }

  const map_score = {};
  for(const x in friends) {
    map_score[x] = { surround: 0, distance: 0 };
  }
  for(const x in enemies) {
    map_score[x] = { surround: 0, distance: 0 };
  }

  for(const f in friends) {
    const fc = COORDS[f];
    for(const e in enemies) {
      const d = COORDS[f].distanceTo(COORDS[e]);
      const d_score = 1.0 / Math.pow(d, 2);
      map_score[e].distance += d_score;
      map_score[f].distance -= d_score;
      if(d == 1) {
        map_score[e].surround += 1;
        map_score[f].surround -= 1;
      }
    }
  }

  let surround_score = 0;
  let distance_score = 0;
  for(const x of Object.values(map_score)) {
    surround_score += Math.pow(x.surround, 2);
    distance_score += Math.pow(x.distance, 2);
  }

  return [unit_score, surround_score, health_score, distance_score];
};


const array_cmp = (a, b) => {
  for(let i = 0; i < a.length; i++) {
    let x = a[i] - b[i];
    if(x != 0) return x;
  }
  return 0;
}

const tick = (friends, enemies, actions) => {
  for(const source in actions) {
    const action = actions[source];
    if(action == null) continue;
    let [action_type, direction] = action;
    if(action_type == MOVE) {
      const target = COORDS[source].add(direction);
      if(!(target in enemies) && !(target in friends) && is_legal_coordinate(target)) {
        const obj = source in friends? friends : enemies;
        obj[target] = obj[source];
        delete obj[source];
      }
    }
  }

  for(const source in actions) {
    const action = actions[source];
    if(action == null) continue;
    let [action_type, direction] = action;
    if(action_type == ATTACK) {
      const target = COORDS[source].add(direction);
      if(target in enemies) enemies[target] -= 1;
      if(target in friends) friends[target] -= 1;
    }
  }

  for(const enemy in enemies) {
    if(enemies[enemy] <= 0) delete enemies[enemy];
  }
  for(const friend in friends) {
    if(friends[friend] <= 0) delete friends[friend];
  }
};

let ACTIONS = null;

function initTurn(state) {
  const friends = {}
  for(const x of state.objsByTeam(state.ourTeam)) {
    friends[x.coords] = x.health;
  }
  const enemies = {}
  for(const x of state.objsByTeam(state.otherTeam)) {
    enemies[x.coords] = x.health;
  }

  let best_actions = {};

  for(const enemy in enemies) {
    best_actions[enemy] = null;
    let lowest_health = 1000;
    for(const direction of ALL_DIRECTIONS) {
      const target = COORDS[enemy].add(direction);
      if(!(target in friends)) continue;
      let h = friends[target];
      if(h > lowest_health) continue;
      lowest_health = h;
      best_actions[enemy] = [ATTACK, direction];
    }
  }

  const possible_actions = {}
  for(const friend in friends) {
    best_actions[friend] = null;
    possible_actions[friend] = [null];
    for(const direction of ALL_DIRECTIONS) {
      const target = COORDS[friend].add(direction);
      if(!is_legal_coordinate(target)) continue;
      if(target in enemies) {
          possible_actions[friend].push([ATTACK, direction]);
      } else {
          possible_actions[friend].push([MOVE, direction]);
      }
    }
  }

  let fs = Object.assign({}, friends);
  let es = Object.assign({}, enemies);
  tick(fs, es, best_actions);
  let best_score = score(fs, es);

  for(const f in friends) {
    let actions = Object.assign({}, best_actions);
    for(const a of possible_actions[f]) {
      actions[f] = a;
      fs = Object.assign({}, friends);
      es = Object.assign({}, enemies);
      tick(fs, es, actions);
      let s = score(fs, es);
      if(array_cmp(best_score, s) < 0) {
        best_score = s;
        best_actions = Object.assign({}, actions);
      }
    }
  }

  ACTIONS = best_actions;
}

function robot(state, unit) {
  let best_action = ACTIONS[unit.coords];

  if(best_action == null) return null;

  [action_type, direction] = best_action;

  if(action_type == ATTACK) {
    return Action.attack(direction);
  } else {
    return Action.move(direction);
  }
};
