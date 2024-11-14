"""
This bot is a combination of my initial idea to analyze groups of bots with a maximum of one empty space between each member of a group as 'gloms' and some iterative improvements over mitch84's retreat_walk2 bot movement style (group coordination, avoid friendly fire, multi-unit retreats). 

The idea behind the glom concept is to generally try to group up bots in such a way that they form a sort of lattice structure where lesser enemies may throw themselves into the gaps between individual bots, only to be immediatly surrounded and pummeled to death. Also, to intentionally surround and pummel smaller groups to death.

The overall strategy at this point is to glom together a group big enough to take the center, then fight defensively to annoy the opponent to death. The advanced movement rules allow for strong offensive play by the smaller gloms, as well. Reliably beats crw_preempt, centerrr, black_magic, rule99, and chaos_legion, per Garage testing. At this point this is likely the strongest bot on the public board.

Given that this bot is the result of an iterative improvement process, its code is ugly as hell. You have my sympathies in the form of copious comments
-- entropicdrifter
"""

from functools import lru_cache

class GlommedBot:
    def __init__(self, bot, glom_id):
        self.bot = bot
        self.glom = glom_id
        self.health = bot.health
        self.coords = bot.coords
        self.location = bot.coords
        self.team = bot.team
        self.id = bot.id
    
    def __str__(self):
        return self.bot.__str__()
    
    def __repr__(self):
        return self.__str__()
    
    def set_target(self, target):
        self.target = target
        
    def set_location(self, coords):
        self.location = coords
        board_state[self.coords.x][self.coords.y] = None
        board_state[coords.x][coords.y] = self
        
    def remove_from_glom(self):
        self.glom.remove_bot(self)

class Glom:
    def __init__(self):
        self.bots = set()
        self.health = 0
        self.enemy_hp = 0
    
    def __str__(self):
        return "\n\tbots: " + str(self.bots) + "\n\t" + "health: " + str(self.health) + "\n\tenemy_hp:"+ str(self.enemy_hp)
    
    def set_target(self, target):
        self.target = target
    
    def add_bot(self, bot):
        self.bots.add(bot)
        self.health += bot.health
        
    def remove_bot(self, bot):
        if bot in self.bots:
            self.bots.remove(bot)
            self.health -= bot.health
            if self.target and self.target.glom:
                self.target.glom.enemy_hp -= bot.health

claimed_locations = None
board_state = None
center = None
moves = None
debug_info = None
threat_map = None
spawn_safe = None

# sort team by shortest walking distance to unit
# using health as a tie-breaker so we finish off the weak first
def nearest(team, unit):
    if team:
        return nearest_inner(tuple(team), unit)
    return None

@lru_cache
def nearest_inner(team, unit):
    return min(team, key=lambda e: e.coords.walking_distance_to(unit.coords) + 0.1 * e.health)


# sort possible moves by shortest walking distance to target location
# using threat level as a tie-breaker so we avoid pointless danger
def best_move(moves, target_location, state, unit):
    if moves:
        return min(moves, key=lambda d: (unit.coords + d).walking_distance_to(target_location) + 0.01 * total_threat(state, unit, unit.coords + d))
    return None

def best_retreat_move(moves, target_location, state, unit):
    if moves:
        return min(moves, key=lambda d: total_threat(state,unit, unit.coords + d))
    return None

def strongest_adj_enemy(unit):
    strongest = None
    for d in Direction:
        location = unit.coords + d
        enemy = board_state[location.x][location.y]
        if enemy and enemy.team != unit.team:
            if not strongest or enemy.health > strongest.health:
                strongest = enemy
    return strongest

# Credit for this method to atl15, taken from the bot centerrr
#checking if bot is in spawn
spawn_coords = [(14,2),(16,4),(16,14),(14,16),(4,16),(2,14),(2,4),(4,2)]
def check_spawn(unit_coord):
    if unit_coord[1] == 1 or unit_coord[1] == 17 or unit_coord[0] == 1 or       unit_coord[0] == 17:
        return True
    for coord in spawn_coords:
        if unit_coord == coord:
            return True
    return False

def in_bounds(state, coord):
    safe = coord[0] > 0 and coord[0] < 18 and coord[1] > 0 and coord[1] < 18 and coord[0]
    if safe:
        obj = state.obj_by_coords(coord)
        safe = not obj or obj.obj_type != ObjType.Terrain
    return safe

def total_threat(state, unit, coords):
    threat = threat_map[coords.x][coords.y][unit.team]
    return threat - unit.health

def potential_threat(state, unit, unit_coord):
    p = 0
    for d in Direction:
        p += total_threat(state, unit, unit_coord + d)
    return p

def direction_to(unit, target):
    other_coords = target.coords
    coords = unit.coords
    if coords[1] > other_coords[1]:
        return Direction.North
    elif coords[0] > other_coords[0]:
        return Direction.West
    elif coords[0] < other_coords[0]:
        return Direction.East
    elif coords[1] < other_coords[1]:
        return Direction.South
    
# Pre-calculate gloms and targets, plus initialize the globals, then pre-calculate all moves
def init_turn(state):
    global gloms
    global claimed_locations
    global center
    global board_state
    global moves
    global debug_info
    global threat_map
    global spawn_safe
    allies = state.objs_by_team(state.our_team)
    enemies = state.objs_by_team(state.other_team)
    unglommed = set(allies + enemies)
    center = Coords(9, 9)
    gloms = dict()
    claimed_locations = set()
    board_state = [[ None for y in range(19) ] for x in range(19)]
    moves = dict()
    debug_info = dict()
    threat_map = [[{state.our_team : 0, state.other_team: 0} for y in range(19) ] for x in range(19)]
    spawn_safe = state.turn % 10 < 7 
    spawn_threat = 1 if state.turn % 10 < 9 else 5
    for n in range(1,18):
        for team in Team:
            threat_map[n][1][team] = spawn_threat
            threat_map[n][17][team] = spawn_threat
            threat_map[1][n][team] = spawn_threat
            threat_map[17][n][team] = spawn_threat
    for coord in spawn_coords:
        for team in Team:
            threat_map[coord[0]][coord[1]][team] = spawn_threat
    if not len(enemies):
        debug_info = { unit.id: dict() for unit in allies }
        moves = { unit.id: choose_move(state, GlommedBot(unit, 1), True, [], None) for unit in allies }
        return
    
    
    def apply_threat(unit, coords, value):
        threat_map[coords.x][coords.y][unit.team.opposite] += value
        threat_map[coords.x][coords.y][unit.team] -= value * 0.1
        if value >= 1:
            for direction in Direction:
                new_location = coords + direction
                if in_bounds(state, new_location):
                    apply_threat(unit, new_location, 0.2 * value)
    
    def glom_all():
        glommed = dict()
        # Glom up all bots
        while len(unglommed):
            current = unglommed.pop()

            glom_unit(current, glommed)
        
        allies = []
        enemies = []
        for bot in glommed.values():
            board_state[bot.coords.x][bot.coords.y] = bot
            if bot.team == state.our_team:
                allies.append(bot)
                apply_threat(bot, bot.coords, 1 + 0.1 * bot.health)
            else:
                enemies.append(bot)
        for enemy in enemies:
            move = retreat(state, enemy)
            if move:
                apply_threat(enemy, enemy.coords + move.direction, 1 + 0.1 * enemy.health)
            else:
                apply_threat(enemy, enemy.coords, 1 + 0.1 * enemy.health)
        return (allies, enemies)
    
    def glom_unit(current, glommed):
        glom_id = len(gloms)
        
        glom = Glom()
        glom_bot = GlommedBot(current, glom)
        glommed[current.id] = glom_bot
            
        glom_neighbors(glom_bot, glom, unglommed, glommed)
        gloms[glom_id] = glom
        # Pick a target for each glom
        target_group = enemies if current.team == state.our_team else allies
        target = nearest(target_group, current)
        target_glommed = None
        if target and target in unglommed:
            unglommed.remove(target)
            target_glommed = glom_unit(target, glommed)
        elif target:
            target_glommed = glommed[target.id]
        glom.set_target(target_glommed)
        target_glommed.glom.enemy_hp += glom.health
        return glom_bot
    
    # This is where the magic happens. 
    # Recursively look at nearby bots who haven't been added to a glom yet
    # and group them up by team.
    def glom_neighbors(bot, glom, unglommed, glommed):
        glom.add_bot(bot)
        
        neighbors = [e for e in unglommed if e and e.coords.walking_distance_to(bot.coords) < 3]
        for neighbor in neighbors:
            if neighbor.team == bot.team and neighbor in unglommed:
                unglommed.remove(neighbor)
                glommed_neighbor = GlommedBot(neighbor, bot.glom)
                glommed[neighbor.id] = glommed_neighbor
                glom_neighbors(glommed_neighbor, glom, unglommed, glommed)
        
        return glom
    
    
    (allies, enemies) = glom_all()
    
    biggest_glom = max(gloms, key=lambda e: len(gloms[e].bots) + 0.01 * gloms[e].health)
    allies = sorted(allies, key=lambda unit: unit.health)
    moves = dict()
    for ally in allies:
        moves[ally.id] = retreat(state, ally)
    for ally in allies[::-1]:
        if not moves.get(ally.id):
            we_big = ally.glom == biggest_glom
            nearest_enemy = nearest(enemies, ally)
            action = choose_attack(state, ally, we_big, nearest_enemy)
            if not action:
                action = choose_move(state, ally, we_big, allies, nearest_enemy)
            moves[ally.id] = action
    for ally in allies[::-1]:
        if not moves.get(ally.id):
            we_big = ally.glom == biggest_glom
            nearest_enemy = nearest(enemies, ally)
            action = choose_move(state, ally, we_big, allies, nearest_enemy)
            moves[ally.id] = action

# Credit to mitch84 for writing the walk_to and retreat_inner methods for the bot retreat_walk2
# The only things I've changed here are to check for movement collisions
# between my own bots using the claimed_locations global set
# and to add one param to the retreat_inner method so I can check
# whether a unit would retreat after making its planned movement
def walk_to(state, unit, to_coords, retreat = False):

    blanks = []
    for direction in Direction:
        possible_location = unit.coords + direction
        if possible_location not in claimed_locations and in_bounds(state, possible_location):
            other = board_state[possible_location.x][possible_location.y]
            if not other and (spawn_safe or not check_spawn(possible_location)):
                blanks.append(direction)
    move = None
    if retreat:
        move = best_retreat_move(blanks, to_coords, state, unit)
    else:
        move = best_move(blanks, to_coords, state, unit)
    debug_info[unit.id]['sorted moves'] = blanks
    
    return move


def retreat(state, unit):
    if moves.get(unit.id):
        return moves[unit.id]
    debug_info[unit.id] = dict()
    unit.threat = total_threat(state, unit, unit.coords)
    options = retreat_inner(state, unit, unit.coords, unit.threat)
    enemy_glom = unit.glom.target.glom
    glom = unit.glom
    if not options:
        return None
    for result in options:
        if result and not result[1]:
            new_location = unit.coords + result[0]
            board_state[new_location.x][new_location.y] = unit
            board_state[unit.coords.x][unit.coords.y] = None
            debug_info[unit.id]['action'] = 'retreating'
            enemy = strongest_adj_enemy(unit)
            if enemy:
                glom.target = enemy
            if glom.target.glom != enemy_glom:
                glom.target.glom.enemy_hp += glom.health - unit.health
                enemy_glom.enemy_hp -= glom.health
            else:
                enemy_glom.enemy_hp -= unit.health
            unit.remove_from_glom()
            return Action.move(result[0])
        elif result and unit.team == state.our_team:
            other = result[1]
            if moves.get(other.id):
                continue
            new_location = unit.coords + result[0]
            if not debug_info.get(other.id):
                debug_info[other.id] = dict()
            for d in Direction:
                ally_direction = try_move(state, other, other.coords + d, 'coordinated retreat', retreat=True)
                if ally_direction:
                    moves[other.id] = Action.move(ally_direction)
                    board_state[new_location.x][new_location.y] = unit
                    board_state[unit.coords.x][unit.coords.y] = None
                    debug_info[unit.id]['action'] = 'coordinated retreat'
                    enemy = strongest_adj_enemy(unit)
                    if enemy:
                        glom.target = enemy
                    if glom.target.glom != enemy_glom:
                        glom.target.glom.enemy_hp += glom.health
                        glom.target.glom.enemy_hp -= unit.health
                        glom.target.glom.enemy_hp -= other.health
                        enemy_glom.enemy_hp -= glom.health
                    else:
                        enemy_glom.enemy_hp -= unit.health
                        enemy_glom.enemy_hp -= other.health
                    unit.remove_from_glom()
                    other.remove_from_glom()
                    return Action.move(result[0])
    return None

def retreat_inner(state, unit, coords, threat):
    enemies = []
    options = []
    for direction in Direction:
        new_location = coords + direction
        if new_location not in claimed_locations and in_bounds(state, new_location) and (spawn_safe or not check_spawn(new_location)):
            new_threat = total_threat(state, unit, new_location)
            other = board_state[new_location.x][new_location.y]
            if other and other.team == unit.team.opposite:
                enemies.append(other)
            # the 'or new_location == unit.coords' bit is necessary 
            # because it allows us to check hypotheticals, 
            # i.e. 'what if I go here, would I retreat?'
            # In those cases, it's necessary to treat the unit's current spot
            # as empty since they won't be there on the turn we're considering.
            # This comparison is consistently safe because 
            # if unit.coords == coords, then new_location will 
            # never == unit.coords
            elif new_threat < threat:
                options.append((new_threat + 0.01 * new_location.walking_distance_to(center), (direction, other)))
    options = [option[1] for option in sorted(options, key=lambda x: x[0])]
    if unit.team == state.our_team:
        debug_info[unit.id]['retreat options'] = options
    if options:
        # retreat if flanked and there is room to retreat
        if len(enemies) > 1:
            return options
        elif len(enemies):
            enemy_threat = total_threat(state, enemies[0], enemies[0].coords)
            if enemy_threat < threat:
                return options
        elif not spawn_safe and check_spawn(coords):
            return options
    return None

# Make a movement action only if we wouldn't retreat immediately after,
# assuming enemies don't move
def try_move(state, unit, target_location, action_name, retreat=False):
    move_direction = walk_to(state, unit, target_location, retreat=retreat)
    if move_direction:
        new_location = unit.coords + move_direction
        future_threat = total_threat(state, unit, new_location)
        would_retreat = retreat_inner(state, unit, new_location, future_threat)
        if not would_retreat:
            debug_info[unit.id]['action'] = action_name
            board_state[new_location.x][new_location.y] = unit
            board_state[unit.coords.x][unit.coords.y] = None
            return move_direction

# Make an attack only if it will not hit any allies
def try_attack(state, unit, attack_direction, action_name):
    attack_location = unit.coords + attack_direction
    if in_bounds(state, attack_location):
        obj = board_state[attack_location.x][attack_location.y]
        if not obj or (obj.team and obj.team == unit.team.opposite):
            debug_info[unit.id]['action'] = action_name
            claimed_locations.add(attack_location)
            return Action.attack(attack_direction)

def choose_move(state, unit, we_big, allies, nearest_enemy):

    if not we_big:
        # If we're not retreating or attacking, try to glom
        ally = nearest([a for a in allies if a.glom != unit.glom and a.glom.health > unit.glom.health], unit)
        if ally:
            debug_info[unit.id]['ally'] = ally
            direction = try_move(state, unit, ally.coords, 'glomming')
            if direction:
                return Action.move(direction)
    
    if unit.coords != center:
        # If we can't find a glom to join, take the center
        direction = try_move(state, unit, center, 'take the center')
        if direction:
            new_location = unit.coords + direction
            # Spread out in a lattice
            for d in Direction:
                claimed_locations.add(new_location + d)
            return Action.move(direction)
    
    # If we got here, we're probably trapped. Fight to the death!
    attack_direction = direction_to(unit, nearest_enemy)
    attack = try_attack(state, unit, attack_direction, 'fight to the death')
    if attack:
        # Spread out in a lattice
        for d in Direction:
            claimed_locations.add(unit.coords + d)
        return attack
    

def choose_attack(state, unit, we_big, nearest_enemy):
    if moves.get(unit.id):
        debug_info[unit.id]['action'] = 'coordinated retreat'
        return moves[unit.id]
    
    glom = unit.glom
    
    # find the target
    enemy = glom.target
    # get the target's glom
    enemy_glom = enemy.glom
    if len(enemy_glom.bots):
        enemy = nearest(enemy_glom.bots, unit)
    if not enemy:
        enemy = nearest_enemy
    unit.set_target(enemy)
    
    attack_direction = direction_to(unit, enemy)
    
    debug_info[unit.id]['glom'] = glom
    debug_info[unit.id]['target'] = enemy.bot
    
    current_threat = total_threat(state, unit, unit.coords)
    n_enemy_threat = total_threat(state, nearest_enemy, nearest_enemy.coords)
    
    # Take any free wins
    if unit.coords.walking_distance_to(nearest_enemy.coords) == 1 and n_enemy_threat > current_threat:
        attack_direction = direction_to(unit, nearest_enemy)
        attack = try_attack(state, unit, attack_direction, 'attack of opportunity')
        if attack:
            return attack
        attack_direction = direction_to(unit, enemy)
    
    
    
    # If our glom is bigger or stronger, kill 'em all
    elif len(enemy_glom.bots) < len(glom.bots)  or glom.health > enemy_glom.health or enemy_glom.enemy_hp > enemy_glom.health:
        
        enemy_dist = unit.coords.walking_distance_to(enemy.coords)
        they_scary = False if enemy_dist == 1 else total_threat(state, enemy, enemy.coords - attack_direction) < total_threat(state, unit, unit.coords + attack_direction)
        # If you're at a local disadvantage or hogging the center, 
        # fight defensively, otherwise go on the offensive and try to surround
        if ((we_big or they_scary) and enemy_dist <= 2) or enemy_dist == 1:
            attack = try_attack(state, unit, attack_direction, 'attack')
            if attack:
                if enemy_dist > 1:
                    for d in Direction:
                        claimed_locations.add(unit.coords + d)
                return attack
        
        if we_big:
            # If we're the biggest glom, take the center
            direction = try_move(state, unit, center, 'take the center')
            if direction:
                new_location = unit.coords + direction
                # Spread out in a lattice when dominant
                for d in Direction:
                    claimed_locations.add(new_location + d)
                return Action.move(direction)
        else:
            # dynamic movement allows us to consistently surround them
            direction = try_move(state, unit, enemy.coords, 'swarm')
            if direction:
                return Action.move(direction)
    
def robot(state, unit):
    for key, value in debug_info[unit.id].items():
        debug.inspect(key, value)
    return moves[unit.id]
