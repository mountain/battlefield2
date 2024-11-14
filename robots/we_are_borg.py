"""
This bot is a combination of my initial idea to analyze groups of bots with a maximum of one empty space between each member of a group as 'gloms' and some iterative improvements over mitch84's retreat_walk2 bot movement style (group coordination, avoid friendly fire, multi-unit retreats). 

The idea behind the glom concept is to generally try to group up bots in such a way that they form a sort of lattice structure where lesser enemies may throw themselves into the gaps between individual bots, only to be immediatly surrounded and pummeled to death. Also, to intentionally surround and pummel smaller groups to death.

The overall strategy at this point is to glom together a group big enough to take the center, then fight defensively to annoy the opponent to death. The advanced movement rules allow for strong offensive play by the smaller gloms, as well. Reliably beats crw_preempt, centerrr, black_magic, and rule99 per Garage testing.

Given that this bot is the result of an iterative improvement process, its code is ugly as hell. You have my sympathies in the form of copious comments
-- entropicdrifter
"""

from functools import lru_cache


gloms = dict()
glom_lookup = dict()
target_ids = dict()
claimed_locations = set()
movement_map = dict()
center = Coords(9, 9)
biggest_glom = None

# There's a bug in The Garage that causes coords.direction_to to malfunction
# This is a workaround method for accurate Garage battles
def direction_to(coords, target):
    other_coords = target
    if coords[1] > other_coords[1]:
        return Direction.North
    elif coords[0] > other_coords[0]:
        return Direction.West
    elif coords[0] < other_coords[0]:
        return Direction.East
    elif coords[1] < other_coords[1]:
        return Direction.South

# sort team by shortest walking distance to unit
# using health as a tie-breaker so we finish off the weak first
def nearest(team, unit):
    if team:
        return min(team, key=lambda e: e.coords.walking_distance_to(unit.coords) + 0.1 * e.health)

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

@lru_cache
def total_threat(state, unit_coord):
    threat = 0
    for direction in Direction:
        obj = state.obj_by_coords(unit_coord + direction)
        if obj and obj.team and obj.team == state.other_team:
            threat += obj.health
    return threat
    
# Pre-calculate gloms and targets, plus initialize the globals
def init_turn(state):
    global gloms
    global glom_lookup
    global claimed_locations
    global center
    global biggest_glom
    global movement_map
    allies = state.objs_by_team(state.our_team)
    enemies = state.objs_by_team(state.other_team)
    unglommed = set(allies + enemies)
    center = Coords(9, 9)
    gloms = dict()
    glom_lookup = dict()
    claimed_locations = set()
    movement_map = {bot.id : None for bot in allies}
    
    # This is where the magic happens. 
    # Recursively look at nearby bots who haven't been added to a glom yet
    # and group them up by team.
    def glom_neighbors(bot, glom, unglommed):
        glom['bots'].add(bot)
        glom['health'] += bot.health
        
        neighbors = [e for e in unglommed if e and e.coords.walking_distance_to(bot.coords) < 3]
        for neighbor in neighbors:
            if neighbor.team == bot.team and neighbor in unglommed:
                unglommed.remove(neighbor)
                glom_lookup[neighbor.id] = glom_lookup[bot.id]
                glom_neighbors(neighbor, glom, unglommed)
        
        return glom
     
    # Glom up all bots
    while len(unglommed):
        current = unglommed.pop()
        
        if not glom_lookup.get(current.id):
            glom_lookup[current.id] = len(gloms)
        
        gloms[glom_lookup[current.id]] = glom_neighbors(current, {'bots': set(), 'health': 0}, unglommed)
        if(current.team == state.our_team):
            # Pick a target for each glom
            target_ids[glom_lookup[current.id]] = nearest(enemies, current).id
    biggest_glom = max(gloms, key=lambda e: len(gloms[e]['bots']) + 0.01 * gloms[e]['health'])

# Credit to mitch84 for writing the walk_to and retreat_inner methods for the bot retreat_walk2
# The only things I've changed here are to check for movement collisions
# between my own bots using the claimed_locations global set
# and to add one param to the retreat_inner method so I can check
# whether a unit would retreat after making its planned movement
def walk_to(state, from_coords, to_coords):
    x = to_coords.x - from_coords.x
    y = to_coords.y - from_coords.y

    xdir = Direction.East if x > 0 else Direction.West
    ydir = Direction.South if y > 0 else Direction.North

    blanks = []
    for direction in Direction:
        possible_location = from_coords + direction
        other = state.obj_by_coords(possible_location)
        if (not other or (other.team == state.our_team and movement_map[other.id])) and (state.turn % 10 < 7 or not check_spawn(possible_location)):
            blanks.append(direction)
            

    if abs(x) > abs(y) and xdir in blanks and from_coords + xdir not in claimed_locations:
        return xdir
    elif abs(y) >= abs(x) and ydir in blanks and from_coords + ydir not in claimed_locations:
        return ydir
    elif xdir in blanks and from_coords + xdir not in claimed_locations:
        return xdir
    elif ydir in blanks and from_coords + ydir not in claimed_locations:
        return ydir
    elif blanks and from_coords + blanks[0] not in claimed_locations:
        return blanks[0]
    else:
        return None

def retreat(state, unit):
    threat = total_threat(state, unit.coords)
    result = retreat_inner(state, unit, unit.coords, threat)
    if not result and not result is None:
        for direction in Direction:
            new_location = unit.coords + direction
            other = state.obj_by_coords(new_location)
            if other and other.team == state.our_team:
                for d in Direction:
                    ally_direction = try_move(state, other, other.coords + d, 'coordinated retreat')
                    if ally_direction:
                        return direction
    return result

def retreat_inner(state, unit, coords, threat):
    enemies = []
    blanks = []
    for direction in Direction:
        new_location = coords + direction
        new_threat = total_threat(state, new_location)
        other = state.obj_by_coords(new_location)
        if other and other.team == state.other_team:
            enemies.append(other)
        # the 'or new_location == unit.coords' bit is necessary 
        # because it allows us to check hypotheticals, 
        # i.e. 'what if I go here, would I retreat?'
        # In those cases, it's necessary to treat the unit's current spot
        # as empty since they won't be there on the turn we're considering.
        # This comparison is consistently safe because 
        # if unit.coords == coords, then new_location will 
        # never == unit.coords
        elif (not other or new_location == unit.coords or (other.team == unit.team and movement_map[other.id])) and new_location not in claimed_locations and (state.turn % 10 < 7 or not check_spawn(new_location)) and new_threat < threat:
            blanks.append((direction, new_threat))
    # retreat if flanked and there is room to retreat
    if len(enemies) > 1:
        if blanks:
            return min(blanks, key=lambda blank: blank[1])[0]
        return blanks
    elif len(enemies) == 1 and enemies[0].health > unit.health:
        if blanks:
            return min(blanks, key=lambda blank: blank[1])[0]
        return blanks
    else:
        return None

# Make a movement action only if we wouldn't retreat immediately after,
# assuming enemies don't move
def try_move(state, unit, target_location, action_name):
    move_direction = walk_to(state, unit.coords, target_location)
    if move_direction:
        new_location = unit.coords + move_direction
        future_threat = total_threat(state, new_location)
        would_retreat = retreat_inner(state, unit, new_location, future_threat)
        if not would_retreat:
            debug.inspect('action', action_name)
            claimed_locations.add(unit.coords + move_direction)
            movement_map[unit.id] = move_direction
            return move_direction

# Make an attack only if it will not hit any allies
def try_attack(state, unit, attack_direction, action_name):
    attack_location = unit.coords + attack_direction
    obj = state.obj_by_coords(attack_location)
    if (not obj and not attack_location in claimed_locations) or (obj and (not obj.team or obj.team == state.other_team)):
        debug.inspect('action', action_name)
        claimed_locations.add(attack_location)
        return Action.attack(attack_direction)

def robot(state, unit):
    
    if movement_map[unit.id]:
        debug.inspect('action', 'coordinated retreat')
        return Action.move(movement_map[unit.id])
    
    allies = state.objs_by_team(state.our_team)
    enemies = state.objs_by_team(state.other_team)
    glom_id = glom_lookup[unit.id]
    glom = gloms[glom_id]
    we_big = glom_id == biggest_glom    
    
    
    # find the target
    enemy = state.obj_by_id(target_ids[glom_id])
    # get the target's glom
    enemy_glom = gloms[glom_lookup[enemy.id]]
    # actually, let's target the closest member of the target glom
    enemy = nearest(enemy_glom['bots'], unit)
    # can't be blind to the *actual* nearest enemy
    nearest_enemy = nearest(enemies, unit)
    if not enemy:
        enemy = nearest_enemy
    
    attack_direction = direction_to(unit.coords, enemy.coords)
    
    debug.inspect('glom', glom)
    debug.inspect('target', enemy)
    
    
    # Take any free wins
    if unit.coords.walking_distance_to(nearest_enemy.coords) == 1 and nearest_enemy.health < unit.health:
        attack_direction = direction_to(unit.coords, nearest_enemy.coords)
        attack = try_attack(state, unit, attack_direction, 'finishing blow')
        if attack:
            return attack
        attack_direction = direction_to(unit.coords, enemy.coords)
    
    retreat_direction = retreat(state, unit)
    # Cut your losses, if applicable
    if retreat_direction:
        direction = try_move(state, unit, unit.coords + retreat_direction, 'retreat')
        if direction:
            return Action.move(direction)
    
    # If our glom is bigger or stronger, kill 'em all
    elif len(enemy_glom['bots']) < len(glom['bots'])  or glom['health'] > enemy_glom['health']:
        
        # If you're alone or hogging the center: fight defensively, 
        # otherwise go on the offensive
        if (len(glom['bots']) > 1 and unit.coords.walking_distance_to(enemy.coords) == 1) or ((we_big or len(glom['bots']) == 1) and unit.coords.walking_distance_to(enemy.coords) <= 2):
            attack = try_attack(state, unit, attack_direction, 'attack')
            if attack:
                return attack
        
        if we_big:
            # If we're the biggest glom, take the center
            direction = try_move(state, unit, center, 'take the center')
            if direction:
                return Action.move(direction)
        else:
            # dynamic movement allows us to consistently surround them
            direction = try_move(state, unit, enemy.coords, 'swarm')
            if direction:
                return Action.move(direction)
    
    if not we_big:
        # If we're not retreating or attacking, try to glom
        ally = nearest([ally for ally in allies if ally.id != unit.id and glom_lookup[ally.id] != glom_id], unit)
        if ally:
            debug.inspect('ally', ally)
            direction = try_move(state, unit, ally.coords, 'glomming')
            if direction:
                return Action.move(direction)
    
    # If we can't find a glom to join, take the center
    direction = try_move(state, unit, center, 'take the center')
    if direction:
        return Action.move(direction)
    
    # If we got here, we're probably trapped. Fight to the death!
    attack_direction = direction_to(unit.coords, nearest_enemy.coords)
    attack = try_attack(state, unit, attack_direction, 'fight to the death')
    if attack:
        return attack
    
