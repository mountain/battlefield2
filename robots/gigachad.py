"""
This bot is a combination of my initial idea to analyze groups of bots with a maximum of one empty space between each member of a group as 'gloms' and some iterative improvements over mitch84's retreat_walk2 bot movement style (group coordination, avoid friendly fire, recursive multi-unit retreats). 

The idea behind the glom concept is to generally try to group up bots in such a way that they form a sort of lattice structure where lesser enemies may throw themselves into the gaps between individual bots, only to be immediatly surrounded and pummeled to death. Also, to intentionally surround and pummel smaller groups to death.

The overall strategy at this point is to glom together a group big enough to take the center, then fight defensively to annoy the opponent to death. The advanced movement rules allow for strong offensive play by the smaller gloms, as well. Reliably beats we-are-borg, crw_preempt, centerrr, black_magic, rule99, and chaos_legion, per Garage testing. At this point this is likely the strongest bot on the public board.

-- entropicdrifter
"""

from rumblelib import *
from typing import Any, Dict, Iterable, List, Optional, Tuple, MutableSet, Union
from functools import total_ordering, lru_cache, wraps
from itertools import chain
import weakref


def weak_lru(*lru_args, **lru_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @wraps(func)
            @lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


biggest_glom = None

ATTACK_DAMAGE = 1


@total_ordering
class GlommedBot:
    def __init__(self, bot, glom, state):
        self.bot = bot
        self.glom: Glom = glom
        self.health = bot.health
        self.coords: Coords = bot.coords
        self.team: Team = bot.team
        self.id = bot.id
        self.old_coords: Optional[Coords] = None
        self.is_enemy = bot.team == state.other_team
        self.allies = []
        self.nearest_enemy = None
        self.move: Optional[Action] = None
        self.move_type = None
        self.move_history = []
        self.targeting_coords: Optional[Coords] = None
        self.displaced = None
        self.lattice = False
        self.reason: Optional[str] = None

    def __str__(self):
        return f"""<id:{self.id}\n
            health:{self.health}\n
            coords:{self.coords}\n
            old_coords:{self.old_coords}\n
            team:{self.team}\n
            threat:{self.current_threat()}\n
            nearest_enemy:{self.nearest_enemy.id if self.nearest_enemy else None}>"""

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.id) + hash(global_state.turn)

    def __lt__(self, other):
        if isinstance(other, GlommedBot):
            if not spawn_safe and check_spawn(self.coords):
                if check_spawn(other.coords):
                    return self.health > other.health
                else:
                    return True
            if self.health < other.health:
                return True
            return self.current_threat() > other.current_threat()
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, GlommedBot):
            return self.id == other.id
        return NotImplemented

    def about_to_die(self):
        return (
            (not spawn_safe) and check_spawn(self.coords)
        ) or self.health <= self.count_adj_enemies() * ATTACK_DAMAGE

    def set_target(self, target):
        self.target = target

    def set_location(self, coords, speculation=False):
        if coords and (
            not in_bounds(coords)
            or self.old_coords
            and self.old_coords.walking_distance_to(coords) > 1
        ):
            return False
        if not coords:
            if self.old_coords:
                coords = self.old_coords
            else:
                coords = self.coords
        if self.old_coords == coords:
            self.old_coords = None
        elif coords and not self.old_coords:
            self.old_coords = self.coords
        if self.coords != coords:
            if not speculation:
                apply_threat(self, self.coords, reverse=True)
            on_board = board_state[self.coords.x][self.coords.y]
            # Don't change board_state for speculation
            if on_board == self and not speculation:
                if self.displaced and not (
                    self.displaced.coords.x == self.coords.x
                    and self.displaced.coords.y == self.coords.y
                ):
                    # If the displaced bot has already moved, don't worry about putting it back on the board
                    self.displaced = None
                if not self.displaced and self.coords in claimed_locations:
                    claimed_locations.discard(self.coords)
                board_state[self.coords.x][self.coords.y] = self.displaced
            elif on_board and on_board.displaced == self and not speculation:
                on_board.displaced = None
            apply_threat(self, coords)
            self.coords = coords
        existing_bot = board_state[coords.x][coords.y]
        if existing_bot:
            self.displaced = existing_bot
        if not speculation:
            board_state[coords.x][coords.y] = self
        return True

    def remove_from_glom(self):
        self.glom.remove_bot(self)

    @weak_lru(maxsize=4)
    def direction_to(self, target, targeting=False):
        other_coords: Coords = target.coords
        coords: Coords = self.coords
        # If the target is approaching and the location is valid, assume that's where they'll be,
        # otherwise assume they're at old_coords
        if target.old_coords:
            if not target.targeting_coords:
                set_targeting_coords(self, target)
            other_coords = target.targeting_coords

        distance = coords.walking_distance_to(other_coords)

        disallowed_directions = set()
        # Don't ever try to attack in the direction of an ally
        if targeting:
            for d in Direction:
                loc = coords + d
                other = board_state[loc.x][loc.y]
                if other and other.team == self.team:
                    disallowed_directions.add(d)

        def most_vulnerable_direction(d: Direction) -> float:
            coords_at_d = coords + d
            return threat_map[self.team.opposite][coords_at_d.x][coords_at_d.y]

        best_direction = None
        best_direction_dist = distance
        for direction in sorted(
            (d for d in Direction if d not in disallowed_directions),
            key=most_vulnerable_direction,
        ):
            loc = coords + direction
            direction_dist = loc.walking_distance_to(other_coords)
            if loc == other_coords or direction_dist < best_direction_dist:
                best_direction = direction
                best_direction_dist = direction_dist
        return best_direction

    # sort team by shortest walking distance to unit
    # using current_threat or health as a tie-breaker so we finish off the weak first
    @weak_lru(maxsize=8)
    def nearest(
        self,
        team: Tuple[Union["GlommedBot", Obj]],
        weakest=True,
        other_team: Optional[Tuple["GlommedBot"]] = None,
    ):
        if team and isinstance(team[0], GlommedBot):
            team = map(lambda e: set_targeting_coords(self, e), team)

            def nearest_targeting_coords(e: GlommedBot) -> float:
                new_distance = e.coords.walking_distance_to(self.coords)
                movement_modifier = 0
                if e.old_coords:
                    # Prioritize attackers and try not to hit retreating opponents
                    old_distance = e.old_coords.walking_distance_to(self.coords)
                    if new_distance > old_distance:
                        # running away, de-prioritize
                        movement_modifier = 0.1
                    else:
                        # approaching or holding still
                        movement_modifier = -0.1
                threat_modifier = (
                    0.01 * e.health
                    if weakest
                    else -0.01 * (e.health - e.current_threat())
                )
                attacking_modifier = (
                    -0.5
                    if e.move
                    and e.move.type == ActionType.Attack
                    and e.coords + e.move.direction == self.coords
                    else 0
                )
                retreating_ally_modifier = 0
                if other_team:
                    for ally in other_team:
                        if (
                            ally.reason
                            and ally.move
                            and ally.move.type == ActionType.Move
                            and ally.nearest_enemy
                            and ally.nearest_enemy.id == e.id
                        ):
                            e_coords = e.old_coords if e.old_coords else e.coords
                            if e_coords.walking_distance_to(ally.old_coords) == 1:
                                retreating_ally_modifier = -0.02
                                e.targeting_coords = ally.old_coords
                return (
                    e.targeting_coords.walking_distance_to(self.coords)
                    + threat_modifier
                    + movement_modifier
                    + retreating_ally_modifier
                    + attacking_modifier
                )

            return min(
                team,
                key=nearest_targeting_coords,
                default=None,
            )
        else:
            return min(
                team,
                key=lambda e: e.coords.walking_distance_to(self.coords)
                + (0.1 * e.health if weakest else -0.1 * e.health),
                default=None,
            )

    @weak_lru(maxsize=4)
    def get_move_rating(self, d: Direction, target_location=None):
        new_location = self.coords + d if not self.old_coords else self.old_coords + d
        rating = (
            new_location.walking_distance_to(target_location)
            + 0.1 * self.total_threat(new_location)
            if target_location  # If there's no target location, retreat mode
            else self.total_threat(new_location)
        )
        other = board_state[new_location.x][new_location.y]
        if other:
            if other.team == self.team:
                rating += 0.01
        nearby_allies = self.get_adj(new_location, team=self.team, distance=2)
        nearest_ally = self.nearest(tuple(nearby_allies))
        if nearest_ally and nearest_ally.coords.walking_distance_to(self.coords) == 2:
            rating -= 0.02  # Bias towards being exactly 2 walking distance from the nearest ally
        rating -= self.enemy_threat(new_location) * 0.001
        return rating

    # sort possible moves by shortest walking distance to target location
    # using threat level as a tie-breaker so we avoid pointless danger
    def best_move(
        self, moves: List[Direction], target_location: Coords
    ) -> Optional[Direction]:
        return min(
            moves,
            key=lambda move: self.get_move_rating(
                move, target_location=target_location
            ),
            default=None,
        )

    def best_retreat_move(self, moves: List[Direction]) -> Optional[Direction]:

        return min(moves, key=self.get_move_rating, default=None)

    def get_adj(
        self, location: Coords, team: Optional[Team] = None, distance=1, targeting=False
    ) -> List[Union["GlommedBot", Obj]]:
        coords = location
        distances = range(1, distance + 1)
        positions: Iterable[Coords] = chain.from_iterable(
            positions_n_away(coords, d) for d in distances
        )
        on_board_at_position: Iterable[Optional[GlommedBot]] = (
            get_from_board(position) for position in positions
        )
        others: Iterable[GlommedBot] = (
            other for other in on_board_at_position if other and other != self
        )
        adj = [
            other
            for other in others
            if (not team or other.team == team)
            and (
                not targeting
                or not other.old_coords
                or other.coords.walking_distance_to(self.coords) < distance + 1
                or other.old_coords.walking_distance_to(self.coords)
                < other.coords.walking_distance_to(self.coords)
            )
        ]
        return adj

    @weak_lru()
    def get_adj_enemies(
        self, location: Coords, distance=1, targeting=False
    ) -> List["GlommedBot"]:
        return self.get_adj(
            location, team=self.team.opposite, distance=distance, targeting=targeting
        )

    def strongest_adj_enemy(self):
        enemies = self.get_adj_enemies(self.coords)
        if enemies:
            return max(enemies, key=lambda e: e.health)
        return None

    def total_threat(self, coords) -> float:
        return threat_map[self.team][coords.x][coords.y]

    def enemy_threat(self, coords) -> float:
        return threat_map[self.team.opposite][coords.x][coords.y]

    def current_threat(self) -> float:
        return self.total_threat(self.coords)

    def count_adj_enemies(self, location=None, distance=1, speculate=False) -> int:
        if not location:
            location = self.coords
        return len(
            self.get_adj_enemies(
                location=location, distance=distance, targeting=speculate
            )
        )

    def get_nearest_enemy(
        self,
        defensive: bool = True,
        close_enemies: Iterable[Union[Obj, "GlommedBot"]] = None,
    ):
        if not close_enemies:
            close_enemies = self.get_adj_enemies(self.coords, distance=5)
            return (
                self.nearest(tuple(close_enemies), weakest=defensive)
                if close_enemies
                else None
            )
        else:
            close_allies = tuple(
                set_targeting_coords(self, a)
                for a in self.get_adj(self.coords, team=self.team, distance=2)
            )
            return (
                self.nearest(
                    tuple(close_enemies), weakest=defensive, other_team=close_allies
                )
                if close_enemies
                else None
            )

    @weak_lru()
    def count_adj_walls(self) -> int:
        walls = 0
        for loc in self.coords.coords_around():
            if not in_bounds(loc):
                walls += 1
        return walls

    def is_surrounded(self) -> bool:
        return self.count_adj_enemies() + self.count_adj_walls() == 4

    @weak_lru()
    def about_to_be_surrounded(self) -> bool:
        potentially_surrounding_units = self.get_adj(self.coords, distance=2)
        walls = self.count_adj_walls()
        safe_sides = []
        for ally in (
            unit for unit in potentially_surrounding_units if unit.team == self.team
        ):
            if ally.coords.walking_distance_to(self.coords) == 1:
                safe_sides.append(self.coords.direction_to(ally.coords))
        potentially_surrounding_enemies = len(
            [
                unit
                for unit in potentially_surrounding_units
                if unit.team == self.team.opposite
            ]
        )
        return not safe_sides and walls + potentially_surrounding_enemies >= 4

    def attack_of_opportunity(
        self, defensive=False, fight_to_the_death=False
    ) -> Optional[Action]:
        if (
            global_state.turn % 10 > 4
            and check_spawn(self.coords)
            and (not self.is_surrounded() and not fight_to_the_death)
        ):  # try to get away from spawn zones
            return None
        close_enemies = self.get_adj_enemies(self.coords, distance=2, targeting=True)
        good_guy = self.team == global_state.our_team
        about_to_be_surrounded = self.about_to_be_surrounded()
        # Take any free wins if they won't retreat
        while close_enemies:
            nearest_enemy: GlommedBot = self.get_nearest_enemy(
                close_enemies=close_enemies,
            )
            nearest_enemy_dist = self.coords.walking_distance_to(
                nearest_enemy.targeting_coords
            )
            enemy_health = nearest_enemy.health
            # When defending, attempt to kite the enemy
            if (
                (defensive or (self.health <= enemy_health)) and nearest_enemy_dist < 3
            ) or (nearest_enemy_dist == 1):
                attack_direction = self.direction_to(nearest_enemy, targeting=True)
                if attack_direction:
                    debug_log(self, "nearest_enemy", nearest_enemy)
                    attack = try_attack(
                        self,
                        attack_direction,
                        (
                            "attack of opportunity"
                            if not defensive
                            else "fight to the death"
                        ),
                        make_lattice=good_guy
                        and not defensive
                        and nearest_enemy_dist <= 2
                        and self.health > 1,
                    )
                    if attack:
                        return attack
            close_enemies.remove(nearest_enemy)

    def take_the_center(
        self,
        fearless=False,
        displace=False,
        speculation=False,
        recursion_parent: Optional["GlommedBot"] = None,
    ) -> Optional[Action]:
        if self.coords != center:
            # If we can't find a glom to join, take the center
            direction = try_move(
                self,
                center,
                "take the center",
                fearless=fearless,
                displace=displace,
                speculation=speculation,
                recursion_parent=recursion_parent,
            )
            if direction:
                if self.health > 1:
                    lattice(self)
                return Action.move(direction)

    def set_move(
        self, move: Optional[Action], source="Unset! Please fix", speculation=False
    ):
        success = False
        if not move:
            success = self.unset_move()
        else:
            if self.move:
                if str(move) == str(self.move):
                    history_entry = self.move_history[-1]
                    history_entry[2] = f"{source}, {history_entry[2]}"
                    self.move_history[-1] = history_entry
                    return True
                else:
                    success = False
            else:
                if move.type == ActionType.Move:
                    if not speculation:
                        claimed_locations.add(self.coords + move.direction)
                    success = self.set_location(
                        self.coords + move.direction, speculation
                    )
                    if success and self.health == 1 and "retreat" in source:
                        apply_threat(
                            self, self.coords, reverse=True, value=0.8
                        )  # 80% sure
                elif move.type == ActionType.Attack:
                    if not speculation:
                        claimed_locations.add(self.coords + move.direction)
                        claimed_locations.add(self.coords)
                    apply_attack_threat(self, self.coords + move.direction)
                    success = self.set_location(self.coords, speculation)
        self.move_history.append([success, move, source])
        if success:
            moves[self.id] = move
            self.move = move
            return True
        return False

    def unset_move(self, speculation=False) -> bool:
        current_move: Action = self.move
        if not current_move:
            return False
        good_guy = self.team == global_state.our_team
        # If the current unit has a move already set, and None was passed in as the new move,
        # That means we should un-set the move.
        if good_guy and current_move.type == ActionType.Attack:
            move_coords = self.coords + current_move.direction
            if self.lattice:
                # Undo latticing to maximize team maneuverability
                for d in Direction:
                    coords = self.coords + d
                    adj_allies: List[GlommedBot] = self.get_adj(coords, self.team)
                    at_coords = get_from_board(coords)
                    if at_coords or adj_allies:
                        # Skip unclaiming locations already claimed by allies
                        if at_coords and at_coords.team == self.team:
                            continue
                        for ally in adj_allies:
                            if ally.move and ally.move.type == ActionType.Attack:
                                if (ally.coords + ally.move.direction) == coords:
                                    continue
                    claimed_locations.discard(coords)
                self.lattice = False
            elif move_coords in claimed_locations:
                claimed_locations.discard(move_coords)
            if self.coords in claimed_locations:
                claimed_locations.discard(self.coords)
            return True
        elif self.old_coords:
            return self.set_location(self.old_coords, speculation=speculation)

    @weak_lru(maxsize=5)
    def would_retreat(self, threat: float, location: Coords = None) -> Optional[str]:
        if not location:
            location = self.coords
        if not spawn_safe and check_spawn(location):
            return "Avoiding spawn location"
        if (
            location != self.coords
            and self.health <= ATTACK_DAMAGE
            and threat > self.current_threat()
        ):
            return "Super Weenie Hut Junior's"
        local_enemies = [
            enemy
            for enemy in self.get_adj_enemies(location, distance=2)
            if location.walking_distance_to(enemy.coords) < 2
            or enemy.old_coords
            and enemy.old_coords.walking_distance_to(self.coords) < 2
            # Don't assume enemy will retreat
        ]
        scariest = min(local_enemies, key=lambda e: e.current_threat(), default=None)
        if threat >= self.health:
            return "High threat"
        if len(local_enemies) >= 3 or self.about_to_be_surrounded():
            return "Surrounded/imminent death"
        if scariest:
            if (
                threat > scariest.current_threat()
                or self.get_adj_enemies(location)
                > scariest.get_adj_enemies(scariest.coords)
            ) and (scariest.health >= self.health or location != self.coords):
                return "Weenie mode"

    def could_die(self, new_location: Coords) -> bool:
        return len(self.get_adj_enemies(new_location)) >= self.health / ATTACK_DAMAGE


@total_ordering
class Glom:
    def __init__(self):
        self.bots = set()
        self.health = 0
        self.enemy_hp = 0
        self.target = None
        self.team = None

    def __str__(self):
        return (
            "\n\tbots: "
            + str(self.bots)
            + "\n\t"
            + "health: "
            + str(self.health)
            + "\n\tenemy_hp:"
            + str(self.enemy_hp)
        )

    def __lt__(self, other):
        if isinstance(other, Glom):
            return (
                len(self.bots) + 0.0001 * self.health
                < len(other.bots) + other.health * 0.0001
            )

    def __eq__(self, other):
        if isinstance(other, Glom):
            return (
                len(self.bots) + 0.0001 * self.health
                == len(other.bots) + other.health * 0.0001
            )

    def set_target(self, target):
        if target:
            self.target = target

    def add_bot(self, bot):
        self.bots.add(bot)
        self.health += bot.health
        if not self.team:
            self.team = bot.team

    def remove_bot(self, bot):
        if bot in self.bots:
            self.bots.remove(bot)
            self.health -= bot.health
            if self.target and isinstance(self.target, GlommedBot) and self.target.glom:
                self.target.glom.enemy_hp -= bot.health

    @weak_lru(maxsize=1)
    def weak_point(self):
        if len(self.bots):
            return max(self.bots, key=lambda bot: bot.health - bot.current_threat())
        return None

    def aim_at_weak_point(self):
        if len(self.bots) and self.target and self.target.glom:
            self.set_target(self.target.glom.weak_point())

    @weak_lru(maxsize=1)
    def we_big(self):
        return self == biggest_glom


claimed_locations: MutableSet = set()
board_state = None
center = None
moves: Dict[str, Action] = dict()
debug_info: Dict[str, Dict[str, str]] = None
threat_map: Dict[Team, Dict[int, Dict[int, float]]] = dict()
spawn_safe = None
global_state: State

# Credit for this method to atl15, taken from the bot centerrr
# checking if bot is in spawn
spawn_coords = SPAWN_COORDS


def check_spawn(unit_coord: Union[Coords, Tuple[int, int]]):
    if isinstance(unit_coord, Coords):
        return unit_coord.is_spawn()
    else:
        return Coords(*unit_coord).is_spawn()


def in_bounds(coord):
    state = global_state
    safe = (
        coord[0] > 0 and coord[0] < 18 and coord[1] > 0 and coord[1] < 18 and coord[0]
    )
    if safe:
        obj = state.obj_by_coords(coord)
        safe = not obj or obj.obj_type != ObjType.Terrain
    return safe


def debug_log(unit, key: str, value):
    if unit.team == global_state.our_team and key and value:
        if debug_info.get(unit.id) and debug_info[unit.id].get(key):
            existing_value = debug_info[unit.id].get(key)
            debug_info[unit.id][key] = f"{existing_value}, {value}"
        else:
            debug_info[unit.id][key] = value


def set_targeting_coords(unit: GlommedBot, enemy: GlommedBot, enemy_retreat: Action | None = None):
    if not enemy.old_coords:
        if enemy_retreat:
            enemy.targeting_coords = enemy.coords + enemy_retreat.direction
        else:
            enemy.targeting_coords = enemy.coords
    else:
        if enemy.coords.walking_distance_to(
            unit.coords
        ) > enemy.old_coords.walking_distance_to(unit.coords):
            # If they should be retreating, make them pay if they fail to do so
            enemy.targeting_coords = enemy.old_coords
        else:
            # If they're approaching, sucker punch
            enemy.targeting_coords = enemy.coords
    return enemy


# Returns a list of all in-bound Coords that are exactly n manhattan distance away from the given Coord
def positions_n_away(pos, n) -> Iterable[Coords]:
    dx_dir = 1
    dy_dir = 1
    dx = n
    dy = 0
    for _ in range(4 * n):
        new_pos = Coords(pos.x + dx, pos.y + dy)
        if in_bounds(new_pos):
            yield new_pos

        if dy == 0:
            dx_dir *= -1
        if dx == 0:
            dy_dir *= -1
        dx += dx_dir
        dy += dy_dir


def get_from_board(coords: Coords) -> Optional[GlommedBot]:
    other = board_state[coords.x][coords.y]
    if other and (other.coords.x != coords.x or other.coords.y != coords.y):
        # Fix incorrect board_state if you find it
        if other.displaced and other.displaced.coords == coords:
            board_state[coords.x][coords.y] = other.displaced
        else:
            board_state[coords.x][coords.y] = None
        board_state[other.coords.x][other.coords.y] = other
        other = get_from_board(coords)
    return other


# Pre-calculate gloms and targets, plus initialize the globals, then pre-calculate all moves
def init_turn(state: State):
    global gloms
    global claimed_locations
    global center
    global board_state
    global moves
    global debug_info
    global threat_map
    global spawn_safe
    global biggest_glom
    global global_state
    allies = state.objs_by_team(state.our_team)
    enemies = state.objs_by_team(state.other_team)
    unglommed = set(allies + enemies)
    center = Coords(9, 9)
    gloms: MutableSet[Glom] = set()
    claimed_locations = set()
    board_state: Dict[int, Dict[int, Optional[GlommedBot]]] = {
        x: {y: None for y in range(19)} for x in range(19)
    }
    moves = dict()
    threat_map = {
        team: {x: {y: 0.0 for y in range(19)} for x in range(19)} for team in Team
    }
    turn_number_out_of_10 = state.turn % 10
    spawn_safe = turn_number_out_of_10 != 0
    spawn_threat = 15 if not spawn_safe else turn_number_out_of_10 / 4
    global_state = state
    debug_info = {unit.id: dict() for unit in chain(allies, enemies)}
    if not len(enemies):
        glom = Glom()
        biggest_glom = glom
        moves = {unit.id: choose_move(GlommedBot(unit, glom, state)) for unit in allies}
        return
    for coords in spawn_coords:
        apply_threat(
            coords=coords,
            value=spawn_threat,
            exact=not spawn_safe,
        )

    moves = dict()
    (allies, enemies) = glom_all(state, unglommed, allies, enemies)

    biggest_glom = max(gloms)
    enemies = sorted(enemies)
    allies = sorted(allies)
    # Set each allied gloms' targets to the weak points of the enemy gloms
    allied_gloms = sorted([glom for glom in gloms if glom.team == state.our_team])
    for glom in allied_gloms:
        if glom == biggest_glom:
            glom.set_target(center)
            continue
        glom.aim_at_weak_point()
        # We only use the gloms to choose who to attack with first
        glom.bots = sorted(glom.bots, reverse=True)

    # Use for debugging threat mapping

    simulate_enemy_turn = False
    if simulate_enemy_turn:
        for enemy in enemies:
            enemy.allies = enemies
            enemy.nearest_enemy = enemy.get_nearest_enemy()
            if not enemy.move:
                move = retreat(enemy, speculation=True)
                if move:
                    enemy.set_move(move, source="enemy_init_retreat", speculation=True)
        for enemy in (e for e in enemies[::-1] if not moves.get(e.id)):
            if not enemy.move:
                move = choose_attack(enemy, speculation=True)
                if move:
                    enemy.set_move(move, source="enemy_init_attack", speculation=True)
        for enemy in enemies:
            if not enemy.move:
                move = choose_move(enemy, speculation=True)
                if move:
                    enemy.set_move(move, source="enemy_init_move", speculation=True)
    # Iterate from weakest to strongest for retreats to prioritize protecting the weakest first
    for ally in allies:
        ally.allies = allies
        ally.nearest_enemy = ally.get_nearest_enemy()
        debug_log(ally, "self", ally)
        if not ally.move:
            move = retreat(ally)
            if move:
                ally.set_move(move, source="init_retreat")
    # Iterate backwards over allies so the strongest units are highest priority for attacks.
    for ally in reversed(allies):
        if not ally.move:
            move = choose_attack(ally)
            if move:
                ally.set_move(move, source="init_attack")
    # Finally, iterate from weakest to strongest for regular troop movements
    # so the weakest end up in the center defended by the strongest
    for ally in allies:
        if not ally.move:
            move = choose_move(ally)
            if move:
                ally.set_move(move, source="init_move")
        debug_log(ally, "move history", ally.move_history)
        if ally.move and not (ally.id in moves.keys()):
            moves[ally.id] = ally.move
    """
    if state.turn % 10 == 1:
        for team in (state.our_team, state.other_team):
            print(f"------{team}------")
            for x in range(1, 18):
                print([threat_map[team][y][x] for y in range(1, 18)])
            print(f"------{team}------")"""


def apply_threat(
    unit=None,
    coords: Optional[Coords] = None,
    reverse=False,
    value: float = 1.0,
    exact=False,
):
    if not coords:
        return
    if not unit:
        teams = [Team.Red, Team.Blue]
    else:
        teams = [unit.team.opposite]
        if unit.health == 1:
            value = value / 2
    if in_bounds(coords):
        if reverse:
            value = value * -1
        value = value * ATTACK_DAMAGE
        for team in teams:
            threat_map[team][coords.x][coords.y] += value
            if exact:
                continue
            for new_location in positions_n_away(coords, 1):
                if in_bounds(new_location):
                    threat_map[team][new_location.x][new_location.y] += (
                        value if unit else value / 2
                    )
            for new_location in positions_n_away(coords, 2):
                if in_bounds(new_location):
                    threat_map[team][new_location.x][new_location.y] += (
                        value / 2 if unit else value / 4
                    )


def apply_attack_threat(unit: GlommedBot, attack_loc: Coords):
    apply_threat(unit, unit.coords, reverse=True, value=0.5)
    threat_map[unit.team.opposite][attack_loc.x][attack_loc.y] += ATTACK_DAMAGE


def revert_attack_threat(unit: GlommedBot, attack_loc: Coords):
    apply_threat(unit, unit.coords, value=0.5)
    threat_map[unit.team.opposite][attack_loc.x][attack_loc.y] -= ATTACK_DAMAGE


def glom_all(
    state, unglommed, allies, enemies
) -> Tuple[List[GlommedBot], List[GlommedBot]]:
    glommed = dict()
    # Glom up all bots
    while len(unglommed):
        current = unglommed.pop()

        glom_unit(state, unglommed, current, glommed, allies, enemies)

    allies_internal = []
    enemies_internal = []
    for bot in glommed.values():
        board_state[bot.coords.x][bot.coords.y] = bot
        if bot.team == state.our_team:
            allies_internal.append(bot)
            apply_threat(bot, bot.coords)
        else:
            enemies_internal.append(bot)
            apply_threat(bot, bot.coords)
    return (allies_internal, enemies_internal)


def glom_unit(state, unglommed, current, glommed, allies, enemies):
    glom = Glom()
    glom_bot = GlommedBot(current, glom, state)
    glommed[current.id] = glom_bot

    glom_neighbors(glom_bot, glom, unglommed, glommed, state)
    gloms.add(glom)
    # Pick a target for each glom
    target_group = enemies if current.team == state.our_team else allies
    target = glom_bot.nearest(tuple(target_group))
    target_glommed = None
    if target and target in unglommed:
        unglommed.remove(target)
        target_glommed = glom_unit(state, unglommed, target, glommed, allies, enemies)
    elif target:
        target_glommed = glommed[target.id]
    if not glom.target and target_glommed:
        glom.set_target(target_glommed)
        target_glommed.glom.enemy_hp += glom.health
    return glom_bot


# This is where the magic happens.
# Recursively look at nearby bots who haven't been added to a glom yet
# and group them up by team.
def glom_neighbors(bot, glom, unglommed, glommed, state):
    glom.add_bot(bot)

    neighbors = [
        e for e in unglommed if e and e.coords.walking_distance_to(bot.coords) < 3
    ]
    for neighbor in neighbors:
        if neighbor.team == bot.team and neighbor in unglommed:
            unglommed.remove(neighbor)
            glommed_neighbor = GlommedBot(neighbor, bot.glom, state)
            glommed[neighbor.id] = glommed_neighbor
            glom_neighbors(glommed_neighbor, glom, unglommed, glommed, state)

    return glom


def retreat(
    unit: GlommedBot, recursion_parent: Optional[GlommedBot] = None, speculation=False
):
    if unit.move or moves.get(unit.id):
        return unit.move if unit.move else moves[unit.id]
    threat = unit.current_threat()
    reason = unit.would_retreat(threat)
    if reason:
        debug_log(unit, "reason", reason)
        unit.reason = reason

        move = try_move(
            unit,
            None,
            "retreating",
            retreat=True,
            recursion_parent=recursion_parent,
            speculation=speculation,
        )
        if move:
            unit.remove_from_glom()
            return Action.move(move)
    return None


def walk_to(
    unit: GlommedBot,
    to_coords: Coords,
    retreating: bool,
    attacking=False,
    displaced=False,
    push=False,
    recursion_parent: Optional[GlommedBot] = None,
    speculation=False,
    fearless=False,
) -> Optional[Direction]:
    if unit.move:
        if unit.move.type == ActionType.Move:
            return unit.move.direction
        else:
            return None
    blanks = []
    others_to_move: Dict[Direction, GlommedBot] = dict()
    need_to_move_this_turn = unit.about_to_die()
    allow_recursion = unit.team == global_state.our_team and (
        retreating or attacking or displaced
    )
    debug_log(unit, "recursion_parent", recursion_parent)
    recursion_parent_direction = (
        recursion_parent.move.direction.opposite if recursion_parent else None
    )
    for direction in Direction:
        if direction == recursion_parent_direction:
            debug_log(
                unit,
                str(direction),
                f"{direction} == recursion_parent_direction",
            )
            debug.locate(unit.bot)
            continue
        possible_location = unit.coords + direction
        if (
            need_to_move_this_turn or possible_location not in claimed_locations
        ) and in_bounds(possible_location):
            other = get_from_board(possible_location)
            if (
                (
                    not other
                    or other.coords != possible_location
                    or (
                        allow_recursion
                        and (
                            (
                                other.team == unit.team
                                and not other.move
                                and (
                                    not other.displaced
                                    and (
                                        not recursion_parent
                                        or not other.id == recursion_parent.id
                                    )
                                )
                            )
                            or (
                                other.team == unit.team.opposite
                                and other.move
                                and other.move.type == ActionType.Move
                                and (retreating or need_to_move_this_turn)
                            )
                        )
                        and (
                            retreating or other.health < unit.health
                        )  # Only displace weaker units when swarming
                    )
                )
                and (spawn_safe or not check_spawn(possible_location))
                and (
                    attacking
                    or unit.count_adj_enemies(location=possible_location) < unit.health
                    or need_to_move_this_turn
                )
            ):
                blanks.append(direction)
                others_to_move[direction] = other
            else:
                debug_log(
                    unit,
                    str(direction),
                    f"other: {other}, possible_location: {possible_location}, allow_recursion: {allow_recursion}, other.move: {other.move if other else None}, other.displaced: {other.displaced if other else None}, retreating: {retreating}, spawn_safe: {spawn_safe}, check_spawn: {check_spawn(possible_location)}, attacking: {attacking}, too many enemies: {unit.count_adj_enemies(location=possible_location) < unit.health}, need_to_move_this_turn: {need_to_move_this_turn}",
                )
        else:
            debug_log(
                unit,
                str(direction),
                f"need_to_move: {need_to_move_this_turn}, location_claimed: {possible_location in claimed_locations}",
            )
    move = None
    if retreating:
        sorter = lambda move: unit.get_move_rating(move)
    else:
        sorter = lambda move: unit.get_move_rating(move, target_location=to_coords)
    blanks.sort(key=sorter)

    debug_log(unit, "blanks", str(blanks))
    debug_log(unit, "fearless", fearless)

    while blanks:
        move = blanks.pop(0)

        debug_log(unit, "move_choice", move)

        ally = others_to_move.get(move)
        if move:
            new_location = (
                unit.coords + move if not unit.old_coords else unit.old_coords + move
            )
            future_threat = unit.total_threat(new_location)
            if (
                fearless
                or not unit.would_retreat(future_threat, location=new_location)
                or (
                    retreat
                    and not future_threat > unit.current_threat()
                    and not unit.count_adj_enemies(location=new_location) >= unit.health
                )
            ):
                can_continue = unit.set_move(
                    Action.move(move), "walk_to attempt", speculation=speculation
                )
                if ally and can_continue:
                    ally_move = ally.move

                    if ally_move or ally.old_coords:
                        # If the ally is already moving away from where we're going, move there
                        if ally.coords != unit.coords:
                            return move
                        else:
                            unit.set_move(
                                None, "walk_to canceled", speculation=speculation
                            )
                            move = None
                            continue

                    ally_move = retreat(ally, recursion_parent=unit)
                    if ally_move and ally.set_move(
                        ally_move, "coordinated retreat", speculation=speculation
                    ):
                        return move

                    ally_move = choose_attack(
                        ally,
                        displace=retreating or displaced,
                        push=push,
                        recursion_parent=unit,
                        speculation=speculation,
                    )
                    if (
                        ally_move
                        and ally_move.type == ActionType.Move
                        and ally.set_move(
                            ally_move, "coordinated attack", speculation=speculation
                        )
                    ):
                        return move

                    ally_move = choose_move(
                        ally,
                        displace=retreating or displaced,
                        push=push,
                        recursion_parent=unit,
                    )
                    if (
                        ally_move
                        and ally_move.type == ActionType.Move
                        and ally.set_move(ally_move, "coordinated move")
                    ):
                        return move

                    unit.set_move(None, "walk_to canceled", speculation=speculation)
                elif can_continue:
                    return move
        move = None

    if not move:
        unit.set_move(None, "walk_to no move found", speculation=speculation)

    return move


def lattice(unit: GlommedBot, attack_direction=None):
    """
    if unit.team == global_state.our_team:
        # Spread out in a lattice
        unit.lattice = True
        for d in (
            dir
            for dir in Direction
            if attack_direction is None or dir != attack_direction.opposite
        ):
            claimed_locations.add(unit.coords + d)"""
    return


# Make a movement action only if we wouldn't retreat immediately after,
# assuming enemies don't move
def try_move(
    unit: GlommedBot,
    target_location,
    action_name,
    retreat=False,
    fearless=False,
    attacking=False,
    displace=False,
    recursion_parent: Optional[GlommedBot] = None,
    speculation=False,
):
    if unit.move and unit.move.type == ActionType.Move:
        return unit.move.direction
    elif unit.move:
        return None
    move_direction = walk_to(
        unit,
        target_location,
        retreat,
        attacking=attacking,
        displaced=displace,
        push=retreat
        and (
            (not spawn_safe and check_spawn(unit.coords))
            or unit.health < unit.current_threat()
        ),
        recursion_parent=recursion_parent,
        speculation=speculation,
        fearless=fearless,
    )
    if move_direction:
        debug_log(unit, "action", action_name)
        return move_direction


# Make an attack only if it will not hit any allies
def try_attack(
    unit: GlommedBot,
    attack_direction: Direction,
    action_name: str,
    make_lattice: bool = False,
) -> Optional[Action]:
    attack_location = unit.coords + attack_direction
    if in_bounds(attack_location):
        obj = board_state[attack_location.x][attack_location.y]
        if not obj or (obj.team and obj.team == unit.team.opposite):
            debug_log(unit, "action", action_name)
            if make_lattice:
                lattice(unit, attack_direction)
            return Action.attack(attack_direction)


def choose_attack(
    unit: GlommedBot,
    displace=False,
    push=False,
    recursion_parent=None,
    speculation=False,
) -> Optional[Action]:
    if unit.move or moves.get(unit.id):
        return unit.move if unit.move else moves[unit.id]
    if unit.about_to_die() and not unit.is_surrounded():
        # Never attack when about to be obliterated
        return None

    nearest_enemy = unit.nearest_enemy
    glom = unit.glom
    we_big = glom.we_big()

    # find the target
    enemy: Union[GlommedBot, Coords] = glom.target
    enemy_glom: Optional[Glom] = None
    if nearest_enemy and unit.coords.walking_distance_to(nearest_enemy.coords) < 3:
        enemy = nearest_enemy
    if isinstance(enemy, GlommedBot):
        enemy_retreat = retreat(enemy, speculation=True)
        set_targeting_coords(unit, enemy, enemy_retreat)
        # get the target's glom
        enemy_glom = enemy.glom
    unit.set_target(enemy)

    debug_log(unit, "target", enemy)

    if not displace:
        aoo = unit.attack_of_opportunity(defensive=unit.health < 3 * ATTACK_DAMAGE)
        if aoo:
            return aoo

    # If our glom is bigger or stronger, kill 'em all
    if enemy_glom and (
        len(enemy_glom.bots) < len(glom.bots)
        or glom.health > enemy_glom.health
        or enemy_glom.enemy_hp > enemy_glom.health
    ):
        enemy_dist = unit.coords.walking_distance_to(enemy.targeting_coords)
        if not displace:
            attack_direction = unit.direction_to(enemy, targeting=True)
            if attack_direction:
                they_scary = enemy_dist == 2 and enemy.total_threat(
                    unit.coords + attack_direction
                ) < unit.total_threat(unit.coords + attack_direction)
                # If you're at a local disadvantage or hogging the center,
                # fight defensively, otherwise go on the offensive and try to surround
                if they_scary or enemy_dist <= 1:
                    attack = try_attack(unit, attack_direction, "attack")
                    if attack:
                        return attack

        if enemy_dist > 1 and (
            (not we_big and unit.health > 1) or unit.health > 2 * ATTACK_DAMAGE
        ):
            # dynamic movement allows us to consistently surround them
            direction = try_move(
                unit,
                enemy.targeting_coords,
                "swarm",
                fearless=push
                and (enemy.current_threat() + 0.5 > enemy.health)
                and unit.health > 3 * ATTACK_DAMAGE,
                attacking=True,
                displace=displace,
                recursion_parent=recursion_parent,
                speculation=speculation,
            )
            if direction:
                return Action.move(direction)


def choose_move(
    unit: GlommedBot,
    displace=False,
    push=False,
    recursion_parent: Optional[GlommedBot] = None,
    speculation=False,
) -> Optional[Action]:
    if unit.move or moves.get(unit.id):
        return unit.move if unit.move else moves[unit.id]
    we_big = unit.glom.we_big()
    about_to_die = unit.about_to_die()
    if not we_big:
        # If we're not retreating or attacking, try to glom
        ally = unit.nearest(
            tuple(
                (
                    a
                    for a in unit.allies
                    if a.glom != unit.glom and a.glom.health > unit.glom.health
                )
            )
        )
        if ally:
            debug_log(unit, "ally", ally)
            direction = try_move(
                unit,
                ally.coords,
                "glomming",
                displace=displace or about_to_die,
                fearless=(push and unit.health > 2 * ATTACK_DAMAGE) or about_to_die,
                recursion_parent=recursion_parent,
                speculation=speculation,
            )
            if direction:
                return Action.move(direction)

    take_center = unit.take_the_center(
        displace=displace or about_to_die,
        fearless=(push and unit.health > 2 * ATTACK_DAMAGE) or about_to_die,
        speculation=speculation,
        recursion_parent=recursion_parent,
    )
    if take_center:
        return take_center
    if not displace and not recursion_parent:
        return unit.attack_of_opportunity(defensive=True, fight_to_the_death=True)


def robot(state, unit):
    if debug_info:
        unit_debug = debug_info.get(unit.id)
        if unit_debug:
            for key, value in unit_debug.items():
                if key and value:
                    debug.inspect(key, value)
    return moves.get(unit.id)
