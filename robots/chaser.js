/**
 * Originally from https://github.com/robot-rumble/builtin-bots/blob/main/LICENSE
 * By Anton Outkine
 *
 * robot-rumble/builtin-bots is licensed under the
 * GNU General Public License v3.0
 *
 */

function robot(state, unit) {
  enemies = state.objsByTeam(state.otherTeam)
  closestEnemy = _.minBy(enemies,
    e => e.coords.distanceTo(unit.coords)
  )
  direction = unit.coords.directionTo(closestEnemy.coords)

  if (unit.coords.distanceTo(closestEnemy.coords) === 1) {
    return Action.attack(direction)
  } else {
    return Action.move(direction)
  }
}
