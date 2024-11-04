/**
 * Originally from https://github.com/robot-rumble/builtin-bots/blob/main/LICENSE
 * By Anton Outkine
 *
 * robot-rumble/builtin-bots is licensed under the
 * GNU General Public License v3.0
 *
 */

function robot(state, unit) {
    const direction = Math.random() * 4;
    if (direction < 1) {
        return Action.move(Direction.East)
    } else if (direction < 2) {
        return Action.move(Direction.West)
    } else if (direction < 3) {
        return Action.move(Direction.North)
    } else {
        return Action.move(Direction.South)
    }
}
