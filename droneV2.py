"""Quadrotor with a standard cascaded PID controller.

Just the drone and the controller - no genetic algorithm, no neural network,
no scoring. Run it and watch it fly to the green sphere and hold there.

    python droneV2.py

The control stack, outermost first:

    target position
        |                position_pid   - where to go  ->  how to accelerate
        v
    desired acceleration (3-vector)
        |                accel_to_tilt  - a quadrotor cannot push sideways,
        v                                 so it LEANS. tilt = accel / g
    desired roll / pitch / vertical acceleration
        |                attitude_pid   - hold that lean
        v
    thrust + roll torque + pitch torque + yaw torque
        |                mix            - share across four rotors
        v
    four rotor thrusts
"""

import time

import numpy as np
import pybullet as p
import pybullet_data


# ======================= simulation =======================

DT = 1 / 240              # simulated seconds per stepSimulation() call
G = 9.81

MASS = 0.5                # kg
HALF = [0.1, 0.1, 0.02]   # HALF-size per axis -> a 20 x 20 x 4 cm slab
ARM = 0.1                 # centre-to-rotor distance, m
ROTOR_R = 0.05            # drawn radius of each rotor disk (cosmetic only)

# rotor positions in the BODY frame, X-layout, one per corner
ROTORS = [
    [ ARM,  ARM, 0],
    [-ARM,  ARM, 0],
    [-ARM, -ARM, 0],
    [ ARM, -ARM, 0],
]

HOVER = MASS * G / 4      # per-rotor thrust that exactly cancels gravity
F_MAX = 2.5 * HOVER       # per-rotor ceiling; thrust-to-weight 2.5


# ======================= PID gains =======================
#
# Passed in as a dict rather than read from globals, so the controller can be
# driven with any set of numbers without touching its code.
#
# The attitude gains look tiny next to the position ones only because they
# are in N*m while the slab's moment of inertia is 0.00173 kg*m^2. They come
# straight from the second-order response of the body:
#     KP = I * wn^2   = 0.00173 * 20^2 = 0.69    -> wn = 20 rad/s
#     KD = I * 2 * wn = 0.00173 * 40   = 0.069   -> zeta = 1.0, no overshoot
#
# The inner loop is 20x faster than the outer one. That separation is what
# makes a cascade work: attitude settles long before position notices.

GAINS = {
    "kp_xy": 1.0,  "kd_xy": 2.0,      # position  -> acceleration (wn ~ 1)
    "kp_z": 5.0,   "kd_z": 4.0,       # altitude  -> acceleration (wn ~ 2.2)
    "kp_att": 0.7, "kd_att": 0.07,    # roll/pitch -> torque      (wn ~ 20)
    "kp_yaw": 0.3, "kd_yaw": 0.067,   # heading    -> torque
}

TILT_MAX = np.radians(25)   # the most the outer loop may ask for
A_MAX = 5.0                 # largest commanded acceleration, m/s^2


# ======================= world =======================

def setup(gui=True):
    """Build the world and the drone. Returns the body id.

    Call this ONCE - rebuilding the scene is slow. Use reset() to start over.
    """
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    p.setGravity(0, 0, -G)       # pybullet starts with ZERO gravity
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)   # we own the clock: one call = one DT, and
                                 # the same inputs always give the same flight

    if gui:
        # the GUI binds w/g/s to wireframe and overlays - turn that off
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        # so a left-drag orbits the camera instead of grabbing the drone
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    # Only the COLLISION shape sets mass and inertia. The visual below draws
    # the body plus four rotor disks as ONE shape - purely cosmetic, the
    # physics is identical to a bare box.
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=HALF)
    vis = p.createVisualShapeArray(
        # every array must be the same length, hence the unused zeros:
        # a box ignores radii/lengths, a cylinder ignores halfExtents
        shapeTypes=[p.GEOM_BOX] + [p.GEOM_CYLINDER] * 4,
        halfExtents=[HALF] + [[0, 0, 0]] * 4,
        radii=[0] + [ROTOR_R] * 4,
        lengths=[0] + [0.008] * 4,
        rgbaColors=[[0.85, 0.15, 0.15, 1]] + [[0.15, 0.15, 0.18, 0.9]] * 4,
        visualFramePositions=[[0, 0, 0]]
        + [[x, y, HALF[2] + 0.006] for x, y, _ in ROTORS],
    )

    return p.createMultiBody(baseMass=MASS,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[0, 0, 1])


def reset(body, pos):
    """Back to a position, level and motionless.

    resetBaseVelocity is NOT optional - without it the drone keeps whatever
    velocity it had, so a fresh run silently inherits the last one's state.
    """
    p.resetBasePositionAndOrientation(body, list(pos), [0, 0, 0, 1])
    p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])


def make_marker(pos, radius=0.15):
    """A visual-only sphere. No collision shape, so the drone flies through
    it and the physics never sees it."""
    v = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                            rgbaColor=[0, 1, 0, 0.35])
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=v, basePosition=list(pos))


def move_marker(marker, pos):
    """Move the existing marker. Creating one per target would leak bodies."""
    p.resetBasePositionAndOrientation(marker, list(pos), [0, 0, 0, 1])


# ======================= sensing / actuation =======================

def get_state(body):
    """The 12 numbers the controller is allowed to see."""
    pos, quat = p.getBasePositionAndOrientation(body)
    vel, ang_vel = p.getBaseVelocity(body)
    rpy = p.getEulerFromQuaternion(quat)   # quaternion -> roll, pitch, yaw

    return (np.array(pos), np.array(vel),
            np.array(rpy), np.array(ang_vel))


def mix(total_thrust, tx, ty):
    """(total thrust, roll torque, pitch torque) -> four rotor thrusts.

    Each rotor gets an equal share of lift plus a correction for where it
    sits: f_i = F/4 + (tx*y_i - ty*x_i) / (4*ARM^2).

    The corrections sum to zero, so the total stays exactly F - which means
    rolling costs no altitude. That decoupling is the point of mixing.
    """
    return [total_thrust / 4 + (tx * y - ty * x) / (4 * ARM ** 2)
            for x, y, _ in ROTORS]


def apply_controls(body, thrusts, tz=0.0):
    """Push each rotor, then apply yaw torque.

    Must be called EVERY step: pybullet clears external forces after each
    stepSimulation(), so this is not a set-and-forget API.
    """
    for rotor, f in zip(ROTORS, thrusts):
        f = float(np.clip(f, 0.0, F_MAX))   # motors cannot pull down
        p.applyExternalForce(body, -1,
                             forceObj=[0, 0, f],
                             posObj=rotor,
                             flags=p.LINK_FRAME)   # body frame, so it tilts

    # Yaw is NOT a rotor force - four upward thrusts produce no torque about
    # the drone's own z-axis. Real quadrotors yaw from propeller drag; we
    # apply it directly. WORLD_FRAME because LINK_FRAME torque has behaved
    # inconsistently across pybullet versions, and below 25 deg of tilt the
    # two axes nearly coincide.
    p.applyExternalTorque(body, -1, torqueObj=[0, 0, float(tz)],
                          flags=p.WORLD_FRAME)


# ======================= the PID =======================

def position_pid(state, target, g=GAINS):
    """OUTER loop: where I want to be -> how I want to accelerate."""
    pos, vel, _, _ = state
    target = np.asarray(target, dtype=float)

    accel = np.array([
        g["kp_xy"] * (target[0] - pos[0]) - g["kd_xy"] * vel[0],
        g["kp_xy"] * (target[1] - pos[1]) - g["kd_xy"] * vel[1],
        g["kp_z"] * (target[2] - pos[2]) - g["kd_z"] * vel[2],
    ])
    return np.clip(accel, -A_MAX, A_MAX)


def accel_to_tilt(accel):
    """Desired acceleration -> desired lean.

    A quadrotor cannot push sideways, so horizontal acceleration has to come
    from tilting. For small angles a_horizontal = g * tilt, so the tilt we
    need is just (acceleration / g).
    """
    ax, ay, az = (float(v) for v in accel)
    pitch_des = float(np.clip(ax / G, -TILT_MAX, TILT_MAX))   # lean -> +x
    roll_des = float(np.clip(-ay / G, -TILT_MAX, TILT_MAX))   # +roll -> -y
    return roll_des, pitch_des, az


def attitude_pid(state, roll_des, pitch_des, a_z, yaw_des=0.0, g=GAINS):
    """INNER loop: desired lean -> four rotor thrusts and a yaw torque."""
    _, _, rpy, ang_vel = state

    # Only T*cos(roll)*cos(pitch) of the thrust fights gravity, so leaning
    # costs lift - about 10% at 25 deg. The floor stops a flipped drone
    # asking for infinite thrust.
    tilt = max(0.3, float(np.cos(rpy[0]) * np.cos(rpy[1])))
    F = MASS * (G + a_z) / tilt      # the G term is feedforward gravity
                                     # compensation, which is why a_z = 0
                                     # means hover exactly, and why no
                                     # integral term is needed

    # PD on angle. The MINUS on each D term is a brake - flip one and the
    # drone tears itself apart in half a second.
    tx = g["kp_att"] * (roll_des - rpy[0]) - g["kd_att"] * ang_vel[0]
    ty = g["kp_att"] * (pitch_des - rpy[1]) - g["kd_att"] * ang_vel[1]
    tz = g["kp_yaw"] * (yaw_des - rpy[2]) - g["kd_yaw"] * ang_vel[2]

    return mix(F, tx, ty), tz


def control(state, target, g=GAINS):
    """The whole cascade: state + target -> (four thrusts, yaw torque)."""
    roll_des, pitch_des, a_z = accel_to_tilt(position_pid(state, target, g))
    return attitude_pid(state, roll_des, pitch_des, a_z, 0.0, g)


# ======================= demo =======================

CAM_SMOOTH = 0.08
_cam = None


def follow_camera(pos):
    """Keep the camera on the drone, preserving the zoom and angle you set by
    dragging with the mouse - only the target point is overridden."""
    global _cam
    _cam = np.array(pos, float) if _cam is None else \
        _cam + CAM_SMOOTH * (np.asarray(pos, float) - _cam)

    info = p.getDebugVisualizerCamera()
    p.resetDebugVisualizerCamera(info[10], info[8], info[9], _cam.tolist())


TARGETS = [
    [2.0, 1.0, 2.0],
    [-1.5, 2.0, 1.0],
    [1.0, -2.0, 2.5],
    [0.0, 0.0, 1.5],
]


def main():
    body = setup(gui=True)
    marker = make_marker(TARGETS[0])
    reset(body, [0, 0, 1])

    print("standard PID - flying the targets in order. close the window to quit")

    i = 0
    while p.isConnected():
        target = np.asarray(TARGETS[i % len(TARGETS)], dtype=float)
        move_marker(marker, target)
        print(f"  -> {target}")

        # fly for 6 seconds, then move to the next target
        for _ in range(6 * 240):
            if not p.isConnected():
                return
            state = get_state(body)
            thrusts, tz = control(state, target)
            apply_controls(body, thrusts, tz)
            p.stepSimulation()
            follow_camera(state[0])
            time.sleep(DT)

        pos, vel, rpy, _ = get_state(body)
        print(f"     err {np.linalg.norm(target - pos):5.3f} m   "
              f"speed {np.linalg.norm(vel):5.3f} m/s   "
              f"tilt {np.degrees(max(abs(rpy[0]), abs(rpy[1]))):4.1f} deg")
        i += 1

    p.disconnect()


if __name__ == "__main__":
    main()
