"""Evolve a neural network that steers the PID drone.

The network replaces ONE function from droneV2: position_pid. It answers a
single question -

        "which way, and how hard, should I accelerate?"

and droneV2's accel_to_tilt + attitude_pid + mix turn that into four rotor
thrusts. The whole genome is network weights; no gains are evolved.

    target position
        |            <- the NETWORK does this bit (6 inputs -> 3 outputs)
        v
    desired acceleration
        |            <- droneV2 does the rest, unchanged
        v
    four rotor thrusts

    python -u gaV2.py       evolve, writes best_genome.json
"""

import json
import time
from pathlib import Path

import numpy as np
import pybullet as p

import droneV2 as d

# Everything this script writes goes next to the script itself, not into
# whatever directory you happened to run it from.
HERE = Path(__file__).resolve().parent


# ##########################################################################
#                          THE DIALS YOU'LL TOUCH
# ##########################################################################

TRAIN = False            # True  = evolve, then watch the winner
                         # False = skip training, just fly best_genome.json

RESUME = False          # True  = carry on from best_genome.json instead of
                         #         starting from random. Ignored if TRAIN is
                         #         False, or if no saved genome exists.
RESUME_SIGMA = 0.05      # how far the seeded variants start from the saved
                         # genome, and the starting mutation size on a resume.
                         # Smaller than SIGMA because resuming is polishing,
                         # not exploring - at 0.15 the first generation would
                         # scatter far enough to throw away what you had.

POP_SIZE = 40            # candidates alive at once (and drones on screen)
GENERATIONS = 1000         # rounds of score -> select -> breed

WATCH = False             # fly the whole population at once in a window
SPEED = 4.0              # playback speed when watching; 1.0 = real time

# ##########################################################################


# ======================= the network =======================

N_IN = 6        # 3 position error + 3 velocity. That is ALL a navigator
                # needs: where to go, and how fast it is already going.
                # Attitude is the PID's business, not the network's.
N_HID = 8
N_OUT = 3       # desired acceleration: ax, ay, az

N_WEIGHTS = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT      # 83

W_MAX = 3.0        # genes in [0,1] map to weights in [-3, +3]
POS_SCALE = 5.0    # divide the inputs by this to keep them near [-1, 1]
VEL_SCALE = 5.0


def make_net(genome):
    """83 numbers in [0,1] -> (W1, b1, W2, b2)."""
    w = (np.asarray(genome, dtype=float) * 2.0 - 1.0) * W_MAX

    i = 0
    W1 = w[i:i + N_IN * N_HID].reshape(N_HID, N_IN);    i += N_IN * N_HID
    b1 = w[i:i + N_HID];                                i += N_HID
    W2 = w[i:i + N_HID * N_OUT].reshape(N_OUT, N_HID);  i += N_HID * N_OUT
    b2 = w[i:i + N_OUT]
    return W1, b1, W2, b2


def net_accel(net, pos, vel, target):
    """-> desired acceleration (3-vector, m/s^2)."""
    W1, b1, W2, b2 = net

    # Position ERROR, never absolute position. Absolute position would teach
    # one fixed route; the error teaches "the target is that way, accelerate
    # that way", which generalises to any A and any B.
    x = np.concatenate([(target - pos) / POS_SCALE,
                        vel / VEL_SCALE])

    h = np.tanh(W1 @ x + b1)
    out = np.tanh(W2 @ h + b2)          # bounded to [-1, 1] by construction

    # tanh is also the safety clamp - the network physically cannot ask for
    # more than A_MAX. And zero output is an exact hover, because attitude_pid
    # cancels gravity as feedforward, so a random generation-0 network floats
    # instead of dropping. That is what gives evolution something to measure
    # from the very first generation.
    return out * d.A_MAX


def nn_control(net, state, target):
    """Drop-in replacement for droneV2.control()."""
    pos, vel, _, _ = state
    accel = net_accel(net, pos, vel, target)
    roll_des, pitch_des, a_z = d.accel_to_tilt(accel)
    return d.attitude_pid(state, roll_des, pitch_des, a_z)


# ======================= the task =======================

# Scoring against ONE route teaches a memorised trajectory, not a controller.
# These four differ in direction, distance, climb and descent, so holding one
# altitude or drifting one way loses on at least three of them.
TRAIN_TARGETS = [
    ([0.0, 0.0, 1.0], [ 2.0,  1.0, 2.0]),    # climb, north-east
    ([0.0, 0.0, 1.5], [-1.5,  2.0, 0.8]),    # descend, opposite way
    ([0.0, 0.0, 1.0], [ 1.0, -2.0, 2.5]),    # climb hard
    ([0.0, 0.0, 2.0], [-2.0, -1.0, 1.2]),    # start high, descend
]

STEPS = 1920                     # 8 s at 240 Hz, identical for every genome

WP_TOL = 0.3                     # inside this counts as "at the target"
HOLD_TIME = 1.5                  # ...and it must STAY for this long
HOLD_STEPS = int(HOLD_TIME * 240)
HOLD_SPEED = 0.3                 # ...moving slower than this, m/s
HOLD_TILT = np.radians(5)        # ...and flat. Hovering in the sphere on a
                                 # 20 deg lean is a drone still accelerating
                                 # sideways; it just has not left yet.
UPRIGHT_TILT = np.radians(10)    # flatter than this counts as "upright"


# ----------------------- the cost -----------------------
#
# Everything is a COST: lower is better, all terms added. Each is normalised
# by the start-to-target distance, so the numbers mean the same thing whether
# the target is 1 m or 5 m away.
#
# There is NO crash penalty and nothing resets. A drone that hits the ground
# stays in the episode - it simply never completes the hold and loses those
# points. The clock is the only thing that ends a run.

W_ARRIVE_XY = 20.0    # horizontal distance left at the end
W_ARRIVE_Z  = 15.0    # vertical distance left - SEPARATE, because height is
                      # pure thrust and sideways is pure tilt. Merged into one
                      # distance, a drone that nails x and y while ignoring z
                      # still collects two thirds of the reward.
W_PATH      = 10.0    # average distance during the flight - the dense signal
W_PROGRESS  = 25.0    # the CLOSEST it ever got. Without this a drone that
                      # reaches 8 cm and then drifts to 2 m is scored as a
                      # 2 m drone - its best moment is invisible. Early on,
                      # when nothing completes the hold, "I got nearer than
                      # my parent did" is far easier to climb than final
                      # distance, so this is the main signal in generation 1.
W_OVERSHOOT =  5.0    # how far it drifted back out after that closest
                      # approach - the natural partner to progress. Together
                      # they say: getting there counts, and so does staying.
W_HOLD      = 25.0    # did it settle and STAY, or just touch and drift off
W_UPRIGHT   = 10.0    # bonus for flying level
W_STOP      =  5.0    # speed at the end
W_TIME      =  5.0    # how long until the hold completed

CAP = 10.0            # how far a term may run before it stops growing. Too
                      # low and every runaway scores the same number, which
                      # is a flat plateau exactly where generation 0 lives.
FLOOR = 0.5           # never normalise by less than this


def cost_terms(d0, d0_xy, d0_z, d_xy, d_z, d_final, mean_dist, speed,
               closest, hold_frac, upright_frac, time_frac):
    """Each term separately, so you can see which one drives a score."""
    d0s = max(d0, FLOOR)
    return {
        "arrive_xy": W_ARRIVE_XY * min(d_xy / max(d0_xy, FLOOR), CAP),
        "arrive_z":  W_ARRIVE_Z * min(d_z / max(d0_z, FLOOR), CAP),
        "path":      W_PATH * min(mean_dist / d0s, CAP),
        # 0 if it reached the target at any point, full if it never got
        # closer than it started
        "progress":  W_PROGRESS * min(closest / d0s, 1.0),
        # drifted back out after its best approach
        "overshoot": W_OVERSHOOT * min(max(0.0, d_final - closest) / d0s, 1.0),
        "stop":      W_STOP * min(speed / 2.0, 1.0),
        "time":      W_TIME * time_frac,
        # both of these are bonuses written as discounts: earn them fully and
        # the term is 0. Kept positive so a log-scale plot still works.
        "hold":      W_HOLD * (1.0 - min(hold_frac, 1.0)),
        "upright":   W_UPRIGHT * (1.0 - min(upright_frac, 1.0)),
    }


def cost(**kw):
    return float(sum(cost_terms(**kw).values()))


# ======================= flying =======================

def fly(bodies, nets, start, target, watch=False, ctrl=nn_control):
    """Fly N drones through one episode at once. Returns one cost each.

    Written once and used for a single drone, the whole population, and the
    hand-tuned PID baseline, so none of those can disagree about what a
    score means. `ctrl(net, state, target) -> (thrusts, yaw_torque)`.
    """
    n = len(nets)
    start_a = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)

    for b in bodies[:n]:
        d.reset(b, start)

    d0 = float(np.linalg.norm(target - start_a))
    dist = np.full(n, d0)
    closest = np.full(n, d0)              # best approach of the whole episode
    dist_sum = np.zeros(n)
    speed = np.zeros(n)
    hold = np.zeros(n, dtype=int)         # current unbroken stretch parked
    best_hold = np.zeros(n, dtype=int)    # longest such stretch
    upright = np.zeros(n, dtype=int)      # steps spent flatter than UPRIGHT
    t_done = np.full(n, STEPS, dtype=float)
    final = np.tile(start_a, (n, 1))

    for i in range(STEPS):
        for k in range(n):
            state = d.get_state(bodies[k])
            pos, vel, rpy, _ = state

            final[k] = pos
            dist[k] = np.linalg.norm(target - pos)
            dist_sum[k] += dist[k]
            closest[k] = min(closest[k], dist[k])
            speed[k] = np.linalg.norm(vel)

            tilt = max(abs(rpy[0]), abs(rpy[1]))
            if tilt < UPRIGHT_TILT:
                upright[k] += 1

            # Parked = in the sphere, slow, AND flat. Touching it at speed
            # does not count - that is the ballistic fly-through.
            level = max(abs(rpy[0]), abs(rpy[1]), abs(rpy[2]))
            if (dist[k] < WP_TOL and speed[k] < HOLD_SPEED
                    and level < HOLD_TILT):
                hold[k] += 1
                best_hold[k] = max(best_hold[k], hold[k])
                if hold[k] >= HOLD_STEPS and t_done[k] == STEPS:
                    t_done[k] = i          # the moment the task was completed
            else:
                hold[k] = 0                # broke the hold - start again

            # no crash check: a drone on the ground stays in the episode
            thrusts, tz = ctrl(nets[k], state, target)
            d.apply_controls(bodies[k], thrusts, tz)

        p.stepSimulation()
        if watch:
            time.sleep(d.DT / SPEED)

    d0_xy = float(np.linalg.norm((target - start_a)[:2]))
    d0_z = float(abs((target - start_a)[2]))

    return np.array([
        cost(d0=d0, d0_xy=d0_xy, d0_z=d0_z,
             d_xy=float(np.linalg.norm((target - final[k])[:2])),
             d_z=float(abs((target - final[k])[2])),
             d_final=float(np.linalg.norm(target - final[k])),
             mean_dist=dist_sum[k] / STEPS,
             speed=speed[k],
             closest=float(closest[k]),
             hold_frac=best_hold[k] / HOLD_STEPS,
             upright_frac=upright[k] / STEPS,
             time_frac=t_done[k] / STEPS)
        for k in range(n)])


def score(pop, bodies, watch=False):
    """Average cost across ALL training targets - this is what makes the
    result a controller rather than one memorised route."""
    nets = [make_net(g) for g in pop]
    total = np.zeros(len(pop))
    for start, target in TRAIN_TARGETS:
        total += fly(bodies, nets, start, target, watch)
    return total / len(TRAIN_TARGETS)


# ======================= the GA =======================

TOURNAMENT_K = 3         # pick this many at random, best becomes a parent
P_CROSSOVER = 0.7        # chance two parents blend instead of cloning one
P_MUTATE = 0.2           # per-gene chance of a random nudge
SIGMA = 0.15             # size of that nudge
SIGMA_DECAY = 0.98       # shrink it each generation: explore early, polish late
SIGMA_MIN = 0.02         # ...but never below this. 0.98 compounds, and by
                         # generation 220 it reaches 0.002 - mutation dead,
                         # search frozen, with most of the run still to go.
N_ELITES = 2             # how many of the best survive untouched
SEED = 0                 # fixed seed = reproducible run

OUT_BEST = HERE / "best_genome.json"
OUT_PLOT = HERE / "convergence.png"


def tournament(pop, costs, rng):
    """Pick TOURNAMENT_K at random, return the best. Random entry means good
    genomes are LIKELY to win but not guaranteed - that leftover variety is
    what stops the population collapsing into clones."""
    idx = rng.integers(0, len(pop), TOURNAMENT_K)
    return pop[idx[np.argmin(costs[idx])]]


def crossover(a, b, rng):
    if rng.random() < P_CROSSOVER:
        return (a + b) / 2.0
    return a.copy()


def mutate(genome, sigma, rng):
    """The ONLY source of genuinely new values - selection and crossover can
    only reshuffle what the population already has."""
    mask = rng.random(len(genome)) < P_MUTATE
    genome = genome + mask * rng.normal(0.0, sigma, len(genome))
    return np.clip(genome, 0.0, 1.0)


def seed_population(rng, genome):
    """Build generation 0 around an already-trained genome.

    Three parts, on purpose:
      * slot 0 is the saved genome EXACTLY, so elitism guarantees the resumed
        run can never end up worse than what you started with
      * three quarters are small variants of it - the local search that
        actually polishes it
      * the last quarter stay random, so the run keeps some fresh material
        and is not purely a hill-climb

    The split was half-and-half at first, and measuring showed that was
    wasteful: the random half sat at a median cost of 44 against a best of
    6.3, so none of them ever won a tournament. Fifteen slots doing nothing.
    A quarter is enough to keep some diversity without burning the budget.
    """
    pop = rng.random((POP_SIZE, N_WEIGHTS))
    pop[0] = genome
    for i in range(1, (3 * POP_SIZE) // 4):
        pop[i] = np.clip(genome + rng.normal(0.0, RESUME_SIGMA, N_WEIGHTS),
                         0.0, 1.0)
    return pop


def evolve(bodies, seed_genome=None):
    rng = np.random.default_rng(SEED)

    if seed_genome is None:
        pop = rng.random((POP_SIZE, N_WEIGHTS))
        sigma = SIGMA
    else:
        pop = seed_population(rng, seed_genome)
        sigma = RESUME_SIGMA        # polishing, not exploring

    history = []
    t0 = time.time()

    for gen in range(GENERATIONS):
        costs = score(pop, bodies, WATCH)                 # 1. score

        order = np.argsort(costs)                         # 2. best first
        pop, costs = pop[order], costs[order]             #    (LOWER wins)

        history.append((float(costs[0]), float(np.median(costs))))
        # median, not mean: the cost range here is ~6 to ~480, so a single
        # unlucky child drags the mean around by tens of points and hides
        # whether the population is actually improving. The median ignores
        # outliers. `best` is monotonic anyway - elitism guarantees it.
        print(f"gen {gen:3d}   best {costs[0]:8.3f}   "
              f"median {np.median(costs):8.3f}   "
              f"worst {costs[-1]:8.1f}   sigma {sigma:.3f}", flush=True)

        next_pop = [pop[i].copy() for i in range(N_ELITES)]   # 3. elitism

        while len(next_pop) < POP_SIZE:                       # 4. breed
            pa = tournament(pop, costs, rng)
            pb = tournament(pop, costs, rng)
            next_pop.append(mutate(crossover(pa, pb, rng), sigma, rng))

        pop = np.array(next_pop)
        sigma = max(sigma * SIGMA_DECAY, SIGMA_MIN)

    costs = score(pop, bodies, False)
    best = pop[int(np.argmin(costs))]
    print(f"\n{GENERATIONS} generations in {time.time() - t0:.0f}s")
    return best, float(costs.min()), history


# ======================= main =======================

def spawn(rgba):
    """One more drone body for the swarm. Collisions are off so they can
    overlap freely - nothing here depends on them touching."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=d.HALF)
    vis = p.createVisualShapeArray(
        shapeTypes=[p.GEOM_BOX] + [p.GEOM_CYLINDER] * 4,
        halfExtents=[d.HALF] + [[0, 0, 0]] * 4,
        radii=[0] + [d.ROTOR_R] * 4,
        lengths=[0] + [0.008] * 4,
        rgbaColors=[list(rgba)] + [[0.15, 0.15, 0.18, 0.6]] * 4,
        visualFramePositions=[[0, 0, 0]]
        + [[x, y, d.HALF[2] + 0.006] for x, y, _ in d.ROTORS],
    )
    body = p.createMultiBody(baseMass=d.MASS, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=[0, 0, 1])
    p.setCollisionFilterGroupMask(body, -1, 0, 0)
    return body


def pid_baseline(body):
    """What droneV2's hand-tuned PID scores on the same task. The number to
    beat - and proof that a good score is reachable at all."""
    def pid_ctrl(_net, state, target):
        return d.control(state, target)

    return float(np.mean([fly([body], [None], s, t, ctrl=pid_ctrl)[0]
                          for s, t in TRAIN_TARGETS]))


def load_genome():
    """The trained genome, or a random one if nothing has been evolved yet."""
    try:
        with open(OUT_BEST) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"no {OUT_BEST.name} yet - flying a RANDOM net. Train first.")
        return np.random.default_rng(0).random(N_WEIGHTS)

    genome = np.array(data["genome"], dtype=float)
    if len(genome) != N_WEIGHTS:
        print(f"{OUT_BEST.name} has {len(genome)} genes but the network needs "
              f"{N_WEIGHTS} - it is out of date. Retrain.")
        return np.random.default_rng(0).random(N_WEIGHTS)

    print(f"loaded {OUT_BEST.name} - cost was {data['cost']:.3f}")
    return genome


def watch_best():
    """Fly the trained network through the targets, forever.

    This lives here rather than in droneV2.py because the network is defined
    here - droneV2 would have to import gaV2, and gaV2 already imports
    droneV2, which is a circular import. droneV2 stays pure drone + PID.
    """
    body = d.setup(gui=True)
    marker = d.make_marker(TRAIN_TARGETS[0][1], radius=WP_TOL)
    net = [make_net(load_genome())]

    print("flying the trained net - close the window to quit\n")
    while p.isConnected():
        for start, target in TRAIN_TARGETS:
            if not p.isConnected():
                return
            d.move_marker(marker, target)
            c = fly([body], net, start, target, watch=True)[0]
            pos, vel, rpy, _ = d.get_state(body)
            print(f"  {str(target):18s} cost {c:7.3f}   "
                  f"err {np.linalg.norm(np.asarray(target) - pos):5.3f} m   "
                  f"speed {np.linalg.norm(vel):5.3f}   "
                  f"tilt {np.degrees(max(abs(rpy[0]), abs(rpy[1]))):4.1f} deg",
                  flush=True)


def main():
    if not TRAIN:
        watch_best()
        return

    body = d.setup(gui=WATCH)
    p.changeVisualShape(body, -1, rgbaColor=[0.2, 0.9, 0.3, 0.9])
    p.setCollisionFilterGroupMask(body, -1, 0, 0)

    # elites green so you can follow them, the rest translucent red
    bodies = [body] + [spawn((0.2, 0.9, 0.3, 0.9) if i < N_ELITES
                             else (0.9, 0.25, 0.2, 0.5))
                       for i in range(1, POP_SIZE)]
    marker = d.make_marker(TRAIN_TARGETS[0][1], radius=WP_TOL)

    print(f"genes={N_WEIGHTS}  pop={POP_SIZE}  gens={GENERATIONS}  "
          f"targets={len(TRAIN_TARGETS)}  "
          f"{'RESUMING' if RESUME else 'from scratch'}  "
          f"{'WATCHING' if WATCH else 'headless'}")
    print("green = the two elites (last generation's best)\n", flush=True)

    # reference points, so the numbers below mean something
    flat = [make_net(np.full(N_WEIGHTS, 0.5))]      # zero weights -> hover
    hover = np.mean([fly([body], flat, s, t)[0] for s, t in TRAIN_TARGETS])
    print(f"hand-tuned PID : {pid_baseline(body):7.3f}   <- beat this")
    print(f"do-nothing net : {hover:7.3f}   <- zero weights, just hovers")

    seed_genome = None
    if RESUME:
        if OUT_BEST.exists():
            seed_genome = load_genome()
            # score it HERE rather than trusting the number in the file - the
            # cost function may well have changed since it was written, and a
            # stale figure would make any "improvement" meaningless
            now = float(np.mean([fly([body], [make_net(seed_genome)], s, t)[0]
                                 for s, t in TRAIN_TARGETS]))
            print(f"resuming from  : {now:7.3f}   <- scored against the "
                  f"CURRENT cost function")
        else:
            print(f"RESUME is on but there is no {OUT_BEST.name} - "
                  f"starting from scratch")
    print(flush=True)

    best, best_cost, history = evolve(bodies, seed_genome)

    with open(OUT_BEST, "w") as f:
        json.dump({"cost": best_cost, "genome": [float(x) for x in best]},
                  f, indent=2)
    print(f"best cost {best_cost:.3f}  ->  saved to {OUT_BEST}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot([h[0] for h in history], label="best")
        plt.plot([h[1] for h in history], label="median")
        plt.yscale("log")
        plt.xlabel("generation"); plt.ylabel("cost")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(OUT_PLOT, dpi=120)
        print(f"saved -> {OUT_PLOT}")
    except ImportError:
        pass

    if WATCH:
        print("\nthe winner - close the window to exit")
        win = [make_net(best)]
        while p.isConnected():
            for start, target in TRAIN_TARGETS:
                d.move_marker(marker, target)
                fly([body], win, start, target, watch=True)


if __name__ == "__main__":
    main()
