"""Environment supervisor for the building site.

- Animates the fires: flame tongues rise, sway and shrink, then start again at
  the base; smoke puffs rise and spread; the firelight flickers.
- Dynamic debris start out still attached to walls. When the robot comes
  close, each one shakes for a moment, then breaks loose and falls into its path.
"""
import math
import random

from controller import Supervisor

# DEF name in the world, mass (kg), what to call it
HANGING_DEBRIS = [
    ("HANGING_CEILING_SLAB", 150, "ceiling slab in the corridor"),
    ("LOOSE_WALL_CHUNK", 90, "chunk of brick wall by the collapsed zone"),
]
WARN_DISTANCE = 4.0   # starts shaking when the robot is this close (meters)
SHAKE_TIME = 1.2      # seconds of shaking before it falls
SHAKE_SIZE = 0.015    # meters

FLAME_RADIUS = 0.14   # must match tools/generate_building_site.py
ANIMATE_EVERY = 2     # update the fire particles every N steps

sup = Supervisor()
timestep = int(sup.getBasicTimeStep())
robot = sup.getFromDef("RESPONDER")


# ------------------------------------------------------------------ fires
# Per particle kind (yellow core, orange middle, red tip): how high it rises,
# how big it gets and how far from the center it starts
FLAME_KINDS = {
    "core": {"height": (0.35, 0.65), "swell": (1.1, 1.4), "spread": 0.25},
    "middle": {"height": (0.7, 1.1), "swell": (0.9, 1.2), "spread": 0.3},
    "tip": {"height": (1.0, 1.5), "swell": (0.6, 0.9), "spread": 0.2},
}
KIND_ORDER = ["core", "middle", "middle", "tip"]  # must match tools/generate_building_site.py


def new_flame(p, size, start_age=None):
    kind = FLAME_KINDS[p["kind"]]
    p["life"] = random.uniform(0.5, 1.0)
    p["age"] = random.uniform(0, p["life"]) if start_age is None else start_age
    ax, ay = p["area"]
    inward = kind["spread"] / 0.3  # core flames stay nearer the middle
    p["x0"] = random.uniform(-0.5, 0.5) * ax * inward
    p["y0"] = random.uniform(-0.5, 0.5) * ay * inward
    p["height"] = random.uniform(*kind["height"]) * size
    p["swell"] = random.uniform(*kind["swell"])
    p["sway"] = random.uniform(0, 6.28)


def new_puff(p, size, start_age=None):
    p["life"] = random.uniform(2.5, 4.0)
    p["age"] = random.uniform(0, p["life"]) if start_age is None else start_age
    ax, ay = p["area"]
    p["x0"], p["y0"] = random.uniform(-0.3, 0.3) * ax, random.uniform(-0.3, 0.3) * ay
    p["drift"] = random.uniform(0.3, 0.7) * size


def find_fires():
    fires = []
    n = 1
    while sup.getFromDef(f"FIRE_{n}") is not None:
        light = sup.getFromDef(f"FIRE_{n}_LIGHT")
        flames, puffs = [], []
        k = 1
        while (node := sup.getFromDef(f"FIRE_{n}_FLAME_{k}")) is not None:
            flames.append({"t": node.getField("translation"), "s": node.getField("scale"),
                           "kind": KIND_ORDER[(k - 1) % 4]})
            k += 1
        k = 1
        while (node := sup.getFromDef(f"FIRE_{n}_SMOKE_{k}")) is not None:
            puffs.append({"t": node.getField("translation"), "s": node.getField("scale")})
            k += 1
        area_node = sup.getFromDef(f"FIRE_{n}_AREA")
        area = area_node.getField("scale").getSFVec3f()[:2] if area_node is not None else [1.0, 1.0]
        for p in flames + puffs:
            p["area"] = area
        size = 1.0
        if flames:
            shape = sup.getFromDef(f"FIRE_{n}_FLAME_1").getField("children").getMFNode(0)
            size = shape.getField("geometry").getSFNode().getField("radius").getSFFloat() / FLAME_RADIUS
        for p in flames:
            new_flame(p, size)
        for p in puffs:
            new_puff(p, size)
        fires.append({"light": light.getField("intensity") if light else None,
                      "base": light.getField("intensity").getSFFloat() if light else 0,
                      "flames": flames, "puffs": puffs, "size": size, "level": 1.0})
        n += 1
    return fires


def animate(fires, t, dt):
    for fire in fires:
        size = fire["size"]
        for p in fire["flames"]:
            p["age"] += dt
            if p["age"] >= p["life"]:
                new_flame(p, size, start_age=0.0)
            k = p["age"] / p["life"]  # 0 at the base, 1 at the tip
            sway = 0.08 * size * math.sin(9 * t + p["sway"]) * k
            x = p["x0"] * (1 - 0.7 * k) + sway
            y = p["y0"] * (1 - 0.7 * k) + 0.6 * sway
            z = 0.03 * size + k * p["height"]
            r = p["swell"] * (0.35 + 0.65 * math.sin(math.pi * min(1.0, 0.25 + k)))  # grows, then shrinks to the tip
            p["t"].setSFVec3f([x, y, z])
            p["s"].setSFVec3f([r, r, r * (1.8 + 1.0 * k)])
        for p in fire["puffs"]:
            p["age"] += dt
            if p["age"] >= p["life"]:
                new_puff(p, size, start_age=0.0)
            k = p["age"] / p["life"]
            p["t"].setSFVec3f([p["x0"] + p["drift"] * k, p["y0"] + 0.3 * p["drift"] * k, (1.4 + 2.6 * k) * size])
            r = 0.5 + 1.5 * k
            p["s"].setSFVec3f([r, r, 0.85 * r])
        if fire["light"] is not None:
            target = 0.8 + 0.3 * random.random()
            fire["level"] += 0.35 * (target - fire["level"])
            fire["light"].setSFFloat(fire["base"] * fire["level"])


# ------------------------------------------------------------------ hanging debris
def find_hanging_debris():
    debris = []
    for def_name, mass, label in HANGING_DEBRIS:
        node = sup.getFromDef(def_name)
        if node is None:
            print(f"[SITE] {def_name} not found in the world, skipping it.")
            continue
        debris.append({"node": node, "mass": mass, "label": label, "shake_start": None,
                       "home": node.getField("translation").getSFVec3f()})
    return debris


def update_debris(hanging, t):
    for item in hanging[:]:
        field = item["node"].getField("translation")
        if item["shake_start"] is None:
            if robot is None:
                continue
            rx, ry, _ = robot.getPosition()
            hx, hy, _ = item["home"]
            if math.hypot(rx - hx, ry - hy) < WARN_DISTANCE:
                item["shake_start"] = t
                print(f"[SITE] The {item['label']} is starting to give way...")
        elif t - item["shake_start"] < SHAKE_TIME:
            field.setSFVec3f([c + random.uniform(-SHAKE_SIZE, SHAKE_SIZE) for c in item["home"]])
        else:
            field.setSFVec3f(item["home"])
            # Giving the piece a Physics node lets gravity take over
            item["node"].getField("physics").importSFNodeFromString(f"Physics {{ density -1 mass {item['mass']} }}")
            print(f"[SITE] The {item['label']} broke loose and fell.")
            hanging.remove(item)


fires = find_fires()
hanging = find_hanging_debris()
print(f"[SITE] {len(fires)} fires burning, {len(hanging)} pieces of debris hanging.")

step = 0
while sup.step(timestep) != -1:
    step += 1
    t = sup.getTime()
    if step % ANIMATE_EVERY == 0:
        animate(fires, t, ANIMATE_EVERY * timestep / 1000.0)
    update_debris(hanging, t)
