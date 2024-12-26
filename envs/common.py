import jax
from jax import numpy as jnp
import xml.etree.ElementTree as ET


def sample_circle(radius, rng) -> jax.Array:
    ang = jnp.pi * 2.0 * jax.random.uniform(rng)
    target_x = radius * jnp.cos(ang)
    target_y = radius * jnp.sin(ang)
    return jax.Array([target_x, target_y])

# NOTE: This is NOT uniform sampling from a disk. 
# Here both angle and distance are sampled uniformly
def sample_disk(rng, min_radius=0.0, max_radius=1.0) -> jax.Array:
    rng, rng1 = jax.random.split(rng, 2)
    radius = jax.random.uniform(rng, minval=min_radius, maxval=max_radius)
    ang = jnp.pi * 2.0 * jax.random.uniform(rng1)
    target_x = radius * jnp.cos(ang)
    target_y = radius * jnp.sin(ang)
    return jnp.array([target_x, target_y])


def sample_choice(rng, choices) -> jax.Array:
    idx = jax.random.randint(rng, (1,), 0, len(choices))
    return jnp.array(choices[idx])[0]


def find(structure, size_scaling, obj):
    objects = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == obj:
                objects.append([i * size_scaling, j * size_scaling])

    return jnp.array(objects)


RESET = R = 'r'
GOAL = G = 'g'
BALL = B = 'b'

def make_maze(xml_path, maze_layout, maze_size_scaling, maze_height=0.5):
    possible_starts = find(maze_layout, maze_size_scaling, RESET)
    possible_goals = find(maze_layout, maze_size_scaling, GOAL)
    possible_balls = find(maze_layout, maze_size_scaling, BALL)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * maze_size_scaling,
                                    j * maze_size_scaling,
                                    maze_height / 2 * maze_size_scaling),
                    size="%f %f %f" % (0.5 * maze_size_scaling,
                                        0.5 * maze_size_scaling,
                                        maze_height / 2 * maze_size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)
    
    return xml_string, possible_starts, possible_goals, possible_balls



