# %%
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

import trees

from importlib import reload

# %%

nodeData = namedtuple("NodeData", ["state"])

T = trees.Tree2(nodeData(np.random.randn(2)))


for i in range(10):
    parent_id = np.random.choice(T.get_node_ids())
    child_data = nodeData(np.random.randn(2))
    T.add_leaf(parent_id, child_data)
T.get_node_ids()

# %%


def plot_tree(T):
    edge_tups = list(T.gen_edge_tups())
    x = np.zeros(len(edge_tups) * 3)
    y = np.zeros(len(edge_tups) * 3)
    for i, (node1, node2) in enumerate(edge_tups):
        j = i * 3
        x[j] = node1.state[0]
        x[j + 1] = node2.state[0]
        x[j + 2] = np.nan
        y[j] = node1.state[1]
        y[j + 1] = node2.state[1]
        y[j + 2] = np.nan
    fig, ax = plt.subplots()
    ax.plot(x, y, linestyle="-", marker=".", markersize=5)
    root_point = T.node_data[T.root_id].state
    ax.plot(root_point[0], root_point[1], ".k", markersize=12)
    newest_leaf_point = T.node_data[max(list(T.node_data))].state
    ax.plot(newest_leaf_point[0], newest_leaf_point[1], ".r", markersize=8)
    ax.set_aspect(1)


# %%
reload(trees)

nodeData = namedtuple("NodeData", ["state"])

rrt = trees.RRT(nodeData(np.random.randn(2)), step_radius=0.2)


for i in range(100):
    rrt.grow_towards(np.random.randn(2))

plot_tree(rrt.T)


# %%

rrt.T.get_ancestor_data(90)

# %%
reload(trees)

nodeData = namedtuple("NodeData", ["state", "inbound_cost"])
# rrt_star = trees.RRTstar(
#     nodeData(state=np.random.randn(2), inbound_cost=0),
#     node_tup_fun=nodeData,
#     step_radius=np.sqrt(2)*1.001,
# )

rrt_star = trees.RRTstar(
    nodeData(state=np.array([0, 0]), inbound_cost=0),
    node_tup_fun=nodeData,
    step_radius=np.sqrt(2) * 1.001,
)


for i in range(10):
    # print("i",i)
    # rrt_star.grow_towards(np.random.randn(2))
    rrt_star.grow_towards(np.array([i + 1, np.cos(i * np.pi / 2)]))
    plot_tree(rrt_star.T)
    plt.show()

for i in range(10):
    # print("i",i)
    # rrt_star.grow_towards(np.random.randn(2))
    rrt_star.grow_towards(np.array([i + 1, i * 0.3]))
    plot_tree(rrt_star.T)
    plt.show()


# %%
reload(trees)

nodeData = namedtuple("NodeData", ["state", "inbound_cost"])
rrt_star = trees.RRTstar(
    nodeData(state=np.random.randn(2), inbound_cost=0),
    node_tup_fun=nodeData,
    step_radius=1,
)
np.random.seed(0)

for i in range(100):
    print(i)
    rrt_star.grow_towards(np.random.randn(2))
    plot_tree(rrt_star.T)
    plt.show()


# %%
