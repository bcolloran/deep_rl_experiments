import numpy as np
from jax_nn_utils import softmax_sample

from collections import deque, namedtuple


class Node:
    def __init__(self, parent=None, data=None):
        if parent:
            parent.add_child(self)
        self.parent = parent
        self.children_of_node = []
        self.data = data

    def add_child(self, child):
        self.children_of_node.append(child)

    def change_parent(self):
        pass


class Tree:
    def __init__(self, root_node):
        self.node_id_num = 0
        self.root_id = root_node
        self.nodes = {}
        # a dict with keys -> a node, values -> parent nodes
        self.parents_of_node = {root_node: None}
        # a dict with keys -> a node, values -> lists of child nodes
        self.children_of_node = {root_node: []}

    def add_node(self, node):
        self.node_id_num += 1
        self.nodes[self.node_id_num] = node

    def add_leaf(self, parent, leaf):
        self.children_of_node[leaf] = []
        self.parents_of_node[leaf] = parent

        self.children_of_node[parent].append(leaf)

    def get_nodes(self):
        return list(self.parents_of_node)

    def get_parent(self, node):
        return self.parents_of_node[node]

    def gen_edges(self, depth_first=True):
        D = deque([self.root])
        pop = D.pop if depth_first else D.popleft

        while len(D) > 0:
            node = pop()
            for child in self.children(node):
                D.append(child)
                yield (node, child)

    def get_leaves(self):
        return [node for node, children in self.children_of_node if children == []]

    def softmax_sample_leaves_by_value(self):
        leaves = self.get_leaves()
        i = softmax_sample([leaf.value_est for leaf in leaves])
        return leaves[i]

    def traverse_from_leaf_to_depth_1(self):
        pass

    def prune_above_node(self, node):
        pass

    def step_down_to_node(self, node):
        self.prune_above_node(node)
        self.root = node

    def step_toward_and_reroot(self, target_node):
        pass


class Tree2:
    def __init__(self, root_node_data):
        self.node_id_num = 0
        self.root_id = self.node_id_num
        self.node_data = {self.root_id: root_node_data}
        # a dict with keys -> a node_id, values -> parent node_ids
        self.parents_of_node = {self.root_id: None}
        # a dict with keys -> a node_id, values -> lists of child node_ids
        self.children_of_node = {self.root_id: []}

    def add_node(self, node_data):
        self.node_id_num += 1
        self.node_data[self.node_id_num] = node_data
        return self.node_id_num

    def add_leaf(self, parent_id, leaf_data):
        new_id = self.add_node(leaf_data)
        self.children_of_node[new_id] = []
        self.parents_of_node[new_id] = parent_id
        self.children_of_node[parent_id].append(new_id)

    def get_node_ids(self):
        return list(self.parents_of_node)

    def gen_node_ids_and_data(self):
        return self.node_data.items()

    def get_parent_id(self, node):
        return self.parents_of_node[node]

    def gen_edge_tups(self, depth_first=True):
        D = deque([self.root_id])
        pop = D.pop if depth_first else D.popleft

        while len(D) > 0:
            parent_id = pop()
            for child_id in self.children_of_node[parent_id]:
                D.append(child_id)
                yield (self.node_data[parent_id], self.node_data[child_id])

    def get_leaf_ids(self):
        return [
            node_id
            for node_id, children in self.children_of_node.items()
            if children == []
        ]

    def softmax_sample_leaves_by_value(self):
        leaves = self.get_leaves()
        i = softmax_sample([leaf.value_est for leaf in leaves])
        return leaves[i]

    def traverse_from_leaf_to_depth_1(self):
        pass

    def prune_above_node(self, node):
        pass

    def step_down_to_node(self, node):
        self.prune_above_node(node)
        self.root = node

    def step_toward_and_reroot(self, target_node):
        pass


nodeData = namedtuple("NodeData", ["state"])


class RRT:
    def __init__(self, root_node, step_fn=None, step_radius=1, distance_fn=None):
        self.T = Tree2(root_node)
        if distance_fn is None:
            # use euclidean dist
            def distance_fn(a, b):
                return np.linalg.norm(a - b)

        self.D = distance_fn

        if step_fn is None:
            # take euclidean step
            def step_fn(x, targ):
                targ_dist = self.D(x, targ)
                if targ_dist <= step_radius:
                    final_pos = targ
                else:
                    final_pos = x + step_radius * (targ - x) / targ_dist
                return nodeData(final_pos)

        self.step = step_fn

    def nearest_id(self, x):
        closest_dist = np.inf
        # closest_node_id = -1
        # closest_node_data
        for n_id, data in self.T.gen_node_ids_and_data():
            # print(n_id, data)
            d = self.D(x, data.state)
            if d < closest_dist:
                closest_dist = d
                closest_node_id = n_id
                closest_node_data = data
        return closest_node_id, closest_node_data

    def grow_towards(self, targ_state):
        near_id, near_data = self.nearest_id(targ_state)
        new = self.step(near_data.state, targ_state)
        self.T.add_leaf(near_id, new)
