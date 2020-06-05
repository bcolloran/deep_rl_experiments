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


class RrtNode:
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


# class Tree:
#     def __init__(self, root_node):
#         self.node_id_num = 0
#         self.root_id = root_node
#         self.nodes = {}
#         # a dict with keys -> a node, values -> parent nodes
#         self.parents_of_node = {root_node: None}
#         # a dict with keys -> a node, values -> lists of child nodes
#         self.children_of_node = {root_node: []}

#     def add_node(self, node):
#         self.node_id_num += 1
#         self.nodes[self.node_id_num] = node

#     def add_leaf(self, parent, leaf):
#         self.children_of_node[leaf] = []
#         self.parents_of_node[leaf] = parent

#         self.children_of_node[parent].append(leaf)

#     def get_nodes(self):
#         return list(self.parents_of_node)

#     def get_parent(self, node):
#         return self.parents_of_node[node]

#     def gen_edges(self, depth_first=True):
#         D = deque([self.root])
#         pop = D.pop if depth_first else D.popleft

#         while len(D) > 0:
#             node = pop()
#             for child in self.children(node):
#                 D.append(child)
#                 yield (node, child)

#     def get_leaves(self):
#         return [node for node, children in self.children_of_node if children == []]

#     def softmax_sample_leaves_by_value(self):
#         leaves = self.get_leaves()
#         i = softmax_sample([leaf.value_est for leaf in leaves])
#         return leaves[i]

#     def traverse_from_leaf_to_depth_1(self):
#         pass

#     def prune_above_node(self, node):
#         pass

#     def step_down_to_node(self, node):
#         self.prune_above_node(node)
#         self.root = node

#     def step_toward_and_reroot(self, target_node):
#         pass


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
        return new_id

    def get_node_data(self, node_id):
        return self.node_data[node_id]

    def set_node_data(self, node_id, data):
        self.node_data[node_id] = data

    def get_node_ids(self):
        return list(self.parents_of_node)

    def gen_node_ids_and_data(self):
        return self.node_data.items()

    def get_parent_id(self, node_id):
        return self.parents_of_node[node_id]

    def change_parent(self, node_id, new_parent_id):
        old_parent_id = self.parents_of_node[node_id]
        self.children_of_node[old_parent_id].remove(node_id)
        self.parents_of_node[node_id] = new_parent_id
        self.children_of_node[new_parent_id].append(node_id)

    def get_ancestor_data(self, node_id):
        data_list = [self.node_data[node_id]]
        node_id = self.get_parent_id(node_id)
        while node_id is not None:
            data_list.append(self.node_data[node_id])
            node_id = self.get_parent_id(node_id)
        return data_list

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

    def softmax_sample_node_date_by_value(self):
        # leaf_ids = self.get_leaf_ids()
        i = softmax_sample(
            [self.get_node_data[l_id].value_est for l_id in list(self.node_data)]
        )
        return self.get_node_data[i]

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
        for n_id, data in self.T.gen_node_ids_and_data():
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


class RRTstar:
    def __init__(
        self, root_node, node_tup_fun, step_fn=None, step_radius=1, distance_fn=None
    ):
        self.node_tup_fun = node_tup_fun
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
                    # numerical issue in near_ids requires that the
                    # step size used here be a bit less than the desired
                    # step_radius
                    final_pos = x + 0.99999 * step_radius * (targ - x) / targ_dist
                return final_pos

        self.step = step_fn
        self.step_radius = step_radius

    def nearest_id(self, x):
        closest_dist = np.inf
        for n_id, data in self.T.gen_node_ids_and_data():
            d = self.D(x, data.state)
            if d < closest_dist:
                closest_dist = d
                closest_node_id = n_id
                closest_node_data = data
        return closest_node_id, closest_node_data

    def near_ids(self, x):
        return [
            n_id
            for n_id, data in self.T.gen_node_ids_and_data()
            if self.D(x, data.state) <= self.step_radius
        ]

    def path_cost(self, node_id):
        return sum(d.inbound_cost for d in self.T.get_ancestor_data(node_id))

    def path_costs_state_inbound_by_ids(self, node_ids):
        return [
            (
                n_id,
                self.path_cost(n_id),
                self.T.get_node_data(n_id).state,
                self.T.get_node_data(n_id).inbound_cost,
            )
            for n_id in node_ids
        ]

    def grow_towards(self, targ_state):
        near_id, near_data = self.nearest_id(targ_state)
        new_state = self.step(near_data.state, targ_state)
        near_ids = self.near_ids(new_state)

        # find cheapest path to new node among near nodes
        near_nodes_info = self.path_costs_state_inbound_by_ids(near_ids)
        min_cost = np.inf
        for n_id, cost, state, in_cost in near_nodes_info:
            dist = self.D(new_state, state)
            if cost + dist < min_cost:
                min_cost = cost + dist
                inbound_cost = dist
                cheapest_parent_id = n_id

        new_id = self.T.add_leaf(
            cheapest_parent_id,
            self.node_tup_fun(state=new_state, inbound_cost=inbound_cost),
        )
        new_path_cost = min_cost

        # rewire near nodes if it's cheaper to go through the new node
        for n_id, cost, state, in_cost in near_nodes_info:
            if n_id == cheapest_parent_id:
                # no need to process the parent of the new node
                continue
            dist = self.D(new_state, state)

            if new_path_cost + dist < cost:
                self.T.change_parent(n_id, new_parent_id=new_id)
                data = self.node_tup_fun(state=state, inbound_cost=dist)
                self.T.set_node_data(n_id, data)
