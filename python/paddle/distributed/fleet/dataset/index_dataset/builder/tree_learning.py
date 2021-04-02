import paddle
from paddle.fluid.core import IndexWrapper, TreeIndex
import paddle.fluid as fluid
from paddle.fluid.framework import Program
import numpy as np
from tdm import TDMBaseModel
import random
import os
import multiprocessing as mp
import json
import time
import math
from utils import mp_run

paddle.enable_static()

class UserPreferenceModel:
    def __init__(self, init_model_path, tree_node_num, node_emb_size):
        self.init_model_path = init_model_path
        self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        
        self.node_emb_size = node_emb_size

        self.model = TDMBaseModel()
        self.create_embedding_lookup_model(tree_node_num)
        self.create_prediction_model()

    def calc_prediction_weight(self, sample_set, paths):
        n_samples = len(sample_set)
        n_path = len(paths)

        user_emb = self.user_embedding_lookup(sample_set)
        user_emb = [np.repeat(user_emb[i], n_path, axis=0) for i in range(len(user_emb))]
        
        node_emb = self.node_embedding_lookup(paths)
        node_emb = np.concatenate([node_emb] * n_samples, axis=0)

        prob = self.calc_prob(user_emb, node_emb)
        return np.sum(prob)

    def calc_prob(self, user_inputs, unit_id_emb):
        feed_dict = {}
        for i in range(69):
            feed_dict["user_emb_{}".format(i)] = user_inputs[i]
        feed_dict["unit_id_emb"] = unit_id_emb
        
        res = self.exe.run(self.prediction_model,
                     feed=feed_dict,
                     fetch_list=self.prediction_model_fetch_vars)
        return res[0]

    def node_embedding_lookup(self, all_nodes):
        """ embedding lookup 
        """
        all_nodes = np.array(all_nodes).reshape([-1, 1]).astype('int64')
        res = []
        res = self.exe.run(self.embedding_lookup_program,
                     feed={"all_nodes": all_nodes},
                     fetch_list=self.embedding_fetch_var_names)
        return np.expand_dims(res[0], axis=1)

    def user_embedding_lookup(self, user_ids, ):
        all_nodes = np.array(user_ids).astype('int64')
        shape = all_nodes.shape
        if (shape[-1] != 1):
            shape = list(shape) + [1]
        all_nodes = all_nodes.reshape(shape)

        res = []
        res = self.exe.run(self.embedding_lookup_program,
                     feed={"all_nodes": all_nodes},
                     fetch_list=self.embedding_fetch_var_names)
        
        user_embeddings = []
        for i in range(all_nodes.shape[1]):
            user_embeddings.append(np.expand_dims(res[0][:,i,:], axis=1))
        return user_embeddings

    def create_prediction_model(self, with_att=False):
        self.prediction_model = Program()
        startup = Program()

        with paddle.fluid.framework.program_guard(self.prediction_model, startup):    
            user_input = [
                fluid.layers.data(
                    name="user_emb_{}".format(i),
                    shape=[-1, 1, self.node_emb_size],
                    dtype="float32",
                ) for i in range(69)
            ]
            unit_id_emb = fluid.layers.data(
                name="unit_id_emb",
                shape=[-1, 1, self.node_emb_size],
                dtype="float32")
            prob = self.model.net(user_input, unit_id_emb, self.node_emb_size, with_att)
            #print(str(self.prediction_model))
            self.prediction_model_fetch_vars = [prob.name]
            self.exe.run(startup)

    def create_embedding_lookup_model(self, tree_node_num):
        """ create embedding lookup model 
        """
        self.embedding_lookup_program = Program()
        startup = Program()

        with paddle.fluid.framework.program_guard(self.embedding_lookup_program, startup):
            all_nodes = fluid.layers.data(
                name="all_nodes",
                shape=[-1, 1],
                dtype="int64",
                lod_level=1,
            )

            output = fluid.layers.embedding(
                input=all_nodes,
                is_sparse=True,
                size=[tree_node_num, self.node_emb_size],
                param_attr=fluid.ParamAttr(
                    name="tdm.bw_emb.weight", 
                    initializer=paddle.fluid.initializer.UniformInitializer())
                    # initializer=paddle.fluid.initializer.ConstantInitializer(value=0.0))
            )
            
            self.embedding_fetch_var_names = [output.name]
            self.exe.run(startup)


def get_itemset_given_ancestor(pi_new, node):
    res = []
    for ci, code in pi_new.items():
        if code == node:
            res.append(ci)
    return res
                    
# you need to define your sample_set
def get_sample_set(ck, sample_nums=-1):
    if not os.path.exists("samples/samples_{}.json".format(ck)):
        return []
    with open("samples/samples_{}.json".format(ck), 'r') as f:
        all_samples = json.load(f)
    if sample_nums > 0:
        size = len(all_samples)
        if (size > sample_nums):
            sample_set = np.random.choice(range(size), size=sample_nums, replace=False).tolist()
            return [all_samples[s] for s in sample_set]
    else:
        return all_samples

def get_weights(C_ni, idx, edge_weights, ni, children_of_ni_in_level_l, tree, node_emb_size=64, init_model_path=""):
    """use the user preference prediction model to calculate the required weights

    Returns:
        all weights

    Args:
        C_ni (item, required): item set whose ancestor is the non-leaf node ni
        ni (node, required): a non-leaf node in level l-d
        children_of_ni_in_level_l (list, required): the level l-th children of ni
        tree (tree, required): the old tree (\pi_{old})

    """
    print("begin idx: {}, C_ni: {}.".format(idx, len(C_ni)))
    tree_node_num = tree.tree_max_node()
    print("tree_node_num: ", tree_node_num)
    prediction_model = UserPreferenceModel(init_model_path, tree_node_num, node_emb_size)

    for ck in C_ni:
        _weights = list()
        # the first element is the list of nodes in level l
        _weights.append([])
        # the second element is the list of corresponding weights
        _weights.append([])

        samples = get_sample_set(ck)  
        for node in children_of_ni_in_level_l:
            path_to_ni = tree.get_travel_path(node, ni)
            if len(samples) == 0:
                weight = 0.0
            else:
                weight = prediction_model.calc_prediction_weight(samples, path_to_ni)
            
            _weights[0].append(node)
            _weights[1].append(weight)
        edge_weights.update({ck: _weights})
    print("end idx: {}, C_ni: {}, edge_weights: {}.".format(idx, len(C_ni), len(edge_weights)))


def assign_parent(tree, l_max, l, d, ni, C_ni):
    """implementation of line 5 of Algorithm 2

    Returns: 
        updated \pi_{new}

    Args:
        l_max (int, required): the max level of the tree
        l (int, required): current assign level
        d (int, required): level gap in tree_learning
        ni (node, required): a non-leaf node in level l-d
        C_ni (item, required): item set whose ancestor is the non-leaf node ni
        tree (tree, required): the old tree (\pi_{old})
    """
    # get the children of ni in level l
    children_of_ni_in_level_l = tree.get_children_given_ancestor_and_level(
        ni, l)

    print(children_of_ni_in_level_l)
    # get all the required weights
    edge_weights = mp.Manager().dict()

    mp_run(C_ni, 12, get_weights, edge_weights, ni, children_of_ni_in_level_l, tree)

    print("finish calculate edge_weights. {}.".format(len(edge_weights)))
    # assign each item to the level l node with the maximum weight
    assign_dict = dict()
    for ci, info in edge_weights.items():
        assign_candidate_nodes = np.array(info[0], dtype=np.int64)
        assign_weights = np.array(info[1], dtype=np.float32)
        sorted_idx = np.argsort(-assign_weights)
        sorted_weights = assign_weights[sorted_idx]
        sorted_candidate_nodes = assign_candidate_nodes[sorted_idx]
        # assign item ci to the node with the largest weight
        max_weight_node = sorted_candidate_nodes[0]
        if max_weight_node in assign_dict:
            assign_dict[max_weight_node].append(
                (ci, 0, sorted_candidate_nodes, sorted_weights))
        else:
            assign_dict[max_weight_node] = [
                (ci, 0, sorted_candidate_nodes, sorted_weights)]

    edge_weights = None

    # get each item's original assignment of level l in tree, used in rebalance process
    origin_relation = tree.get_pi_relation(C_ni, l)
    # for ci in C_ni:
    #     origin_relation[ci] = self._tree.get_ancestor(ci, l)

    # rebalance
    max_assign_num = int(math.pow(2, l_max - l))
    processed_set = set()

    while True:
        max_assign_cnt = 0
        max_assign_node = None

        for node in children_of_ni_in_level_l:
            if node in processed_set:
                continue
            if node not in assign_dict:
                continue
            if len(assign_dict[node]) > max_assign_cnt:
                max_assign_cnt = len(assign_dict[node])
                max_assign_node = node

        if max_assign_node == None or max_assign_cnt <= max_assign_num:
            break

        # rebalance
        processed_set.add(max_assign_node)
        elements = assign_dict[max_assign_node]
        elements.sort(key=lambda x: ( 
            int(max_assign_node != origin_relation[x[0]]), -x[3][x[1]]))
        for e in elements[max_assign_num:]:
            idx = e[1] + 1
            while idx < len(e[2]):
                other_parent_node = e[2][idx]
                if other_parent_node in processed_set:
                    idx += 1
                    continue
                if other_parent_node not in assign_dict:
                    assign_dict[other_parent_node] = [(e[0], idx, e[2], e[3])]
                else:
                    assign_dict[other_parent_node].append(
                        (e[0], idx, e[2], e[3]))
                break

        del elements[max_assign_num:]

    pi_new = dict()
    for parent_code, value in assign_dict.items():
        max_assign_num = int(math.pow(2, l_max - l))
        assert len(value) <= max_assign_num
        for e in value:
            assert e[0] not in pi_new
            pi_new[e[0]] = parent_code

    return pi_new


def process(nodes, idx, pi_new_final, tree, l, d):
    l_max = tree.height() - 1
    print("begin to process {}".format(idx))
    for ni in nodes:
        C_ni = get_itemset_given_ancestor(pi_new_final, ni)
        print("begin to handle {}, have {} items.".format(ni, len(C_ni)))
        pi_star = assign_parent(tree, l_max, l, d, ni, C_ni)

        # update pi_new according to the found optimal pi_star
        for item, node in pi_star.items():
            pi_new_final.update({item: node})
        print("end to handle {}.".format(ni))

def tree_learning(tree, d, output_filename):
    l = d

    pi_new = dict()

    all_items = tree.get_all_items()
    pi_new = tree.get_pi_relation(all_items, l-d)

    pi_new_final = mp.Manager().dict()
    pi_new_final.update(pi_new)

    del all_items
    del pi_new

    while d > 0:
        print("begin to re-assign {} layer by {} layer.".format(l, l-d))
        nodes = tree.get_nodes_given_level(l - d, True)
        real_process_num = mp_run(nodes, 12, process, pi_new_final, tree, l, d)
        print("real_process \'process\' function num: {}".format(real_process_num))
        # for ni in nodes:
        #     C_ni = get_itemset_given_ancestor(pi_new, ni)
        #     print("begin to handle {}, have {} items.".format(ni, len(C_ni)))
        #     pi_star = assign_parent(tree, l_max, l, d, ni, C_ni)

        #     # update pi_new according to the found optimal pi_star
        #     for item, node in pi_star.items():
        #         pi_new[item] = node
        #     print("end to handle {}.".format(ni))

        d = min(d, l_max - l)
        l = l + d
    
    
if __name__ == '__main__':
    index_wrapper = IndexWrapper()
    index_wrapper.insert_tree_index("ub_first_tree", "/work/ub.pb")
    tree = index_wrapper.get_tree_index("ub_first_tree")
    tree_learning(tree, 7, "output.pb")
    