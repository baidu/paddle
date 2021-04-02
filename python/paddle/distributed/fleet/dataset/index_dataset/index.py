from paddle.fluid.core import IndexWrapper, TreeIndex
from builder import TreeIndexBuilder, GraphIndexBuilder


class Index(object):
    def __init__(self, name):
        self._name = name


class GraphIndex(Index):
    def __init__(self, name, width, height, item_path_nums):
        super(GraphIndex, self).__init__(name)
        self._graph = None
        self.name = name
        self._builder = GraphIndexBuilder(name, width, height, item_path_nums)
        self._wrapper = IndexWrapper()

    def _init_by_random(self, input_filename, output_filename):
        self._builder.graph_init_by_random(input_filename, output_filename)
        self._init_graph(output_filename)

    def _init_graph(self, filename):
        self._wrapper.insert_graph_index(self._name, filename)
        self._graph = self._wrapper.get_graph_index(self.name)

    def get_path_of_item(self, id):
        if isinstance(id, list):
            assert len(id) > 0
            assert isinstance(id[0], int)
            return self._graph.get_path_of_item(id)
        elif isinstance(id, int):
            return self._graph.get_path_of_item([id])
        else:
            raise ValueError(
                "Illegal input type {}, required list or int".format(type(id)))

    def get_item_of_path(self, path):
        if isinstance(path, list):
            assert len(path) > 0
            assert isinstance(path[0], int)
            return self._graph.get_item_of_path(path)
        elif isinstance(path, int):
            return self._graph.get_item_of_path([path])
        else:
            raise ValueError(
                "Illegal input type {}, required list or int".format(type(id)))


class TreeIndex(Index):
    def __init__(self, name):
        super(TreeIndex, self).__init__(name)
        self._tree = None
        self._builder = TreeIndexBuilder(name)
        self._wrapper = IndexWrapper()

    def _init_by_category(self, input_filename, output_filename):
        self._builder.tree_init_by_category(input_filename, output_filename)
        self._init_tree(output_filename)

    def _init_by_kmeans(self, output_filename):
        self._builder.tree_init_by_kmeans(output_filename)
        self._init_tree(output_filename)

    def _init_tree(self, filename):
        self._tree = self._wrapper.insert_tree_index(self._name, filename)

    def get_nodes_given_level(self, level):
        return self._tree.get_nodes_given_level(level)

    def get_parent_path(self, id, start_level=-1):
        return self._tree.get_parent_path(id, start_level)

    def tree_height(self):
        return self._tree.height()

    def tree_branch(self):
        return self._tree.branch()
