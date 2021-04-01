from paddle.fluid.core import IndexWrapper, TreeIndex, GraphIndex
from builder import TreeIndexBuilder, GraphIndexBuilder

class Index(object):
    def __init__(self, name):
        self._name = name

class GraphIndex(Index):
    def __init__(self, name):
        super(GraphIndex, self).__init__(name)
        self._graph=None
        self._builder=GraphIndexBuilder(name)
        self._wrapper=IndexWrapper()

    def _init_by_category(self,input_filename,output_filename):
        self._builder.graph_init_by_category(input_filename, output_filename)
        self._init_graph(output_filename)

    def _init_by_kmeans(self,output_filename):
        self._builder.graph_init_by_kmeans(output_filename)
        self._init_graph(output_filename)

    def _init_graph(self,filename):
        self._graph=self._wrapper.insert_graph_index(self._name, filename)

    def get_path_given_item(self,item_id):
        return self._graph.get_path_given_item(item_id)

    def get_item_given_path(self,path_code):
        return self._graph.get_item_given_path(path_code)

    def graph_width(self):
        return self._graph.width()

    def graph_depth(self):
        return self._graph.depth()

class TreeIndex(Index):
    def __init__(self, name):
        super(TreeIndex, self).__init__(name)
        self._tree = None
        self._builder = TreeIndexBuilder(name)
        self._wrapper = IndexWrapper()

    def _init_by_category(self,input_filename, output_filename):
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