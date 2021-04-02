from paddle.fluid.core import IndexSampler
from .index import Index, TreeIndex, GraphIndex

class IndexDataset(object):
    def __init__(index):
        if not isinstance(index, Index):
            raise TypeError("index must be instance of Index.")
        self._index = index
        self._layerwise_sampler = None
        self._beamsearch_sampler = None

    def init_layerwise_sampler(self, layer_sample_counts, name):
        assert self._layerwise_sampler is None
        self._layerwise_sampler = IndexSampler("by_layerwise", name)
        self._layerwise_sampler.init_layerwise_conf(layer_sample_counts)

    def init_beamsearch_sampler(self, top_k):
        assert self._beamsearch_sampler is None
        self._beamsearch_sampler = IndexSampler("by_beamsearch", name)
        self._beamsearch_sampler.init_beamsearch_conf(top_k)

    def layerwise_sample(user_input, index_input):
        if self._layerwise_sampler is None:
            raise ValueError("please init layerwise_sampler first.")
        return self._layerwise_sampler.sample(user_input, index_input)

    def beamsearch_sampler(user_input, index_input):
        if self._beamsearch_sampler is None:
            raise ValueError("please init _beamsearch_sampler first.")
        return self._beamsearch_sampler.sample(user_input, index_input)