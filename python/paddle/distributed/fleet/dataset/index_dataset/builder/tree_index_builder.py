from paddle.fluid.proto import index_dataset_pb2
import numpy as np
import struct

class TreeIndexBuilder:
    def __init__(self, name=None, branch=2):
        self.name = name
        self.branch = branch

    def tree_init_by_category(self, input_filename, output_filename):
        class Item:
            def __init__(self, item_id, cat_id):
                self.item_id = item_id
                self.cat_id = cat_id
                self.code = 0

            def __lt__(self, other):
                return self.cat_id < other.cat_id or \
                    (self.cat_id == other.cat_id and
                     self.item_id < other.item_id)
        
        items = []
        item_id_set = set()
        with open(input_filename, 'r') as f:
            for line in f:
                iterobj = line.split()
                item_id = int(iterobj[0])
                cat_id = int(iterobj[1])
                if item_id not in item_id_set:
                    items.append(Item(item_id, cat_id))
                    item_id_set.add(item_id)
        del item_id_set
        items.sort()

        def gen_code(start, end, code):
            if end <= start:
                return
            if end == start + 1:
                items[start].code = code
                return
            num = int((end - start) / self.branch)
            remain = int((end - start) % self.branch)
            for i in range(self.branch):
                _sub_end = start + i * num
                if (remain > 0):
                    remain -= 1
                    _sub_end += 1
                _sub_end = min(_sub_end, end)
                gen_code(start, _sub_end, self.branch * code + self.branch - i)
                start = _sub_end

            # mid = int((start + end) / 2)
            # gen_code(mid, end, 2 * code + 1)
            # gen_code(start, mid, 2 * code + 2)

        gen_code(0, len(items), 0)
        ids = np.array([item.item_id for item in items])
        codes = np.array([item.code for item in items])
        #data = np.array([[] for i in range(len(ids))])
        self.build(output_filename, ids, codes)

    def tree_init_by_kmeans(self):
        pass

    def build(self, output_filename, ids, codes, data=None, id_offset=None):
        # process id offset
        if not id_offset:
            max_id = 0
            for id in ids:
                if id > max_id:
                    max_id = id
            id_offset = max_id + 1

        # sort by codes
        argindex = np.argsort(codes)
        codes = codes[argindex]
        ids = ids[argindex]

        # Trick, make all leaf nodes to be in same level
        min_code = 0
        max_code = codes[-1]
        while max_code > 0:
            min_code = min_code * 2 + 1
            max_code = int((max_code - 1) / 2)

        for i in range(len(codes)):
            while codes[i] < min_code:
                codes[i] = codes[i] * 2 + 1

        filter_set = set()
        max_level = 0
        tree_meta = index_dataset_pb2.TreeMeta()

        with open(output_filename, 'wb') as f:
            for id, code in zip(ids, codes):
                node = index_dataset_pb2.Node()
                node.id = id
                node.is_leaf = True
                node.probability = 1.0
                
                kv_item = index_dataset_pb2.KVItem()
                kv_item.key = self._make_key(code)
                kv_item.value = node.SerializeToString()
                self._write_kv(f, kv_item.SerializeToString())

                ancessors = self._ancessors(code)
                if len(ancessors) + 1 > max_level:
                    max_level = len(ancessors) + 1

                for ancessor in ancessors:
                    if ancessor not in filter_set:
                        node = index_dataset_pb2.Node()
                        node.id = id_offset + ancessor  # id = id_offset + code
                        node.is_leaf = False
                        node.probability = 1.0
                        kv_item = index_dataset_pb2.KVItem()
                        kv_item.key = self._make_key(ancessor)
                        kv_item.value = node.SerializeToString()
                        self._write_kv(f, kv_item.SerializeToString())
                        filter_set.add(ancessor)
                        
            tree_meta.branch = self.branch
            tree_meta.height = max_level
            kv_item = index_dataset_pb2.KVItem()
            kv_item.key = '.tree_meta'
            kv_item.value = tree_meta.SerializeToString()
            self._write_kv(f, kv_item.SerializeToString())

    def _ancessors(self, code):
        ancs = []
        while code > 0:
            code = int((code - 1) / 2)
            ancs.append(code)
        return ancs

    def _make_key(self, code):
        return str(code)

    def _write_kv(self, fwr, message):
        fwr.write(struct.pack('i', len(message)))
        fwr.write(message)