from paddle.fluid.proto import index_dataset_pb2
import numpy as np
import struct

# 放在Cpp
# class Item:   # 每个Item对应的 ID - 类型 - 路径编码 - 概率
#     def __init__(self, item_id, cat_id):  #但是 pair<cat_id, probility>
#         self.item_id = item_id
#         self.cat_id = cat_id
#         self.path_codes=[]

#     def add(self,path_code,probility):
#         self.path_codes.append((path_code,probility))

#     def __lt__(self, other):   
#         return self.cat_id < other.cat_id or \
#             (self.cat_id == other.cat_id and
#                 self.item_id < other.item_id)  

# class Node:  # <node_id,[<node_id, probility>] >
#     def __init__(self, node_id0):
#         self.node_id=node_id0
#         self.node_next=[]

#     def add(self,node_id1, probility):
#         self.node_next.append((node_id1, probility))

class GraphIndexBuilder:
    def __init__(self, name=None, width, depth):
        self.name = name
        self.width = width
        self.depth = depth

    def gen_code():
        pro=0.5
        nodeList=[]   
        node_tmp_id=0
        for di in range(self.depth):
            for wi in range(self.width):
                node_id0=node_tmp_id+1
                curnode=Node(node_id0)
                # node.node_id0=node_tmp_id
                if di == self.depth-1:
                    nodeList.append(curnode)
                elif di < self.depth-1:                    
                    for wj in range(self.width):
                        node_id1=node_id0-wi+self.width+wj
                        curnode.add(node_id1,probility)
                    nodeList.append(curnode)
        return nodeList

    def gen_path_code(self,item):
#       graph--->index 
#       item--->pathCode;
        pathToCode
        

    def graph_init_by_category(self, input_filename, output_filename):
        items = []
        item_id_set = set()
        with open(input_filename, 'r') as f:  # 取到文件中的itemId-catId
            for line in f:
                iterobj = line.split()
                item_id = int(iterobj[0])
                cat_id = int(iterobj[1])
                if item_id not in item_id_set:    
                    items.append(Item(item_id, cat_id))  
                    # items:[Item] <=== Item ===> item_id, cat_id, pathcode, probility 
                    item_id_set.add(item_id)
        del item_id_set
        items.sort()  # # items:[Item] ===> item_id, cat_id, pathcode, probility 

        def get_code(width,depth,probility):
            if width 


        def gen_path_code(start, end, pathcode, probility):  # gen_path_code  为每个Item产生一个路径编码
            if end <= start:
                return               
            if end == start + 1:        
                items[start].pathcode = pathcode     # 不断将itemID-PathCode 添加到items中 脚标自0增长；
                items[start].probility = probility
                return
            num = int((end - start) / self.branch)  
            remain = int((end - start) % self.branch)
            for i in range(self.branch):  
                _sub_end = start + i * num   #不断更新start-end; 
                if (remain > 0):
                    remain -= 1
                    _sub_end += 1
                _sub_end = min(_sub_end, end)
                gen_path_code(start, _sub_end, self.branch * code + self.branch - i)
                start = _sub_end

            # mid = int((start + end) / 2)
            # gen_code(mid, end, 2 * code + 1)
            # gen_code(start, mid, 2 * code + 2)

        gen_path_code(0, len(items), 0)   # 为每个Item 生成路径编号
        ids = np.array([item.item_id for item in items])   # itemID
        codes = np.array([item.code for item in items])   # codes 路径编码
        #data = np.array([[] for i in range(len(ids))])
        self.build(output_filename, ids, codes)   # 将itemID 与 路径编码 build放在文件

    def graph_init_by_kmeans(self):
        pass

    def build(self, output_filename, ids, codes, data=None, id_offset=None):
        # process id offset
        if not id_offset:    # 
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
        graph_meta = index_dataset_pb2.GraphMeta()

        with open(output_filename, 'wb') as f:
            for id, code in zip(ids, codes):   # itemId - pathCode
                node = index_dataset_pb2.Node()
                node.id = id                   # 节点映射的ItemID
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
        
