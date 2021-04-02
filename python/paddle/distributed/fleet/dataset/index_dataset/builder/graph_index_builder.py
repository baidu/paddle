from paddle.fluid.proto import index_dataset_pb2
import numpy as np
import struct


class GraphIndexBuilder:
    def __init__(self, name=None, width=1, height=1, item_path_nums=1):
        self.name = name
        self.width = width  # D
        self.height = height  # K
        self.item_path_nums = item_path_nums

    def graph_init_by_random(self, input_filename, output_filename):
        total_path_num = pow(self.height, self.width)  # K^D
        item_path_dict = {}

        with open(input_filename, 'r') as f_in:
            for line in f_in:
                item_id = int(line.split()[0])
                if item_id not in item_path_dict:
                    item_path_dict[item_id] = []
                else:
                    continue

                item_path_dict[item_id] = list(np.random.randint(
                    0, total_path_num, self.item_path_nums))

        self.build(output_filename, item_path_dict)

    def build(self, output_filename, item_path_dict):
        graph_meta = index_dataset_pb2.GraphMeta()
        graph_meta.width = self.width
        graph_meta.height = self.height
        graph_meta.item_path_nums = self.item_path_nums

        with open(output_filename, 'wb') as f_out:
            kv_item = index_dataset_pb2.KVItem()
            kv_item.key = '.graph_meta'
            kv_item.value = graph_meta.SerializeToString()

            self._write_kv(f_out, kv_item.SerializeToString())

            for item in item_path_dict:
                graph_item = index_dataset_pb2.GraphItem()
                graph_item.item_id = item
                for path in item_path_dict[item]:
                    graph_item.path_id.append(path)
                node_kv_item = index_dataset_pb2.KVItem()
                node_kv_item.key = str(item)
                node_kv_item.value = graph_item.SerializeToString()
                self._write_kv(f_out, node_kv_item.SerializeToString())

    def _write_kv(self, fwr, message):
        fwr.write(struct.pack('i', len(message)))
        fwr.write(message)
