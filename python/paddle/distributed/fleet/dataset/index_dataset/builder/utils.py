from multiprocessing import Process 
import time

def mp_run(data, process_num, func, *args):
    """ run func with multi process
    """
    level_start = time.time()
    partn = max(len(data) / process_num, 1)
    start = 0
    p_idx = 0
    ps = []
    while start < len(data):
        local_data = data[start: start + partn]
        start += partn
        p = Process(target=func, args=(local_data, p_idx) + args)
        ps.append(p)
        p.start()
        p_idx += 1       
    for p in ps:
        p.join()

    for p in ps:
        p.terminate()
    return p_idx

def prepare_sample_set(files, idx, process_num=12, feature_num=69):
    def parse_data(files, idx, feature_num=69):
        print("parse data, {} : {}".format(idx, files))
        history_ids = [0] * feature_num
        samples = dict()
        process = 0
        for filename in files:
            process += 1
            print("process {} / {}.".format(process, len(files)))
            with open(filename) as f:
                print("Begin to handle {}.".format(filename))
                for line in f:
                    features = line.strip().split("|")[2].split(";")
                    item_id = None
                    for item in features:
                        f = item.split("@")
                        if f[0] == "train_unit_id":
                            item_id = int(f[1].split(":")[0])
                        else:
                            history_ids[int(f[0].split('_')[1]) - 1] = int(f[1])
                    if item_id is None:
                        continue
                    if item_id not in samples:
                        samples[item_id] = list()
                    samples[item_id].append(history_ids) 

        with open("parse_data_{}.json".format(idx), 'w') as json_file:
            json.dump(samples, json_file)

    real_process_num = mp_run(files, process_num, parse_data, feature_num)
    
    num = 0
    all_samples = dict()
    print(real_process_num)
    for i in range(real_process_num):
        with open("parse_data_{}.json".format(i), 'r') as json_file:
            print("parse_data_{}.json".format(i))
            each_samples = json.load(json_file)
            print("finish load. ", len(each_samples))
            for key in each_samples:
                if key not in self._samples:
                    all_samples[key] = []
                all_samples[key] += each_samples[key]
                num += len(each_samples[key])
            print(num)
    print(len(all_samples), num)

    for ck in all_samples:
            with open("samples/samples_{}.json".format(ck), 'w') as f:
                json.dump(self._samples[ck], f)