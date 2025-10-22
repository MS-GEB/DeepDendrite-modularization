import numpy as np
from neuron import h
from utils import IdManager
from commonutils import timeit
import os
import copy

h.Random().Random123_globalindex(1234)
h.load_file('stdgui.hoc')
pc = h.ParallelContext()
# set number of threads
pc.nthread(1, 0)
rank, nhost = int(pc.id()), int(pc.nhost())

# Not used by DeepDendrite
def train(net, x_train, y_train, lr_rate=0.005):
    h.dt = 10
    net.is_train(start=200, end=500, lr_rate=lr_rate)
    net.load_weights()

    # procedure for doing and plotting each simulation
    def train_single(input, target):
        net.set_stim(input, target)

        h.t = 0
        h.tstop = 500
        h.finitialize(0.0)

        pc.psolve(int(h.tstop))
        # timeit("stim time", rank)

        net.save_weights()
        net.load_weights()
        # timeit("update time", rank)

    # run the experiments, store the results
    for i, (x, y) in enumerate(zip(x_train, y_train)):
        if rank == 0:
            print(f"train #: {i:d}", end='\r')
        train_single(x, y)

# Not used by DeepDendrite
def test(net, x_test, y_test):
    h.dt = 10
    net.is_test()
    net.load_weights()

    # procedure for doing and plotting each simulation
    def test_single(input):
        net.set_stim(input)

        h.t = 0
        h.tstop = 500

        # setup recording
        v_outs = []
        for i, gid_cell in enumerate(net.output.gids_cell):
            if net.output.pc.gid_exists(gid_cell):
                cell = net.output.pc.gid2cell(gid_cell)
                v_out = h.Vector()
                v_out.record(net.output.cell_soma_sec(cell)(0.5)._ref_v)
                v_outs.append(v_out)

        h.finitialize(0.0)

        pc.psolve(int(h.tstop))

        if v_outs:
            v_outs = np.mean(np.array(v_outs)[:, int(200 / h.dt):], axis=1)
        # v_outs_gather = pc.py_gather(v_outs, 0)
        v_outs_gather = pc.py_alltoall([v_outs if i == 0 else None for i in range(nhost)])
        if rank == 0:
            v_outs_flat = []
            for i in range(len(v_outs_gather[0])):
                for j in range(nhost):
                    try:
                        v_outs_flat.append(v_outs_gather[j][i])
                    except IndexError:
                        pass
            pred = np.argmax(v_outs_flat)
            return pred
        return -1

    # run the experiments, store the results
    preds = []
    for i, x in enumerate(x_test):
        if rank == 0:
            print(f"test #: {i:d}", end='\r')
        preds.append(test_single(x))

    if rank == 0:
        count = 0
        for pred, tgt in zip(preds, y_test):
            if pred == tgt:
                count += 1
        print(f"\nacc: {count / len(y_test):f}")

    return preds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mnist')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-f', '--folder', type=str, default='./coredat/demo_train')
    parser.add_argument('-w', '--weights', type=str, default='./coredat/demo_test/weights_51to60.npy')

    args, _ = parser.parse_known_args()

    if args.dataset == 'mnist':
        from demo_networks import MnistFCNet as MyNet
        with np.load('./datasets/mnist.npz', allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

        x_train = x_train.reshape((-1, 1*28*28))
        y_train = y_train.reshape((-1,))
        x_test = x_test.reshape((-1, 1*28*28))
        y_test = y_test.reshape((-1,))

        x_train = x_train.astype('float64') / 255
        y_train = y_train.astype('int32')
        x_test = x_test.astype('float64') / 255
        y_test = y_test.astype('int32')
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")

    batch_size = args.batch_size

    idm = IdManager()

    pc.barrier()
    timeit(None, rank)

    net_list = []
    for i in range(batch_size):
        print(f"net {i:d}")
        net = MyNet(idm=idm, seed=1234)
        #net.setup_backward()
        net_list.append(net)

    for inet, net in enumerate(net_list):
        for mod_dict in net.seg2synlist.values():
            for synlist in mod_dict.values():
                synlist.reverse()
        net.is_test()
        # for i, layer in enumerate(net.train_layers):
        #     layer.weight = copy.deepcopy(net_list[0].train_layers[i].weight)
        net.load_weights()
        net.set_stim(x_train[inet], None)

    pc.barrier()
    timeit("create network", rank)

    h.cvode.cache_efficient(1)
    pc.setup_transfer()
    pc.set_maxstep(20)
    h.stdinit()
    pc.barrier()

    bbcore_folder = args.folder
    if rank == 0 and not os.path.isdir(bbcore_folder):
        os.mkdir(bbcore_folder)

    #net.set_stim(x_train[0], y_train[0])
    out_gap_params = []
    out_stim_amp = []
    out_label_params = []
    out_weight_params = []
    out_delta_weight_params = []
    out_param2rec = []

    for net in net_list:
        gap_lines = net.gen_gap_param_file()
        stim_lines = net.gen_stim_file(net.input.stims)
        #label_lines = net.gen_label_file(net.output.synlist_out2inter)
        #w_lines, dw_lines = net.gen_set_weight_file()

        out_gap_params += gap_lines
        out_stim_amp += stim_lines

        #net.gen_weight_file(bbcore_folder, "weights")

    with open(os.path.join(bbcore_folder, "stim_amp"), "w") as f:
        f.writelines(f"{len(out_stim_amp):d}\n")
        f.writelines(out_stim_amp)
    with open(os.path.join(bbcore_folder, "gap_params"), "w") as f:
        f.writelines(f"{len(out_gap_params):d}\n")
        f.writelines(out_gap_params)

    weights = np.load(args.weights) # load trained weights, shape (nweight, nepoch)

    nweights, nepoch = weights.shape

    out_set_weights_file = []
    for net in net_list:
        out_w = net.gen_trainable_syn_params()
        for i, w_line in enumerate(out_w):
            out_set_weights_file.append(w_line[:-1] + f" {weights[i, -1]:.6f}\n")
    with open(os.path.join(bbcore_folder, "weights_param_val"), "w") as f:
        f.writelines(f"{len(out_set_weights_file):d}\n")
        f.writelines(out_set_weights_file)

    out_param2rec = []
    for net in net_list:
        for cell in net.output.cells:
            sec = net.output.cell_soma_sec(cell)
            out_param2rec.append("0 %s 0.5 soma 0 v\n"%sec)

    with open(os.path.join(bbcore_folder, "param2rec"), "w") as f:
        f.writelines(f"{len(out_param2rec):d} -1 200 500\n")
        f.writelines(out_param2rec)

    pc.nrnbbcore_write(bbcore_folder)
    os.system(f"mv ./secnamelist0.txt ./map_node2sec0 {bbcore_folder}")
    #train(net, x_train, y_train, lr_rate=0.005)
    pc.barrier()
    timeit("write data", rank)
    #timeit("training time", rank)

    #test(net, x_test, y_test)
    #pc.barrier()
    #timeit("test time", rank)

    if rank == 0:
        print("Done")

    pc.barrier()
    h.quit()
