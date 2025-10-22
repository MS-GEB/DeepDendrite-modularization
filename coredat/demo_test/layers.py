import numpy as np
from neuron import h
from utils import *

mod_name_list = ['Linear_Grad_Syn', 'Linear_Syn', 'ReLU_Grad_Syn', 'ReLU_Syn', 'Softmax_Syn', 'Softmax_Syn_v2',
                 'Linear_Grad_Syn_avg', 'Linear_Syn_avg', 'ReLU_Grad_Syn_avg', 'ReLU_Syn_avg', 'Softmax_Syn_avg',]

class BaseLayer:
    def __init__(self, pc, idm, seg2synlist, target_var_list, rng = None, seed=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = rng
        self.pc = pc
        self.nhost = int(self.pc.nhost())
        self.ihost = int(self.pc.id())
        self.idm = idm
        self.seg2synlist = seg2synlist
        self.target_var_list = target_var_list
        self.requires_input_grad = False

    def get_param_info(self, mech):
        seg = mech.get_segment()
        mname = str(mech)
        i1 = mname.find('[')
        if i1 > -1:
            mname = mname[:i1]
        mid = self.seg2synlist[seg][mname].index(mech)
        #mid = self.get_mech_id(mech, self.seg2synlist[seg])
        return seg.sec, seg.x, mname, mid

    def get_mech_id(self, mech, synlist):
        idx = 0
        mname = str(mech)
        i1 = mname.find('[')
        for i, syn in enumerate(synlist):
            if mech == syn:
                break
            tgt_mname = str(syn)
            i2 = tgt_mname.find('[')
            if i1 > -1 and i2 > -1 and mname[:i1] == tgt_mname[:i2]:
                idx += 1
            else:
                if mname == tgt_mname:
                    idx += 1
        return idx

    def add_mech_to_synlist(self, seg, mech):
        mname = str(mech)
        i1 = mname.find('[')
        if i1 > -1:
            mname = mname[:i1]
        if not seg in self.seg2synlist:
            self.seg2synlist[seg] = {}
            for mod_name in mod_name_list:
                self.seg2synlist[seg][mod_name] = []
        if not mech in self.seg2synlist[seg][mname]:
            self.seg2synlist[seg][mname].append(mech)

    def add_target_var(self, seg, mech, var, sid):
        self.target_var_list.append((sid, seg, mech, var))


class NetStimInput(BaseLayer):
    def __init__(self, N_in, **kwargs):
        super(NetStimInput, self).__init__(**kwargs)
        self.N_in = N_in

        self._create_cells()
        self._connect_cells()

    def cell_instance(self):
        cell = h.PointNeuron()
        for sec in cell.all:
            sec.Ra = 100
            sec.cm = 1
            sec.g_pas = 1e-5
        return cell

    def cell_soma_sec(self, cell):
        return cell.soma

    def _create_cells(self):
        # netstim
        self.stims = []
        self.gids_stim = self.idm.alloc_gid(self.N_in)
        for i, gid_stim in enumerate(self.gids_stim):
            if (i % self.nhost) != self.ihost:
                continue
            self.pc.set_gid2node(gid_stim, self.ihost)
            stim = h.NetStim()
            stim.interval = 1e9
            stim.start = 1e9
            stim.number = 0
            # poisson
            stim.noise = 0
            stim.noiseFromRandom123(i, 0, 0)
            self.stims.append(stim)
            ncstim = h.NetCon(stim, None)
            self.pc.cell(gid_stim, ncstim)

        # cell
        h.load_file("import3d.hoc")
        h.load_file("PointNeuron.hoc")
        self.cells = []
        self.gids_cell = self.idm.alloc_gid(self.N_in)
        for i, gid_cell in enumerate(self.gids_cell):
            if (i % self.nhost) != self.ihost:
                continue
            self.pc.set_gid2node(gid_cell, self.ihost)
            cell = self.cell_instance()
            self.cells.append(cell)
            sec = self.cell_soma_sec(cell)
            nc = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
            self.pc.cell(gid_cell, nc)

    def _connect_cells(self):
        self.synlist_stim2in = []
        self.ncstimlist = []
        for gid_stim, gid_cell in zip(self.gids_stim, self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                cell = self.pc.gid2cell(gid_cell)
                syn = h.ExpSyn(self.cell_soma_sec(cell)(0.5))
                syn.tau = 5
                syn.e = 1
                self.synlist_stim2in.append(syn)
                ncstim = self.pc.gid_connect(gid_stim, syn)
                ncstim.delay = 1
                ncstim.weight[0] = 0.05
                self.ncstimlist.append(ncstim)

        # output sids
        self.sids_out = self.idm.alloc_sid(self.N_in)
        for gid_cell, sid_out in zip(self.gids_cell, self.sids_out):
            if self.pc.gid_exists(gid_cell):
                cell = self.pc.gid2cell(gid_cell)
                sec = self.cell_soma_sec(cell)
                #self.pc.source_var(sec(0.5)._ref_v, sid_out, sec=sec)
                self.idm.sid2param[sid_out] = (sec(0.5), None, "v") # (seg, mech, var)
        if self.nhost > 1:
            self.pc.barrier()

    def set_input(self, input):
        assert len(input) == self.N_in
        for gid_stim, x in zip(self.gids_stim, input):
            if self.pc.gid_exists(gid_stim):
                stim = self.pc.gid2cell(gid_stim)
                stim.interval = 50 / (x + 0.01)
                stim.start = 90 + stim.interval
                stim.number = 1e9


class OutputPoint(BaseLayer):
    def __init__(self, N_in, N_out, sids_in, in_activation='ReLU', **kwargs):
        super(OutputPoint, self).__init__(**kwargs)
        self.N_in = N_in
        self.N_out = N_out
        assert len(sids_in) == N_in
        self.sids_in = sids_in
        if in_activation not in ['Linear', 'ReLU']:
            raise ValueError("Undefined input activation, must be one of 'Linear' or 'ReLU'")
        self.in_activation = in_activation

        self._create_cells()
        self._connect_cells()

    def cell_instance(self):
        cell = h.PointNeuron()
        for sec in cell.all:
            sec.Ra = 100
            sec.cm = 1
            sec.g_pas = 1e-4
        return cell

    def cell_soma_sec(self, cell):
        return cell.soma

    def inter_instance(self):
        cell = h.PointNeuron()
        for sec in cell.all:
            sec.Ra = 100
            sec.cm = 1
            sec.g_pas = 1
        return cell

    def inter_soma_sec(self, inter):
        return inter.soma

    def grad_instance(self):
        return self.inter_instance()

    def grad_soma_sec(self, grad):
        return grad.soma

    def _create_cells(self):
        h.load_file("import3d.hoc")
        h.load_file("PointNeuron.hoc")
        self.cells = []
        self.gids_cell = self.idm.alloc_gid(self.N_out)
        for i, gid_cell in enumerate(self.gids_cell):
            if (i % self.nhost) != self.ihost:
                continue
            self.pc.set_gid2node(gid_cell, self.ihost)
            cell = self.cell_instance()
            self.cells.append(cell)
            sec = self.cell_soma_sec(cell)
            nc = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
            self.pc.cell(gid_cell, nc)

    def _connect_cells(self):
        # weight matrix input-to-output, all to all
        limit = np.sqrt(6 / self.N_in)
        self.weight = self.rng.uniform(-limit, limit, (self.N_in, self.N_out))
        self.bias = np.zeros((self.N_out,))
        self.delta_weight = np.zeros((self.N_in, self.N_out))
        self.delta_bias = np.zeros((self.N_out,))

        # transfer resistance
        tmpcell = self.cell_instance()
        impd = h.Impedance()
        impd.loc(0.5, sec=self.cell_soma_sec(tmpcell))
        impd.compute(0)
        self.cell_rsoma = impd.transfer(0.5, sec=self.cell_soma_sec(tmpcell))
        del tmpcell, impd

        # y = W * f(x)
        self.synarray_forward = np.full((self.N_in, self.N_out), fill_value=None)
        self.synarray_bias = np.full((self.N_out,), fill_value=None)
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                cell = self.pc.gid2cell(gid_cell)
                for j, sid_v_in in enumerate(self.sids_in):
                    if self.in_activation == 'Linear':
                        syn = h.Linear_Syn_avg(self.cell_soma_sec(cell)(0.5))
                    else:
                        syn = h.ReLU_Syn_avg(self.cell_soma_sec(cell)(0.5))
                    self.add_mech_to_synlist(syn.get_segment(), syn)
                    self.synarray_forward[j, i] = syn

                    syn.g = 1 / self.cell_rsoma
                    syn.learning_rate = 1.

                    #self.pc.target_var(syn, syn._ref_v_in, sid_v_in)
                    self.add_target_var(syn.get_segment(), syn, "v_in", sid_v_in)
                
                # input bias
                syn = h.Linear_Syn_avg(self.cell_soma_sec(cell)(0.5))
                self.add_mech_to_synlist(syn.get_segment(), syn)
                self.synarray_bias[i] = syn

                syn.g = 1 / self.cell_rsoma
                syn.learning_rate = 1.

                syn.v_in = 1.0

        # output sids
        self.sids_out = self.idm.alloc_sid(self.N_out)
        for gid_cell, sid_out in zip(self.gids_cell, self.sids_out):
            if self.pc.gid_exists(gid_cell):
                cell = self.pc.gid2cell(gid_cell)
                sec = self.cell_soma_sec(cell)
                #self.pc.source_var(sec(0.5)._ref_v, sid_out, sec=sec)
                self.idm.sid2param[sid_out] = (sec(0.5), None, "v")
        if self.nhost > 1:
            self.pc.barrier()

    def setup_backward(self, requires_input_grad=True):
        self.requires_input_grad = requires_input_grad

        # inter for computing sum of exp outputs
        self.inter = []
        self.gid_inter = self.idm.alloc_gid(1)[0]
        self.synlist_out2inter = []
        self.sids_grad_out = self.idm.alloc_sid(self.N_out)
        if self.ihost == 0:
            self.pc.set_gid2node(self.gid_inter, self.ihost)
            inter = self.inter_instance()
            self.inter.append(inter)
            sec = self.inter_soma_sec(inter)
            nc = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
            self.pc.cell(self.gid_inter, nc)

            # softmax(y) in output-to-inter synpases
            syn = h.Softmax_Syn_avg(sec(0.5))
            self.add_mech_to_synlist(syn.get_segment(), syn)
            self.synlist_out2inter.append(syn)
            for i, (sid_out, sid_grad_out) in enumerate(zip(self.sids_out, self.sids_grad_out)):
                #self.pc.target_var(syn, syn._ref_v_out, sid_out)
                self.add_target_var(syn.get_segment(), syn, f"v_out_{i}", sid_out)
                # inter v calculates sum of exps, each synapse calculates corresponding softmax
                #self.pc.source_var(syn._ref_grad_out, sid_grad_out, sec=sec)
                self.idm.sid2param[sid_grad_out] = (syn.get_segment(), syn, f"grad_out_{i}")
        if self.nhost > 1:
            self.pc.barrier()

        # dW = dy * f(x), db = dy
        for i, (gid_cell, sid_grad_out) in enumerate(zip(self.gids_cell, self.sids_grad_out)):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    syn = self.synarray_forward[j, i]
                    #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                    self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)
                syn = self.synarray_bias[i]
                #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)

        if self.requires_input_grad:
            # inters for computing input grads
            self.grads = []
            self.gids_grad = self.idm.alloc_gid(self.N_in)
            for i, gid_grad in enumerate(self.gids_grad):
                if (i % self.nhost) != self.ihost:
                    continue
                self.pc.set_gid2node(gid_grad, self.ihost)
                grad = self.grad_instance()
                self.grads.append(grad)
                sec = self.grad_soma_sec(grad)
                nc = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
                self.pc.cell(gid_grad, nc)

            # transfer resistance
            tmpgrad = self.grad_instance()
            impd = h.Impedance()
            impd.loc(0.5, sec=self.grad_soma_sec(tmpgrad))
            impd.compute(0)
            self.grad_rsoma = impd.transfer(0.5, sec=self.grad_soma_sec(tmpgrad))
            del tmpgrad, impd

            # dx = dy * W' * df
            self.synarray_backward = np.full((self.N_out, self.N_in), fill_value=None)
            for i, (gid_grad, sid_v_in) in enumerate(zip(self.gids_grad, self.sids_in)):
                if self.pc.gid_exists(gid_grad):
                    grad = self.pc.gid2cell(gid_grad)
                    sec = self.grad_soma_sec(grad)
                    for j, sid_grad_out in enumerate(self.sids_grad_out):
                        if self.in_activation == 'Linear':
                            syn = h.Linear_Grad_Syn_avg(sec(0.5))
                            self.add_mech_to_synlist(syn.get_segment(), syn)
                        else:
                            syn = h.ReLU_Grad_Syn_avg(sec(0.5))
                            self.add_mech_to_synlist(syn.get_segment(), syn)
                            #self.pc.target_var(syn, syn._ref_v_in, sid_v_in)
                            self.add_target_var(syn.get_segment(), syn, "v_in", sid_v_in)
                        self.synarray_backward[j, i] = syn

                        syn.g = 1 / self.grad_rsoma

                        #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                        self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)

            # grad sids
            self.sids_grad_in = self.idm.alloc_sid(self.N_in)
            for gid_grad, sid_grad_in in zip(self.gids_grad, self.sids_grad_in):
                if self.pc.gid_exists(gid_grad):
                    grad = self.pc.gid2cell(gid_grad)
                    sec = self.grad_soma_sec(grad)
                    #self.pc.source_var(sec(0.5)._ref_v, sid_grad_in, sec=sec)
                    self.idm.sid2param[sid_grad_in] = (sec(0.5), None, "v")
            if self.nhost > 1:
                self.pc.barrier()

    def set_target(self, target):
        assert target < self.N_out
        if self.ihost == 0:
            syn = self.synlist_out2inter[0]
            for i in range(self.N_out):
                if i == target:
                    setattr(syn, f'target_{i}', 1)
                else:
                    setattr(syn, f'target_{i}', 0)

    def is_train(self, start=20, end=50, lr_rate=0.005):
        self.mode = "Train"
        self.lr_start, self.lr_end, self.lr_dur, self.lr_rate = start, end, end - start, lr_rate
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    syn = self.synarray_forward[j, i]
                    syn.lr_start, syn.lr_end = self.lr_start, self.lr_end
                    syn.learning_rate *= self.lr_rate
                syn = self.synarray_bias[i]
                syn.lr_start, syn.lr_end = self.lr_start, self.lr_end
                syn.learning_rate *= self.lr_rate
        for syn in self.synlist_out2inter:
            syn.lr_start, syn.lr_end = self.lr_start, self.lr_end

    def is_test(self):
        self.mode = "Test"
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    syn = self.synarray_forward[j, i]
                    syn.lr_start, syn.lr_end = 1e9, 1e9
                syn = self.synarray_bias[i]
                syn.lr_start, syn.lr_end = 1e9, 1e9

    def save_weights(self):
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    self.weight[j, i] += self.synarray_forward[j, i].delta_w / (self.lr_dur / h.dt)
                self.bias[i] += self.synarray_bias[i].delta_w / (self.lr_dur / h.dt)

    def save_delta_weights(self):
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    self.delta_weight[j, i] = self.synarray_forward[j, i].delta_w / (self.lr_dur / h.dt)
                self.delta_bias[i] = self.synarray_bias[i].delta_w / (self.lr_dur / h.dt)

    def load_weights(self):
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    self.synarray_forward[j, i].w = self.weight[j, i]
                self.synarray_bias[i].w = self.bias[i]
        if self.requires_input_grad:
            for i, gid_grad in enumerate(self.gids_grad):
                if self.pc.gid_exists(gid_grad):
                    for j in range(self.N_out):
                        self.synarray_backward[j, i].w = self.weight[i, j]

    # def add_weights(self, seg2synlist, out_lines):
    #     for i, gid_cell in enumerate(self.gids_cell):
    #         if self.pc.gid_exists(gid_cell):
    #             for j in range(self.N_in):
    #                 syn = self.synarray_forward[j, i]
    #                 sec, x, mname, mid = self.get_param_info(syn)
    #                 src_str = "0 %s %.6f %s %d w\n"%(sec, x, mname, mid)
    #                 out_lines.append(src_str)

    def flatten_weight_index(self, idxs):
        return idxs[0] * self.N_out + idxs[1]
    
    def flatten_bias_index(self, idxs):
        return idxs[0]

    def add_weight_param(self, seg2synlist, index_start, w_out_lines, dw_out_lines):
        self.nweight = self.weight.size
        self.nbias = self.bias.size
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    syn = self.synarray_forward[j, i]
                    sec, x, mname, mid = self.get_param_info(syn)
                    idx = index_start + self.flatten_weight_index([j, i,])
                    w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                    w_out_lines.append(w_str)
                    dw_str = "0 %s %.6f %s %d delta_w %d\n"%(sec, x, mname, mid, idx)
                    dw_out_lines.append(dw_str)
                
                syn = self.synarray_bias[i]
                sec, x, mname, mid = self.get_param_info(syn)
                idx = index_start + self.nweight + self.flatten_bias_index([i,])
                w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                w_out_lines.append(w_str)
                dw_str = "0 %s %.6f %s %d delta_w %d\n"%(sec, x, mname, mid, idx)
                dw_out_lines.append(dw_str)

        if self.requires_input_grad:
            for i, gid_grad in enumerate(self.gids_grad):
                if self.pc.gid_exists(gid_grad):
                    for j in range(self.N_out):
                        syn = self.synarray_backward[j, i]
                        sec, x, mname, mid = self.get_param_info(syn)
                        idx = index_start + self.flatten_weight_index([i, j,])
                        w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                        w_out_lines.append(w_str)


class FullyConnectedHPC(BaseLayer):
    def __init__(self, N_in, N_out, N_multi, sids_in, in_activation='ReLU', **kwargs):
        super(FullyConnectedHPC, self).__init__(**kwargs)
        self.N_in = N_in
        self.N_out = N_out
        self.N_multi = N_multi
        assert len(sids_in) == N_in
        self.sids_in = sids_in
        if in_activation not in ['Linear', 'ReLU']:
            raise ValueError("Undefined input activation, must be one of 'Linear' or 'ReLU'")
        self.in_activation = in_activation

        self._create_cells()
        self._connect_cells()

    def cell_instance(self):
        cell = setup_hpc("PassiveHPC", "2013_03_06_cell11_1125_H41_06.asc")
        for sec in cell.all:
            sec.cm = 1.5
            # sec.g_pas = 1e-3
        return cell

    def cell_soma_sec(self, cell):
        return cell.soma[0]

    def grad_instance(self):
        grad = h.PointNeuron()
        for sec in grad.all:
            sec.Ra = 100
            sec.cm = 1
            sec.g_pas = 1
        return grad

    def grad_soma_sec(self, grad):
        return grad.soma

    def _create_cells(self):
        h.load_file("import3d.hoc")
        h.load_file("PassiveHPC.hoc")
        self.cells = []
        self.gids_cell = self.idm.alloc_gid(self.N_out)
        for i, gid_cell in enumerate(self.gids_cell):
            if i % 50 == 0:
                print(i, gid_cell)
            if (i % self.nhost) != self.ihost:
                continue
            self.pc.set_gid2node(gid_cell, self.ihost)
            cell = self.cell_instance()
            self.cells.append(cell)
            sec = self.cell_soma_sec(cell)
            nc = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
            self.pc.cell(gid_cell, nc)

    def _connect_cells(self):
        tmpcell = self.cell_instance()
        self.total_dend = len(tmpcell.dend)

        # connection matrix input-to-out dendrites, each input connects N_multi (1 default) sites for each pyramidal
        # self.conn_dend = self.rng.integers(0, self.total_dend, (self.N_in, self.N_out, self.N_multi))
        # self.conn_loc = self.rng.random((self.N_in, self.N_out, self.N_multi))
        self.conn_dend = np.zeros((self.N_in, self.N_out, self.N_multi), dtype=int)
        self.conn_loc = np.zeros((self.N_in, self.N_out, self.N_multi))
        for i in range(self.N_in):
            for j in range(self.N_out):
                for k in range(self.N_multi):
                    dist = 1e9  # proximal
                    # dist = 0    # distal
                    while dist > 50:    # proximal
                    # while dist < 200:   # distal
                        dend_id = self.rng.integers(0, self.total_dend)
                        loc = self.rng.random()
                        dist = h.distance(self.cell_soma_sec(tmpcell)(0.5), tmpcell.dend[dend_id](loc))
                    self.conn_dend[i, j, k] = dend_id
                    self.conn_loc[i, j, k] = loc

        # weight matrix input-to-output dendrites
        limit = np.sqrt(6 / (self.N_in * self.N_multi))
        self.weight = self.rng.uniform(-limit, limit, (self.N_in, self.N_out, self.N_multi))
        self.bias = np.zeros((self.N_out,))
        self.delta_weight = np.zeros((self.N_in, self.N_out, self.N_multi))
        self.delta_bias = np.zeros((self.N_out,))

        # transfer resistance matrix
        impd = h.Impedance()
        impd.loc(0.5, sec=self.cell_soma_sec(tmpcell))
        impd.compute(0)
        self.transr = np.ones((self.N_in, self.N_out, self.N_multi))
        for i in range(self.N_in):
            for j in range(self.N_out):
                for k in range(self.N_multi):
                    dend_id = self.conn_dend[i, j, k]
                    loc = self.conn_loc[i, j, k]
                    self.transr[i, j, k] = impd.transfer(loc, sec=tmpcell.dend[dend_id])
        self.cell_rsoma = impd.transfer(0.5, sec=self.cell_soma_sec(tmpcell))
        self.cell_rmean = np.sqrt(np.max(self.transr) * np.min(self.transr))
        del tmpcell, impd

        # y = W * f(x) + b
        self.synarray_forward = np.full((self.N_in, self.N_out, self.N_multi), fill_value=None)
        self.synarray_bias = np.full((self.N_out,), fill_value=None)
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                cell = self.pc.gid2cell(gid_cell)
                for j, sid_v_in in enumerate(self.sids_in):
                    for k in range(self.N_multi):
                        dend_id = self.conn_dend[j, i, k]
                        loc = self.conn_loc[j, i, k]
                        if self.in_activation == 'Linear':
                            syn = h.Linear_Syn_avg(cell.dend[dend_id](loc))
                        else:
                            syn = h.ReLU_Syn_avg(cell.dend[dend_id](loc))
                        self.add_mech_to_synlist(syn.get_segment(), syn)
                        self.synarray_forward[j, i, k] = syn

                        r = self.transr[j, i, k]
                        syn.g = 1 / r
                        syn.learning_rate = 1.

                        #self.pc.target_var(syn, syn._ref_v_in, sid_v_in)
                        self.add_target_var(syn.get_segment(), syn, "v_in", sid_v_in)

                # input bias
                syn = h.Linear_Syn_avg(self.cell_soma_sec(cell)(0.5))
                self.add_mech_to_synlist(syn.get_segment(), syn)
                self.synarray_bias[i] = syn

                syn.g = 1 / self.cell_rsoma
                syn.learning_rate = 1.

                syn.v_in = 1.0

        # output sids
        self.sids_out = self.idm.alloc_sid(self.N_out)
        for gid_cell, sid_out in zip(self.gids_cell, self.sids_out):
            if self.pc.gid_exists(gid_cell):
                cell = self.pc.gid2cell(gid_cell)
                sec = self.cell_soma_sec(cell)
                #self.pc.source_var(sec(0.5)._ref_v, sid_out, sec=sec)
                self.idm.sid2param[sid_out] = (sec(0.5), None, "v")
        if self.nhost > 1:
            self.pc.barrier()

    def setup_backward(self, sids_grad_out, requires_input_grad=True):
        assert len(sids_grad_out) == self.N_out
        self.sids_grad_out = sids_grad_out

        self.requires_input_grad = requires_input_grad

        # dW = dy * f(x), db = dy
        for i, (gid_cell, sid_grad_out) in enumerate(zip(self.gids_cell, self.sids_grad_out)):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    for k in range(self.N_multi):
                        syn = self.synarray_forward[j, i, k]
                        #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                        self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)
                syn = self.synarray_bias[i]
                #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)

        if self.requires_input_grad:
            # inters for computing input grads
            h.load_file("PointNeuron.hoc")
            self.grads = []
            self.gids_grad = self.idm.alloc_gid(self.N_in)
            for i, gid_grad in enumerate(self.gids_grad):
                if (i % self.nhost) != self.ihost:
                    continue
                self.pc.set_gid2node(gid_grad, self.ihost)
                grad = self.grad_instance()
                self.grads.append(grad)
                sec = self.grad_soma_sec(grad)
                nc = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
                self.pc.cell(gid_grad, nc)

            # transfer resistance
            tmpgrad = self.grad_instance()
            impd = h.Impedance()
            impd.loc(0.5, sec=self.grad_soma_sec(tmpgrad))
            impd.compute(0)
            self.grad_rsoma = impd.transfer(0.5, sec=self.grad_soma_sec(tmpgrad))
            del tmpgrad, impd

            # dx = dy * W' * df
            self.synarray_backward = np.full((self.N_out, self.N_in, self.N_multi), fill_value=None)
            for i, (gid_grad, sid_v_in) in enumerate(zip(self.gids_grad, self.sids_in)):
                if self.pc.gid_exists(gid_grad):
                    grad = self.pc.gid2cell(gid_grad)
                    sec = self.grad_soma_sec(grad)
                    for j, sid_grad_out in enumerate(self.sids_grad_out):
                        for k in range(self.N_multi):
                            if self.in_activation == 'Linear':
                                syn = h.Linear_Grad_Syn_avg(sec(0.5))
                                self.add_mech_to_synlist(syn.get_segment(), syn)
                            else:
                                syn = h.ReLU_Grad_Syn_avg(sec(0.5))
                                self.add_mech_to_synlist(syn.get_segment(), syn)
                                self.add_target_var(syn.get_segment(), syn, "v_in", sid_v_in)
                                #self.pc.target_var(syn, syn._ref_v_in, sid_v_in)
                            self.synarray_backward[j, i, k] = syn

                            syn.g = 1 / self.grad_rsoma

                            #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                            self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)

            # grad sids
            self.sids_grad_in = self.idm.alloc_sid(self.N_in)
            for gid_grad, sid_grad_in in zip(self.gids_grad, self.sids_grad_in):
                if self.pc.gid_exists(gid_grad):
                    grad = self.pc.gid2cell(gid_grad)
                    sec = self.grad_soma_sec(grad)
                    #self.pc.source_var(sec(0.5)._ref_v, sid_grad_in, sec=sec)
                    self.idm.sid2param[sid_grad_in] = (sec(0.5), None, "v")
            if self.nhost > 1:
                self.pc.barrier()

    def is_train(self, start=20, end=50, lr_rate=0.005):
        self.mode = "Train"
        self.lr_start, self.lr_end, self.lr_dur, self.lr_rate = start, end, end - start, lr_rate
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    for k in range(self.N_multi):
                        syn = self.synarray_forward[j, i, k]
                        syn.lr_start, syn.lr_end = self.lr_start, self.lr_end
                        syn.learning_rate *= self.lr_rate
                syn = self.synarray_bias[i]
                syn.lr_start, syn.lr_end = self.lr_start, self.lr_end
                syn.learning_rate *= self.lr_rate

    def is_test(self):
        self.mode = "Test"
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    for k in range(self.N_multi):
                        syn = self.synarray_forward[j, i, k]
                        syn.lr_start, syn.lr_end = 1e9, 1e9
                syn = self.synarray_bias[i]
                syn.lr_start, syn.lr_end = 1e9, 1e9

    def save_weights(self):
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    for k in range(self.N_multi):
                        self.weight[j, i, k] += self.synarray_forward[j, i, k].delta_w / (self.lr_dur / h.dt)
                self.bias[i] += self.synarray_bias[i].delta_w / (self.lr_dur / h.dt)

    def save_delta_weights(self):
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    for k in range(self.N_multi):
                        self.delta_weight[j, i, k] = self.synarray_forward[j, i, k].delta_w / (self.lr_dur / h.dt)
                self.delta_bias[i] = self.synarray_bias[i].delta_w / (self.lr_dur / h.dt)

    def load_weights(self):
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    for k in range(self.N_multi):
                        self.synarray_forward[j, i, k].w = self.weight[j, i, k]
                self.synarray_bias[i].w = self.bias[i]
        if self.requires_input_grad:
            for i, gid_grad in enumerate(self.gids_grad):
                if self.pc.gid_exists(gid_grad):
                    for j in range(self.N_out):
                        for k in range(self.N_multi):
                            self.synarray_backward[j, i, k].w = self.weight[i, j, k] # self.synarray_forward[i,j].w += self.synarray_forward[i,j].delta_w

    # def add_weights(self, seg2synlist, out_lines):
    #     for i, gid_cell in enumerate(self.gids_cell):
    #         if self.pc.gid_exists(gid_cell):
    #             for j in range(self.N_in):
    #                 for k in range(self.N_multi):
    #                     syn = self.synarray_forward[j, i, k]
    #                     sec, x, mname, mid = self.get_param_info(syn)
    #                     src_str = "0 %s %.6f %s %d w\n"%(sec, x, mname, mid)
    #                     out_lines.append(src_str)
    #         syn = self.synarray_bias[i]
    #         sec, x, mname, mid = self.get_param_info(syn)
    #         src_str = "0 %s %.6f %s %d w\n"%(sec, x, mname, mid)
    #         out_lines.append(src_str)

    def flatten_weight_index(self, idxs):
        return idxs[0] * self.N_out * self.N_multi + idxs[1] * self.N_multi + idxs[2]

    def flatten_bias_index(self, idxs):
        return idxs[0]

    def add_weight_param(self, seg2synlist, index_start, w_out_lines, dw_out_lines):
        self.nweight = self.weight.size
        self.nbias = self.bias.size
        for i, gid_cell in enumerate(self.gids_cell):
            if self.pc.gid_exists(gid_cell):
                for j in range(self.N_in):
                    for k in range(self.N_multi):
                        syn = self.synarray_forward[j, i, k]
                        sec, x, mname, mid = self.get_param_info(syn)
                        idx = index_start + self.flatten_weight_index([j, i, k,])
                        w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                        w_out_lines.append(w_str)
                        dw_str = "0 %s %.6f %s %d delta_w %d\n"%(sec, x, mname, mid, idx)
                        dw_out_lines.append(dw_str)

                syn = self.synarray_bias[i]
                sec, x, mname, mid = self.get_param_info(syn)
                idx = index_start + self.nweight + self.flatten_bias_index([i,])
                w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                w_out_lines.append(w_str)
                dw_str = "0 %s %.6f %s %d delta_w %d\n"%(sec, x, mname, mid, idx)
                dw_out_lines.append(dw_str)

        if self.requires_input_grad:
            for i, gid_grad in enumerate(self.gids_grad):
                if self.pc.gid_exists(gid_grad):
                    for j in range(self.N_out):
                        for k in range(self.N_multi):
                            syn = self.synarray_backward[j, i, k]
                            sec, x, mname, mid = self.get_param_info(syn)
                            idx = index_start + self.flatten_weight_index([i, j, k,])
                            w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                            w_out_lines.append(w_str)


class Conv2DPoint(BaseLayer):
    def __init__(self, in_channels, out_channels, sids_in, kernel_size, stride, padding,
                 weight_share=True, in_activation='ReLU', **kwargs):
        super(Conv2DPoint, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.weight_share = weight_share
        self.sids_in = sids_in  # (in_channels, in_H, in_W)
        self.in_shape = self.sids_in.shape
        assert self.in_shape[0] == self.in_channels
        self.in_size = self.sids_in.size
        self._cal_out_shape()   # (out_channels, out_H, out_W)
        self.out_size = np.prod(self.out_shape)
        self._cal_conn_sids()   # {(out_i, out_j) -> [(padded_in_i, padded_in_j), ...]}
        if in_activation not in ['Linear', 'ReLU']:
            raise ValueError("Undefined input activation, must be one of 'Linear' or 'ReLU'")
        self.in_activation = in_activation

        self._create_cells()
        self._connect_cells()

    def _cal_out_shape(self):
        in_H, in_W = self.in_shape[1], self.in_shape[2]
        out_H = int(np.floor((in_H + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        out_W = int(np.floor((in_W + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.out_shape = (self.out_channels, out_H, out_W)

    def _cal_conn_sids(self):
        self.conn_out2in = {}
        for i in range(self.out_shape[1]):
            for j in range(self.out_shape[2]):
                self.conn_out2in[i, j] = [(iw, jw)
                                          for iw in range(i * self.stride[0], i * self.stride[0] + self.kernel_size[0])
                                          for jw in range(j * self.stride[1], j * self.stride[1] + self.kernel_size[1])]

    def cell_instance(self):
        cell = h.PointNeuron()
        for sec in cell.all:
            sec.Ra = 100
            sec.cm = 1
            sec.g_pas = 2e-5
        return cell

    def cell_soma_sec(self, cell):
        return cell.soma

    def grad_instance(self):
        cell = h.PointNeuron()
        for sec in cell.all:
            sec.Ra = 100
            sec.cm = 1
            sec.g_pas = 1
        return cell

    def grad_soma_sec(self, grad):
        return grad.soma

    def _create_cells(self):
        h.load_file("import3d.hoc")
        h.load_file("PointNeuron.hoc")
        self.cells = []
        self.gids_cell = np.reshape(self.idm.alloc_gid(self.out_size), self.out_shape)
        for c in range(self.out_shape[0]):
            if (c % self.nhost) != self.ihost:
                continue
            for i in range(self.out_shape[1]):
                for j in range(self.out_shape[2]):
                    gid = self.gids_cell[c, i, j]
                    self.pc.set_gid2node(gid, self.ihost)
                    cell = self.cell_instance()
                    self.cells.append(cell)
                    sec = self.cell_soma_sec(cell)
                    nc = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
                    self.pc.cell(gid, nc)

    def _connect_cells(self):
        # weight matrix input-to-out dendrites
        limit = np.sqrt(6 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        if self.weight_share:
            self.weight = self.rng.uniform(-limit, limit, (self.kernel_size[0] * self.kernel_size[1], self.in_channels, self.out_channels))
            self.delta_weight = np.zeros((self.kernel_size[0] * self.kernel_size[1], self.in_channels, self.out_channels))
            self.bias = np.zeros((self.out_channels,))
            self.delta_bias = np.zeros((self.out_channels,))
        else:
            self.weight = {}
            self.delta_weight = {}
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for padded_in_idx in padded_in_idxs:
                    ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                    if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                        self.weight[out_idx, (ii, ji)] = self.rng.uniform(-limit, limit, (self.in_channels, self.out_channels))
                        self.delta_weight[out_idx, (ii, ji)] = np.zeros((self.in_channels, self.out_channels))
            self.bias = np.zeros(self.out_shape)
            self.delta_bias = np.zeros(self.out_shape)

        # transfer resistance
        tmpcell = self.cell_instance()
        impd = h.Impedance()
        impd.loc(0.5, sec=self.cell_soma_sec(tmpcell))
        impd.compute(0)
        self.cell_rsoma = impd.transfer(0.5, sec=self.cell_soma_sec(tmpcell))
        del tmpcell, impd

        # y = conv(W, f(x)) + b
        self.synarray_forward = {}
        self.synarray_bias = np.full(self.out_shape, fill_value=None)
        for out_idx, padded_in_idxs in self.conn_out2in.items():
            for co in range(self.out_channels):
                gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                if self.pc.gid_exists(gid_cell):
                    cell = self.pc.gid2cell(gid_cell)
                    for padded_in_idx in padded_in_idxs:
                        ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                        if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                            if (out_idx, (ii, ji)) not in self.synarray_forward.keys():
                                self.synarray_forward[out_idx, (ii, ji)] = np.full((self.in_channels, self.out_channels), fill_value=None)
                            for ci in range(self.in_channels):
                                sid_v_in = self.sids_in[ci, ii, ji]
                                if self.in_activation == 'Linear':
                                    syn = h.Linear_Syn_avg(self.cell_soma_sec(cell)(0.5))
                                else:
                                    syn = h.ReLU_Syn_avg(self.cell_soma_sec(cell)(0.5))
                                self.add_mech_to_synlist(syn.get_segment(), syn)
                                self.synarray_forward[out_idx, (ii, ji)][ci, co] = syn

                                syn.g = 1 / self.cell_rsoma
                                syn.learning_rate = 1.

                                #self.pc.target_var(syn, syn._ref_v_in, sid_v_in)
                                self.add_target_var(syn.get_segment(), syn, "v_in", sid_v_in)

                    # input bias
                    syn = h.Linear_Syn_avg(self.cell_soma_sec(cell)(0.5))
                    self.add_mech_to_synlist(syn.get_segment(), syn)
                    self.synarray_bias[co, out_idx[0], out_idx[1]] = syn

                    syn.g = 1 / self.cell_rsoma
                    syn.learning_rate = 1.

                    syn.v_in = 1.0

        # output sids
        self.sids_out = np.reshape(self.idm.alloc_sid(self.out_size), self.out_shape)
        for gid_cell, sid_out in zip(self.gids_cell.flatten(), self.sids_out.flatten()):
            if self.pc.gid_exists(gid_cell):
                cell = self.pc.gid2cell(gid_cell)
                sec = self.cell_soma_sec(cell)
                #self.pc.source_var(sec(0.5)._ref_v, sid_out, sec=sec)
                self.idm.sid2param[sid_out] = (sec(0.5), None, "v")
        if self.nhost > 1:
            self.pc.barrier()

    def setup_backward(self, sids_grad_out, requires_input_grad=True):
        assert sids_grad_out.shape == self.out_shape
        self.sids_grad_out = sids_grad_out

        self.requires_input_grad = requires_input_grad

        # dW = dy * f(x), db = dy
        for out_idx, padded_in_idxs in self.conn_out2in.items():
            for co in range(self.out_channels):
                gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                sid_grad_out = self.sids_grad_out[co, out_idx[0], out_idx[1]]
                if self.pc.gid_exists(gid_cell):
                    for padded_in_idx in padded_in_idxs:
                        ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                        if (ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]):
                            for ci in range(self.in_channels):
                                syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                                self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)
                    syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                    #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                    self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)

        if self.requires_input_grad:
            # inters for computing input grads
            h.load_file("PointNeuron.hoc")
            self.grads = []
            self.gids_grad = np.reshape(self.idm.alloc_gid(self.in_size), self.in_shape)
            for ci in range(self.in_shape[0]):
                if (ci % self.nhost) != self.ihost:
                    continue
                for ii in range(self.in_shape[1]):
                    for ji in range(self.in_shape[2]):
                        gid_grad = self.gids_grad[ci, ii, ji]
                        self.pc.set_gid2node(gid_grad, self.ihost)
                        grad = self.grad_instance()
                        self.grads.append(grad)
                        sec = self.grad_soma_sec(grad)
                        nc = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
                        self.pc.cell(gid_grad, nc)

            # transfer resistance
            tmpgrad = self.grad_instance()
            impd = h.Impedance()
            impd.loc(0.5, sec=self.grad_soma_sec(tmpgrad))
            impd.compute(0)
            self.grad_rsoma = impd.transfer(0.5, sec=self.grad_soma_sec(tmpgrad))
            del tmpgrad, impd

            # dx = dy * W' * df
            self.synarray_backward = {}
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for ci in range(self.in_channels):
                    for padded_in_idx in padded_in_idxs:
                        ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                        if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                            if (out_idx, (ii, ji)) not in self.synarray_backward.keys():
                                self.synarray_backward[out_idx, (ii, ji)] = np.full((self.in_channels, self.out_channels), fill_value=None)
                            sid_v_in = self.sids_in[ci, ii, ji]
                            gid_grad = self.gids_grad[ci, ii, ji]
                            if self.pc.gid_exists(gid_grad):
                                grad = self.pc.gid2cell(gid_grad)
                                sec = self.grad_soma_sec(grad)
                                for co in range(self.out_channels):
                                    sid_grad_out = self.sids_grad_out[co, out_idx[0], out_idx[1]]
                                    if self.in_activation == 'Linear':
                                        syn = h.Linear_Grad_Syn_avg(sec(0.5))
                                        self.add_mech_to_synlist(syn.get_segment(), syn)
                                    else:
                                        syn = h.ReLU_Grad_Syn_avg(sec(0.5))
                                        self.add_mech_to_synlist(syn.get_segment(), syn)
                                        #self.pc.target_var(syn, syn._ref_v_in, sid_v_in)
                                        self.add_target_var(syn.get_segment(), syn, "v_in", sid_v_in)
                                    self.synarray_backward[out_idx, (ii, ji)][ci, co] = syn

                                    syn.g = 1 / self.grad_rsoma

                                    #self.pc.target_var(syn, syn._ref_grad_out, sid_grad_out)
                                    self.add_target_var(syn.get_segment(), syn, "grad_out", sid_grad_out)

            # grad sids
            self.sids_grad_in = np.reshape(self.idm.alloc_sid(self.in_size), self.in_shape)
            for gid_grad, sid_grad_in in zip(self.gids_grad.flatten(), self.sids_grad_in.flatten()):
                if self.pc.gid_exists(gid_grad):
                    grad = self.pc.gid2cell(gid_grad)
                    sec = self.grad_soma_sec(grad)
                    #self.pc.source_var(sec(0.5)._ref_v, sid_grad_in, sec=sec)
                    self.idm.sid2param[sid_grad_in] = (sec(0.5), None, "v")
            if self.nhost > 1:
                self.pc.barrier()

    def is_train(self, start=20, end=50, lr_rate=0.005):
        self.mode = "Train"
        self.lr_start, self.lr_end, self.lr_dur, self.lr_rate = start, end, end - start, lr_rate
        for out_idx, padded_in_idxs in self.conn_out2in.items():
            for co in range(self.out_channels):
                gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                if self.pc.gid_exists(gid_cell):
                    for padded_in_idx in padded_in_idxs:
                        ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                        if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                            for ci in range(self.in_channels):
                                syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                syn.lr_start, syn.lr_end = self.lr_start, self.lr_end
                                syn.learning_rate *= self.lr_rate
                    syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                    syn.lr_start, syn.lr_end = self.lr_start, self.lr_end
                    syn.learning_rate *= self.lr_rate

    def is_test(self):
        self.mode = "Test"
        for out_idx, padded_in_idxs in self.conn_out2in.items():
            for co in range(self.out_channels):
                gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                if self.pc.gid_exists(gid_cell):
                    for padded_in_idx in padded_in_idxs:
                        ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                        if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                            for ci in range(self.in_channels):
                                syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                syn.lr_start, syn.lr_end = 1e9, 1e9
                    syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                    syn.lr_start, syn.lr_end = 1e9, 1e9

    def save_weights(self):
        if self.weight_share:
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for co in range(self.out_channels):
                    gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                    if self.pc.gid_exists(gid_cell):
                        for iker, padded_in_idx in enumerate(padded_in_idxs):
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                for ci in range(self.in_channels):
                                    syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                    self.weight[iker, ci, co] += syn.delta_w / (self.lr_dur / h.dt)
                        syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                        self.bias[co] += syn.delta_w / (self.lr_dur / h.dt)
        else:
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for co in range(self.out_channels):
                    gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                    if self.pc.gid_exists(gid_cell):
                        for padded_in_idx in padded_in_idxs:
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                for ci in range(self.in_channels):
                                    syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                    self.weight[out_idx, (ii, ji)][ci, co] += syn.delta_w / (self.lr_dur / h.dt)
                        syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                        self.bias[co, out_idx[0], out_idx[1]] += syn.delta_w / (self.lr_dur / h.dt)

    def save_delta_weights(self):
        if self.weight_share:
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for co in range(self.out_channels):
                    gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                    if self.pc.gid_exists(gid_cell):
                        for iker, padded_in_idx in enumerate(padded_in_idxs):
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                for ci in range(self.in_channels):
                                    syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                    self.delta_weight[iker, ci, co] += syn.delta_w / (self.lr_dur / h.dt)
                        syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                        self.delta_bias[co] += syn.delta_w / (self.lr_dur / h.dt)
        else:
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for co in range(self.out_channels):
                    gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                    if self.pc.gid_exists(gid_cell):
                        for padded_in_idx in padded_in_idxs:
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                for ci in range(self.in_channels):
                                    syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                    self.delta_weight[out_idx, (ii, ji)][ci, co] = syn.delta_w / (self.lr_dur / h.dt)
                        syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                        self.delta_bias[co, out_idx[0], out_idx[1]] = syn.delta_w / (self.lr_dur / h.dt)

    def load_weights(self):
        if self.weight_share:
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for co in range(self.out_channels):
                    gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                    if self.pc.gid_exists(gid_cell):
                        for iker, padded_in_idx in enumerate(padded_in_idxs):
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                for ci in range(self.in_channels):
                                    self.synarray_forward[out_idx, (ii, ji)][ci, co].w = self.weight[iker, ci, co]
                        self.synarray_bias[co, out_idx[0], out_idx[1]].w = self.bias[co]
            if self.requires_input_grad:
                for out_idx, padded_in_idxs in self.conn_out2in.items():
                    for ci in range(self.in_channels):
                        for iker, padded_in_idx in enumerate(padded_in_idxs):
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                gid_grad = self.gids_grad[ci, ii, ji]
                                if self.pc.gid_exists(gid_grad):
                                    for co in range(self.out_channels):
                                        self.synarray_backward[out_idx, (ii, ji)][ci, co].w = self.weight[iker, ci, co]
        else:
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for co in range(self.out_channels):
                    gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                    if self.pc.gid_exists(gid_cell):
                        for padded_in_idx in padded_in_idxs:
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                for ci in range(self.in_channels):
                                    self.synarray_forward[out_idx, (ii, ji)][ci, co].w = self.weight[out_idx, (ii, ji)][ci, co]
                        self.synarray_bias[co, out_idx[0], out_idx[1]].w = self.bias[co, out_idx[0], out_idx[1]]
            if self.requires_input_grad:
                for out_idx, padded_in_idxs in self.conn_out2in.items():
                    for ci in range(self.in_channels):
                        for padded_in_idx in padded_in_idxs:
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                gid_grad = self.gids_grad[ci, ii, ji]
                                if self.pc.gid_exists(gid_grad):
                                    for co in range(self.out_channels):
                                        self.synarray_backward[out_idx, (ii, ji)][ci, co].w = self.weight[out_idx, (ii, ji)][ci, co]

    # def add_weights(self, seg2synlist, out_lines):
    #     if self.weight_share:
    #         for out_idx, padded_in_idxs in self.conn_out2in.items():
    #             for co in range(self.out_channels):
    #                 gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
    #                 if self.pc.gid_exists(gid_cell):
    #                     for iker, padded_in_idx in enumerate(padded_in_idxs):
    #                         ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
    #                         if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
    #                             for ci in range(self.in_channels):
    #                                 #self.synarray_forward[out_idx, (ii, ji)][ci, co].w = self.weight[iker, ci, co]
    #                                 syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
    #                                 sec, x, mname, mid = self.get_param_info(syn)
    #                                 src_str = "0 %s %.6f %s %d w\n"%(sec, x, mname, mid)
    #                                 out_lines.append(src_str)

    #                     syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
    #                     sec, x, mname, mid = self.get_param_info(syn)
    #                     src_str = "0 %s %.6f %s %d w\n"%(sec, x, mname, mid)
    #                     out_lines.append(src_str)

    #     else:
    #         for out_idx, padded_in_idxs in self.conn_out2in.items():
    #             for co in range(self.out_channels):
    #                 gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
    #                 if self.pc.gid_exists(gid_cell):
    #                     for padded_in_idx in padded_in_idxs:
    #                         ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
    #                         if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
    #                             for ci in range(self.in_channels):
    #                                 #self.synarray_forward[out_idx, (ii, ji)][ci, co].w = self.weight[out_idx, (ii, ji)][ci, co]
    #                                 syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
    #                                 sec, x, mname, mid = self.get_param_info(syn)
    #                                 src_str = "0 %s %.6f %s %d w\n"%(sec, x, mname, mid)
    #                                 out_lines.append(src_str)

    #                     syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
    #                     sec, x, mname, mid = self.get_param_info(syn)
    #                     src_str = "0 %s %.6f %s %d w\n"%(sec, x, mname, mid)
    #                     out_lines.append(src_str)

    def flatten_weight_index(self, idxs):
        if self.weight_share:   # [iker, ci, co,]
            return idxs[0] * self.in_channels * self.out_channels + \
                   idxs[1] * self.out_channels + idxs[2]
        else:                   # [(out_idx, (ii, ji)), ci, co,]
            conn_idx = self.conn_list.index(idxs[0])
            return conn_idx * self.in_channels * self.out_channels + \
                   idxs[1] * self.out_channels + idxs[2]

    def flatten_bias_index(self, idxs):
        if self.weight_share:   # [co,]
            return idxs[0]
        else:                   # [co, io, jo]
            return idxs[0] * self.out_shape[1] * self.out_shape[2] + \
                   idxs[1] * self.out_shape[2] + idxs[2]

    def add_weight_param(self, seg2synlist, index_start, w_out_lines, dw_out_lines):
        if self.weight_share:
            self.nweight = self.weight.size
            self.nbias = self.bias.size
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for co in range(self.out_channels):
                    gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                    if self.pc.gid_exists(gid_cell):
                        for iker, padded_in_idx in enumerate(padded_in_idxs):
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                for ci in range(self.in_channels):
                                    #self.synarray_forward[out_idx, (ii, ji)][ci, co].w = self.weight[iker, ci, co]
                                    syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                    sec, x, mname, mid = self.get_param_info(syn)
                                    idx = index_start + self.flatten_weight_index([iker, ci, co,])
                                    w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                                    w_out_lines.append(w_str)
                                    dw_str = "0 %s %.6f %s %d delta_w %d\n"%(sec, x, mname, mid, idx)
                                    dw_out_lines.append(dw_str)
                        #self.synarray_bias[co, out_idx[0], out_idx[1]].w = self.bias[co]
                        syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                        sec, x, mname, mid = self.get_param_info(syn)
                        idx = index_start + self.nweight + self.flatten_bias_index([co,])
                        w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                        w_out_lines.append(w_str)
                        dw_str = "0 %s %.6f %s %d delta_w %d\n"%(sec, x, mname, mid, idx)
                        dw_out_lines.append(dw_str)
            if self.requires_input_grad:
                for out_idx, padded_in_idxs in self.conn_out2in.items():
                    for ci in range(self.in_channels):
                        for iker, padded_in_idx in enumerate(padded_in_idxs):
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                gid_grad = self.gids_grad[ci, ii, ji]
                                if self.pc.gid_exists(gid_grad):
                                    for co in range(self.out_channels):
                                        #self.synarray_backward[out_idx, (ii, ji)][ci, co].w = self.weight[iker, ci, co]
                                        syn = self.synarray_backward[out_idx, (ii, ji)][ci, co]
                                        sec, x, mname, mid = self.get_param_info(syn)
                                        idx = index_start + self.flatten_weight_index([iker, ci, co,])
                                        w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                                        w_out_lines.append(w_str)
        else:
            self.nweight = len(self.weight) * self.in_channels * self.out_channels
            self.nbias = self.bias.size
            self.conn_list = list(self.weight.keys())
            for out_idx, padded_in_idxs in self.conn_out2in.items():
                for co in range(self.out_channels):
                    gid_cell = self.gids_cell[co, out_idx[0], out_idx[1]]
                    if self.pc.gid_exists(gid_cell):
                        for padded_in_idx in padded_in_idxs:
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                for ci in range(self.in_channels):
                                    #self.synarray_forward[out_idx, (ii, ji)][ci, co].w = self.weight[out_idx, (ii, ji)][ci, co]
                                    syn = self.synarray_forward[out_idx, (ii, ji)][ci, co]
                                    sec, x, mname, mid = self.get_param_info(syn)
                                    idx = index_start + self.flatten_weight_index([(out_idx, (ii, ji)), ci, co,])
                                    w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                                    w_out_lines.append(w_str)
                                    dw_str = "0 %s %.6f %s %d delta_w %d\n"%(sec, x, mname, mid, idx)
                                    dw_out_lines.append(dw_str)
                        #self.synarray_bias[co, out_idx[0], out_idx[1]].w = self.bias[co, out_idx[0], out_idx[1]]
                        syn = self.synarray_bias[co, out_idx[0], out_idx[1]]
                        sec, x, mname, mid = self.get_param_info(syn)
                        idx = index_start + self.nweight + self.flatten_bias_index([co, out_idx[0], out_idx[1],])
                        w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                        w_out_lines.append(w_str)
                        dw_str = "0 %s %.6f %s %d delta_w %d\n"%(sec, x, mname, mid, idx)
                        dw_out_lines.append(dw_str)
            if self.requires_input_grad:
                for out_idx, padded_in_idxs in self.conn_out2in.items():
                    for ci in range(self.in_channels):
                        for padded_in_idx in padded_in_idxs:
                            ii, ji = padded_in_idx[0] - self.padding[0], padded_in_idx[1] - self.padding[1]
                            if ii >= 0 and ji >= 0 and ii < self.in_shape[1] and ji < self.in_shape[2]:
                                gid_grad = self.gids_grad[ci, ii, ji]
                                if self.pc.gid_exists(gid_grad):
                                    for co in range(self.out_channels):
                                        #self.synarray_backward[out_idx, (ii, ji)][ci, co].w = self.weight[out_idx, (ii, ji)][ci, co]
                                        syn = self.synarray_backward[out_idx, (ii, ji)][ci, co]
                                        sec, x, mname, mid = self.get_param_info(syn)
                                        idx = index_start + self.flatten_weight_index([(out_idx, (ii, ji)), ci, co,])
                                        w_str = "0 %s %.6f %s %d w %d\n"%(sec, x, mname, mid, idx)
                                        w_out_lines.append(w_str)

