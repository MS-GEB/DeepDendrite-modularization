import os
import numpy as np
from neuron import h
import layers

class BaseNet:
    def __init__(self, idm, seed=None):
        self.idm = idm
        self.rng = np.random.default_rng(seed=seed)
        self.pc = h.ParallelContext()
        self.seg2synlist = {}

        self.train_layers = []
        self.target_var_list = [] # (source_sid, target_seg, target_mech, target_var)

    def is_train(self, start, end, lr_rate):
        for layer in self.train_layers:
            layer.is_train(start=start, end=end, lr_rate=lr_rate)

    def is_test(self):
        for layer in self.train_layers:
            layer.is_test()

    def set_stim(self, input, target=None):
        self.input.set_input(input=input)
        if target:
            self.output.set_target(target=target)

    def load_weights(self):
        for layer in self.train_layers:
            layer.load_weights()

    def save_weights(self):
        for layer in self.train_layers:
            layer.save_weights()

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

    def gen_gap_param_file(self, save_dir=None, save_name=None):
        out_lines = []
        for i, (sid, tgt_seg, tgt_mech, tgt_var) in enumerate(self.target_var_list):
            src_seg, src_mech, src_var = self.idm.sid2param[sid]
            src_id = 0
            tgt_id = 0
            src_mech_name = "None"
            tgt_mech_name = "None"
            if src_mech is not None:
                #src_id = self.get_mech_id(src_mech, self.seg2synlist[src_seg])
                src_mech_name = str(src_mech)
                i1 = src_mech_name.find('[')
                src_mech_name = src_mech_name[:i1]
                src_id = self.seg2synlist[src_seg][src_mech_name].index(src_mech)
            if tgt_mech is not None:
                #tgt_id = self.seg2synlist[tgt_seg].index(tgt_mech)
                #tgt_id = self.get_mech_id(tgt_mech, self.seg2synlist[tgt_seg])
                tgt_mech_name = str(tgt_mech)
                i1 = tgt_mech_name.find('[')
                tgt_mech_name = tgt_mech_name[:i1]
                tgt_id = self.seg2synlist[tgt_seg][tgt_mech_name].index(tgt_mech)

            src_str = "%s %.6f %s %d %s"%(src_seg.sec, src_seg.x, src_mech_name, src_id, src_var)
            tgt_str = "%s %.6f %s %d %s"%(tgt_seg.sec, tgt_seg.x, tgt_mech_name, tgt_id, tgt_var)
            # if i % 10000 == 0:
            #     print(src_str)
            #     print(tgt_str)
            #     print("\n")
            out_lines.append(src_str + " " + tgt_str + "\n")

        if save_dir is not None and save_name is not None:
            with open(os.path.join(save_dir, save_name), "w") as f:
                f.writelines("%d\n"%len(out_lines))
                f.writelines(out_lines)

        return out_lines

    def gen_stim_file(self, stim_list, save_dir=None, save_name=None):
        out_lines = []
        for stim in stim_list:
            stim_name = str(stim)
            i1 = stim_name.find("[")
            i2 = stim_name.find("]")
            order = int(stim_name[i1 + 1 : i2])
            out_lines.append("1 NetStim %d interval\n"%order)
            out_lines.append("1 NetStim %d start\n"%order)

        if save_dir is not None and save_name is not None:
            with open(os.path.join(save_dir, save_name), "w") as f:
                f.writelines("%d\n"%len(out_lines))
                f.writelines(out_lines)
        return out_lines

    def gen_label_file(self, label_syn_list, nclasses, save_dir=None, save_name=None):
        out_lines = []
        for mech in label_syn_list:
            seg = mech.get_segment()
            mech_name = str(mech)
            i1 = mech_name.find('[')
            mech_name = mech_name[:i1]
            mech_id = self.seg2synlist[seg][mech_name].index(mech)
            #mech_id = self.get_mech_id(mech, self.seg2synlist[seg])
            for i in range(nclasses):
                out_lines.append("0 %s %.6f %s %d target_%d\n"%(seg.sec, seg.x, mech_name, mech_id, i))

        if save_dir is not None and save_name is not None:
            with open(os.path.join(save_dir, save_name), "w") as f:
                f.writelines("%d\n"%len(out_lines))
                f.writelines(out_lines)
        return out_lines

    def gen_set_weight_file(self, save_dir=None, save_name=None):
        w_out_lines = []
        dw_out_lines = []
        total_index = 0
        for layer in self.train_layers:
            layer.add_weight_param(self.seg2synlist, total_index, w_out_lines, dw_out_lines)
            total_index += layer.nweight + layer.nbias

        if save_dir is not None and save_name is not None:
            with open(os.path.join(save_dir, "weights"), "w") as f:
                f.writelines("%d %d\n"%(len(w_out_lines), total_index))
                f.writelines(w_out_lines)

            with open(os.path.join(save_dir, "delta_weights"), "w") as f:
                f.writelines("%d\n"%len(dw_out_lines))
                f.writelines(dw_out_lines)
        return w_out_lines, dw_out_lines, total_index

    # def gen_weight_file(self, save_dir, save_name):
    #     out_lines = []
    #     for layer in self.train_layers:
    #         layer.add_weights(self.seg2synlist, out_lines)

    #     if save_dir is not None and save_name is not None:
    #         with open(os.path.join(save_dir, save_name), "w") as f:
    #             f.writelines("%d\n"%len(out_lines))
    #             f.writelines(out_lines)
    #     return out_lines

    def gen_trainable_syn_params(self, save_dir=None, save_name=None):
        param_list = []
        for i, layer in enumerate(self.train_layers):
            if isinstance(layer.synarray_forward, dict):
                for conn in layer.synarray_forward.values():
                    for syn in conn.flatten():
                        sec, x, mname, mid = layer.get_param_info(syn)
                        param_list.append("0 %s %.6f %s %d w\n"%(sec, x, mname, mid))
            else: 
                synarray = layer.synarray_forward.flatten()
                for syn in synarray:
                    sec, x, mname, mid = layer.get_param_info(syn)
                    param_list.append("0 %s %.6f %s %d w\n"%(sec, x, mname, mid))

            if hasattr(layer, "synarray_bias"):
                synarray = layer.synarray_bias.flatten()
                for syn in synarray:
                    sec, x, mname, mid = layer.get_param_info(syn)
                    param_list.append("0 %s %.6f %s %d w\n"%(sec, x, mname, mid))

        return param_list


class MnistTinyFCNet(BaseNet):
    def __init__(self, idm, seed=None):
        super(MnistTinyFCNet, self).__init__(idm, seed)

        # forward
        self.input = layers.NetStimInput(N_in=1*28*28, pc=self.pc, idm=self.idm, rng=self.rng,
                                         seg2synlist=self.seg2synlist, target_var_list=self.target_var_list)
        self.fc1 = layers.FullyConnectedHPC(N_in=1*28*28, N_out=64, N_multi=1, sids_in=self.input.sids_out,
                                            in_activation='Linear', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.output = layers.OutputPoint(N_in=64, N_out=10, sids_in=self.fc1.sids_out,
                                         in_activation='ReLU', pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list=self.target_var_list)
        self.train_layers = [self.fc1, self.output]
    
    def setup_backward(self):
        # backward
        self.output.setup_backward()
        self.fc1.setup_backward(sids_grad_out=self.output.sids_grad_in)


class MnistFCNet(BaseNet):
    def __init__(self, idm, seed=None):
        super(MnistFCNet, self).__init__(idm, seed)

        # forward
        self.input = layers.NetStimInput(N_in=1*28*28, pc=self.pc, idm=self.idm, rng=self.rng,
                                         seg2synlist=self.seg2synlist, target_var_list=self.target_var_list)
        self.fc1 = layers.FullyConnectedHPC(N_in=1*28*28, N_out=256, N_multi=1, sids_in=self.input.sids_out,
                                            in_activation='Linear', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.fc2 = layers.FullyConnectedHPC(N_in=256, N_out=256, N_multi=1, sids_in=self.fc1.sids_out,
                                            in_activation='ReLU', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.fc3 = layers.FullyConnectedHPC(N_in=256, N_out=256, N_multi=1, sids_in=self.fc2.sids_out,
                                            in_activation='ReLU', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.fc4 = layers.FullyConnectedHPC(N_in=256, N_out=256, N_multi=1, sids_in=self.fc3.sids_out,
                                            in_activation='ReLU', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.fc5 = layers.FullyConnectedHPC(N_in=256, N_out=256, N_multi=1, sids_in=self.fc4.sids_out,
                                            in_activation='ReLU', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.output = layers.OutputPoint(N_in=256, N_out=10, sids_in=self.fc5.sids_out,
                                         in_activation='ReLU', pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list=self.target_var_list)
        self.train_layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.output]

    def setup_backward(self):
        # backward
        self.output.setup_backward()
        self.fc5.setup_backward(sids_grad_out=self.output.sids_grad_in)
        self.fc4.setup_backward(sids_grad_out=self.fc5.sids_grad_in)
        self.fc3.setup_backward(sids_grad_out=self.fc4.sids_grad_in)
        self.fc2.setup_backward(sids_grad_out=self.fc3.sids_grad_in)
        self.fc1.setup_backward(sids_grad_out=self.fc2.sids_grad_in, requires_input_grad=False)


class MnistConvNet(BaseNet):
    def __init__(self, idm, seed=None):
        super(MnistConvNet, self).__init__(idm, seed)

        # forward
        self.input = layers.NetStimInput(N_in=1*28*28, pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list=self.target_var_list)
        self.input_reshape = np.reshape(self.input.sids_out, (1, 28, 28))
        self.conv1 = layers.Conv2DPoint(in_channels=1, out_channels=32, sids_in=self.input_reshape,
                                        kernel_size=3, stride=2, padding=0, weight_share=True, in_activation='Linear',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list = self.target_var_list)
        self.conv2 = layers.Conv2DPoint(in_channels=32, out_channels=64, sids_in=self.conv1.sids_out,
                                        kernel_size=3, stride=2, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.flat = np.reshape(self.conv2.sids_out, -1)
        self.fc1 = layers.FullyConnectedHPC(N_in=self.flat.size, N_out=1024, N_multi=1, sids_in=self.flat,
                                            in_activation='ReLU', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.output = layers.OutputPoint(N_in=1024, N_out=10, sids_in=self.fc1.sids_out,
                                         in_activation='ReLU', pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list=self.target_var_list)
        self.train_layers = [self.conv1, self.conv2, self.fc1, self.output]

    def setup_backward(self):
        # backward
        self.output.setup_backward()
        self.fc1.setup_backward(sids_grad_out=self.output.sids_grad_in)
        self.conv2.setup_backward(sids_grad_out=np.reshape(self.fc1.sids_grad_in, self.conv2.out_shape))
        self.conv1.setup_backward(sids_grad_out=self.conv2.sids_grad_in, requires_input_grad=False)


class Cifar10FCNet(BaseNet):
    def __init__(self, idm, seed=None):
        super(Cifar10FCNet, self).__init__(idm, seed)

        # forward
        self.input = layers.NetStimInput(N_in=3*32*32, pc=self.pc, idm=self.idm, rng=self.rng,
                                         seg2synlist=self.seg2synlist, target_var_list=self.target_var_list)
        self.fc1 = layers.FullyConnectedHPC(N_in=3*32*32, N_out=1024, N_multi=1, sids_in=self.input.sids_out,
                                            in_activation='Linear', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list = self.target_var_list)
        self.fc2 = layers.FullyConnectedHPC(N_in=1024, N_out=1024, N_multi=1, sids_in=self.fc1.sids_out,
                                            in_activation='ReLU', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.fc3 = layers.FullyConnectedHPC(N_in=1024, N_out=1024, N_multi=1, sids_in=self.fc2.sids_out,
                                            in_activation='ReLU', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.output = layers.OutputPoint(N_in=1024, N_out=10, sids_in=self.fc3.sids_out,
                                         in_activation='ReLU', pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list=self.target_var_list)
        self.train_layers = [self.fc1, self.fc2, self.fc3, self.output]
        #self.train_layers = [self.output]

    def setup_backward(self):
        # backward
        self.output.setup_backward()
        self.fc3.setup_backward(sids_grad_out=self.output.sids_grad_in)
        self.fc2.setup_backward(sids_grad_out=self.fc3.sids_grad_in)
        self.fc1.setup_backward(sids_grad_out=self.fc2.sids_grad_in)


class Cifar10ConvNet(BaseNet):
    def __init__(self, idm, seed=None):
        super(Cifar10ConvNet, self).__init__(idm, seed)

        # forward
        self.input = layers.NetStimInput(N_in=3*32*32, pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list=self.target_var_list)
        self.input_reshape = np.reshape(self.input.sids_out, (3, 32, 32))
        self.conv1 = layers.Conv2DPoint(in_channels=3, out_channels=64, sids_in=self.input_reshape,
                                        kernel_size=5, stride=2, padding=0, weight_share=True, in_activation='Linear',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list = self.target_var_list)
        self.conv2 = layers.Conv2DPoint(in_channels=64, out_channels=128, sids_in=self.conv1.sids_out,
                                        kernel_size=5, stride=2, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv3 = layers.Conv2DPoint(in_channels=128, out_channels=256, sids_in=self.conv2.sids_out,
                                        kernel_size=3, stride=1, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.flat = np.reshape(self.conv3.sids_out, -1)
        self.fc1 = layers.FullyConnectedHPC(N_in=self.flat.size, N_out=1024, N_multi=1, sids_in=self.flat,
                                            in_activation='ReLU', pc=self.pc, idm=self.idm,
                                            rng=self.rng, seg2synlist=self.seg2synlist,
                                            target_var_list=self.target_var_list)
        self.output = layers.OutputPoint(N_in=1024, N_out=10, sids_in=self.fc1.sids_out,
                                         in_activation='ReLU', pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list=self.target_var_list)
        self.train_layers = [self.conv1, self.conv2, self.conv3, self.fc1, self.output]

    def setup_backward(self):
        # backward
        self.output.setup_backward()
        self.fc1.setup_backward(sids_grad_out=self.output.sids_grad_in)
        self.conv3.setup_backward(sids_grad_out=np.reshape(self.fc1.sids_grad_in, self.conv3.out_shape))
        self.conv2.setup_backward(sids_grad_out=self.conv3.sids_grad_in)
        self.conv1.setup_backward(sids_grad_out=self.conv2.sids_grad_in, requires_input_grad=False)


class ImageNetConvNet(BaseNet):
    def __init__(self, idm, seed=None):
        super(ImageNetConvNet, self).__init__(idm, seed)

        # forward
        self.input = layers.NetStimInput(N_in=3*224*224, pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list = self.target_var_list)
        self.input_reshape = np.reshape(self.input.sids_out, (3, 224, 224))
        self.conv1 = layers.Conv2DPoint(in_channels=3, out_channels=48, sids_in=self.input_reshape,
                                        kernel_size=9, stride=4, padding=0, weight_share=True, in_activation='Linear',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list = self.target_var_list)
        self.conv2 = layers.Conv2DPoint(in_channels=48, out_channels=48, sids_in=self.conv1.sids_out,
                                        kernel_size=3, stride=2, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv3 = layers.Conv2DPoint(in_channels=48, out_channels=96, sids_in=self.conv2.sids_out,
                                        kernel_size=5, stride=1, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv4 = layers.Conv2DPoint(in_channels=96, out_channels=96, sids_in=self.conv3.sids_out,
                                        kernel_size=3, stride=2, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv5 = layers.Conv2DPoint(in_channels=96, out_channels=192, sids_in=self.conv4.sids_out,
                                        kernel_size=3, stride=1, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv6 = layers.Conv2DPoint(in_channels=192, out_channels=192, sids_in=self.conv5.sids_out,
                                        kernel_size=3, stride=2, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv7 = layers.Conv2DPoint(in_channels=192, out_channels=384, sids_in=self.conv6.sids_out,
                                        kernel_size=3, stride=1, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.flat = np.reshape(self.conv7.sids_out, -1)
        self.output = layers.OutputHPC(N_in=self.flat.size, N_out=1000, N_multi=1, sids_in=self.flat,
                                        in_activation='ReLU', pc=self.pc, idm=self.idm,
                                        rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.train_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.output]

    def setup_backward(self):
        # backward
        self.output.setup_backward()
        self.conv7.setup_backward(sids_grad_out=np.reshape(self.output.sids_grad_in, self.conv7.out_shape))
        self.conv6.setup_backward(sids_grad_out=self.conv7.sids_grad_in)
        self.conv5.setup_backward(sids_grad_out=self.conv6.sids_grad_in)
        self.conv4.setup_backward(sids_grad_out=self.conv5.sids_grad_in)
        self.conv3.setup_backward(sids_grad_out=self.conv4.sids_grad_in)
        self.conv2.setup_backward(sids_grad_out=self.conv3.sids_grad_in)
        self.conv1.setup_backward(sids_grad_out=self.conv2.sids_grad_in, requires_input_grad=False)


class TinyImageNetConvNet(BaseNet):
    def __init__(self, idm, seed=None):
        super(TinyImageNetConvNet, self).__init__(idm, seed)

        # forward
        self.input = layers.NetStimInput(N_in=3*64*64, pc=self.pc, idm=self.idm,
                                         rng=self.rng, seg2synlist=self.seg2synlist,
                                         target_var_list = self.target_var_list)
        self.input_reshape = np.reshape(self.input.sids_out, (3, 64, 64))
        self.conv1 = layers.Conv2DPoint(in_channels=3, out_channels=48, sids_in=self.input_reshape,
                                        kernel_size=5, stride=2, padding=0, weight_share=True, in_activation='Linear',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list = self.target_var_list)
        self.conv2 = layers.Conv2DPoint(in_channels=48, out_channels=48, sids_in=self.conv1.sids_out,
                                        kernel_size=3, stride=2, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv3 = layers.Conv2DPoint(in_channels=48, out_channels=96, sids_in=self.conv2.sids_out,
                                        kernel_size=5, stride=1, padding=2, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv4 = layers.Conv2DPoint(in_channels=96, out_channels=96, sids_in=self.conv3.sids_out,
                                        kernel_size=3, stride=2, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv5 = layers.Conv2DPoint(in_channels=96, out_channels=192, sids_in=self.conv4.sids_out,
                                        kernel_size=3, stride=1, padding=1, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.conv6 = layers.Conv2DPoint(in_channels=192, out_channels=192, sids_in=self.conv5.sids_out,
                                        kernel_size=3, stride=2, padding=0, weight_share=True, in_activation='ReLU',
                                        pc=self.pc, idm=self.idm, rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.flat = np.reshape(self.conv6.sids_out, -1)
        self.output = layers.OutputHPC(N_in=self.flat.size, N_out=200, N_multi=1, sids_in=self.flat,
                                        in_activation='ReLU', pc=self.pc, idm=self.idm,
                                        rng=self.rng, seg2synlist=self.seg2synlist,
                                        target_var_list=self.target_var_list)
        self.train_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.output]
    
    def setup_backward(self):
        # backward
        self.output.setup_backward()
        self.conv6.setup_backward(sids_grad_out=np.reshape(self.output.sids_grad_in, self.conv6.out_shape))
        self.conv5.setup_backward(sids_grad_out=self.conv6.sids_grad_in)
        self.conv4.setup_backward(sids_grad_out=self.conv5.sids_grad_in)
        self.conv3.setup_backward(sids_grad_out=self.conv4.sids_grad_in)
        self.conv2.setup_backward(sids_grad_out=self.conv3.sids_grad_in)
        self.conv1.setup_backward(sids_grad_out=self.conv2.sids_grad_in, requires_input_grad=False)
