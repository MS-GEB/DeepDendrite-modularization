import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

workdir = "../.."
deepdend = "../../install/bin/deepdendrite"
dataset = 'mnist'
batchsize = 4
coredat_train = f"{workdir}/coredat/demo_train"
coredat_test = f"{workdir}/coredat/demo_test"
timestep = 5
lr_start = 200
lr_end = 500
datapath = f"{workdir}/datasets/{dataset}"
nclasses = 10
epochs = 60
test_epoch = -1
testlen = 10000
testtime = testlen * 500 // batchsize + 20
weights_file = f"weights_51to60.npy"
suffix = f"{epochs:d}_test{epochs:d}"
weights_param_val_file = f"weights_param_val_{suffix}"
base_model_list = ['res', 'vgg']
attack_method_list = ['fgsm', 'pgd5', 'deepfool']
epsilon_list = [0.02 * (i + 1) for i in range(10)]

cmd = f"echo $(TZ=Asia/Shanghai date +%F%n%T)"
os.system(cmd)

cmd = f"python3 {workdir}/gen_weights_param_val.py -w {coredat_train}/{weights_file} "\
      f"-e {test_epoch:d} -r {coredat_test}/weights_param_val -o {coredat_test}/{weights_param_val_file}"
os.system(cmd)

# clean
cmd = f"OMP_NUM_THREADS=1 {deepdend} -d {coredat_test} -e {testtime:d} -dt {timestep:.2f} -v 0 --nonv-gap 1 --need-record 1 "\
      f"--rec-file {coredat_test}/test_outputs_soma_v_{suffix} --input-file {datapath}/stim_img_test --param-val {coredat_test}/{weights_param_val_file} "\
      f"--lr_start {lr_start:d} --lr_end {lr_end:d} --mindelay 20 --training 0 --testing 1 --batchsize {batchsize:d} --cell-permute 3 --cell-nthread 16 --gpu"
os.system(cmd)

cmd = f"python3 {workdir}/read_file_by_col.py -i {coredat_test}/test_outputs_soma_v_{suffix} "\
      f"-o {coredat_test}/test_outputs_soma_v_{suffix}.npy -t -n {nclasses:d}"
os.system(cmd)

# attack
for base_model in base_model_list:
    for attack_method in attack_method_list:

        for epsilon in epsilon_list:
            cmd = f"OMP_NUM_THREADS=1 {deepdend} -d {coredat_test} -e {testtime:d} -dt {timestep:.2f} -v 0 --nonv-gap 1 --need-record 1 "\
                  f"--rec-file {coredat_test}/{attack_method}{base_model}{epsilon:.2f}_soma_v_{suffix} --input-file {datapath}/{attack_method}{base_model}{epsilon:.2f}_img_test "\
                  f"--param-val {coredat_test}/{weights_param_val_file} --lr_start {lr_start:d} --lr_end {lr_end:d} --mindelay 20 --training 0 --testing 1 --batchsize {batchsize:d} --cell-permute 3 --cell-nthread 16 --gpu"
            os.system(cmd)

            cmd = f"python3 {workdir}/read_file_by_col.py -i {coredat_test}/{attack_method}{base_model}{epsilon:.2f}_soma_v_{suffix} "\
                  f"-o {coredat_test}/{attack_method}{base_model}{epsilon:.2f}_soma_v_{suffix}.npy -t -n {nclasses:d}"
            os.system(cmd)

        cmd = f"echo {base_model} {attack_method}"
        os.system(cmd)

        cmd = f"python3 {workdir}/cal_core_acc.py -o {coredat_test}/test_outputs_soma_v_{suffix}.npy -d {dataset} -l {testlen:d}"
        os.system(cmd)
        for epsilon in epsilon_list:
            cmd = f"python3 {workdir}/cal_core_acc.py -o {coredat_test}/{attack_method}{base_model}{epsilon:.2f}_soma_v_{suffix}.npy -d {dataset} -l {testlen:d}"
            os.system(cmd)

        cmd = f"echo $(TZ=Asia/Shanghai date +%F%n%T)"
        os.system(cmd)
