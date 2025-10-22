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
lr_scale = 0.005
total_epochs = 60
epoch_step = 10
trainlen = 60000
testlen = 10000
traintime = trainlen * epoch_step * 500 // batchsize + 20
testtime = testlen * 500 // batchsize + 20
test_step = 3

for epoch_start in range(1, total_epochs, epoch_step):
    epoch_end = epoch_start + epoch_step - 1

    # Train
    cmd = f"echo $(TZ=Asia/Shanghai date +%F%n%T)"
    os.system(cmd)
    cmd = f"cd {coredat_train}"
    os.system(cmd)
    if epoch_start == 1:
        print('### build train network ###')
        cmd = f"{workdir}/x86_64/special ./demo_train.py -d {dataset} -l {trainlen} -b {batchsize} -e {epoch_step} -f {coredat_train}"
        os.system(cmd)
        
    cmd = f"python3 {coredat_train}/gen_param2rec.py -l {trainlen} -b {batchsize} -e {epoch_step} -r {coredat_train}/param2rec -o {coredat_train}/param2rec"
    os.system(cmd)
    if epoch_start == 1:
        print('### start new ###')
        cmd = f"OMP_NUM_THREADS=1 {deepdend} -d {coredat_train} -e {traintime} -dt {timestep} -v 0 --nonv-gap 1 --need-record 1 --rec-file {coredat_train}/weights_{epoch_start}to{epoch_end} "\
              f"--input-file {datapath}/stim_img_train --lr_scale {lr_scale} --lr_start {lr_start} --lr_end {lr_end} --avg_train --mindelay 20 --training 1 --testing 0 --batchsize {batchsize} --shared_weight --cell-permute 3 --cell-nthread 16 --gpu"
        os.system(cmd)
    else:
        print('### load trained ###')
        prev_epoch_start = epoch_start - epoch_step
        prev_epoch_end = epoch_start - 1
        load_epoch = epoch_step - 1
        cmd = f"OMP_NUM_THREADS=1 {deepdend} -d {coredat_train} -e {traintime} -dt {timestep} -v 0 --nonv-gap 1 --need-record 1 --rec-file {coredat_train}/weights_{epoch_start}to{epoch_end} "\
              f"--input-file {datapath}/stim_img_train --load-weights-file {coredat_train}/weights_{prev_epoch_start}to{prev_epoch_end} --load-weights-epoch {load_epoch} --lr_scale {lr_scale} "\
              f"--lr_start {lr_start} --lr_end {lr_end} --avg_train --mindelay 20 --training 1 --testing 0 --batchsize {batchsize} --shared_weight --cell-permute 3 --cell-nthread 16 --gpu"
        os.system(cmd)
    cmd = f"python3 {workdir}/read_file_by_col.py -i {coredat_train}/weights_{epoch_start}to{epoch_end} -o {coredat_train}/weights_{epoch_start}to{epoch_end}.npy"
    os.system(cmd)

    # Test
    cmd = f"echo $(TZ=Asia/Shanghai date +%F%n%T)"
    os.system(cmd)
    cmd = f"cd {coredat_test}"
    os.system(cmd)
    if epoch_start == 1:
        print('### build test network ###')
        cmd = f"{workdir}/x86_64/special ./demo_test.py -d {dataset} -b {batchsize} -f {coredat_test} -w {coredat_train}/weights_{epoch_start}to{epoch_end}.npy"
        os.system(cmd)
    for test_epoch in range(epoch_start, epoch_end, test_step):
        print(f'test epoch {test_epoch}')
        idx = test_epoch - epoch_start
        cmd = f"python3 {workdir}/gen_weights_param_val.py -w {coredat_train}/weights_{epoch_start}to{epoch_end}.npy -e {idx} -r {coredat_test}/weights_param_val -o {coredat_test}/weights_param_val_{total_epochs}_test{test_epoch}"
        os.system(cmd)
        cmd = f"OMP_NUM_THREADS=1 {deepdend} -d {coredat_test} -e {testtime} -dt {timestep} -v 0 --nonv-gap 1 --need-record 1 --rec-file {coredat_test}/test_outputs_soma_v_{total_epochs}_test{test_epoch} "\
              f"--input-file {datapath}/stim_img_test --param-val {coredat_test}/weights_param_val_{total_epochs}_test{test_epoch} --lr_start {lr_start} --lr_end {lr_end} --mindelay 20 --training 0 --testing 1 --batchsize {batchsize} --cell-permute 3 --cell-nthread 16 --gpu"
        os.system(cmd)
        cmd = f"python3 {workdir}/read_file_by_col.py -i {coredat_test}/test_outputs_soma_v_{total_epochs}_test{test_epoch} -o {coredat_test}/test_outputs_soma_v_{total_epochs}_test{test_epoch}.npy -t -n {nclasses}"
        os.system(cmd)
    for test_epoch in range(epoch_start, epoch_end, test_step):
        cmd = f"python3 {workdir}/cal_core_acc.py -o {coredat_test}/test_outputs_soma_v_{total_epochs}_test{test_epoch}.npy -d {dataset} -l {testlen}"
        os.system(cmd)
