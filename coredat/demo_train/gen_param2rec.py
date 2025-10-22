import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--trainlen', type=int, default=60000)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-r', '--ref_file', type=str, default='')
    parser.add_argument('-o', '--output_file', type=str, default='')

    args, _ = parser.parse_known_args()

    src_file = open(args.ref_file, 'r')
    src_lines = src_file.readlines()
    total_len = src_lines[0].split()[0]
    src_file.close()

    nimg = args.trainlen
    batch_size = args.batch_size
    train_epochs = args.epochs
    
    sim_time_per_img = 500 # 50ms
    sim_time_per_epoch = nimg * sim_time_per_img / batch_size # ms
    save_times = [int(i * sim_time_per_epoch + 20) for i in range(1, train_epochs + 1)]
    line1 = f"{total_len} {train_epochs:d} "
    for t in save_times:
        line1 += (str(t) + " ")
    line1 += "\n"

    with open(args.output_file, 'w+') as f:
        f.writelines(line1)
        f.writelines(src_lines[1:])