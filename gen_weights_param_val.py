import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_file', type=str, default='')
    parser.add_argument('-e', '--epoch', type=int, default=-1)
    parser.add_argument('-r', '--ref_file', type=str, default='')
    parser.add_argument('-o', '--output_file', type=str, default='')

    args, _ = parser.parse_known_args()

    w_file = args.weights_file
    src_filename = args.ref_file
    save_file = args.output_file

    w = np.load(w_file)
    print(w[:, args.epoch].shape, w[:, args.epoch].min(), w[:, args.epoch].max())
    src_file = open(src_filename, 'r')
    nweights = w.shape[0]

    out_lines = []
    for i, line in enumerate(src_file.readlines()):
        if i == 0:
            continue
        val = w[(i - 1) % nweights, args.epoch]
        line = line.split()
        single_line = ''
        for j in range(len(line) - 1):
            single_line += line[j] + ' '
        single_line += f'{val:.6f}\n'
        out_lines.append(single_line)
        # out_lines.append("%s %.6f\n"%(line[:-2], val))
    src_file.close()

    with open(save_file, 'w') as f:
        f.writelines("%d\n"%len(out_lines))
        f.writelines(out_lines)
