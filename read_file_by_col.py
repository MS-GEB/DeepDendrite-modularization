import struct
import numpy as np


def read_data_from_file(file_name):
    f = open(file_name, "rb")
    line = f.readline()
    line = line.split()
    nparam = int(line[0])
    rec_len = int(line[1])
    nrec = int(line[2])
    nlast = int(line[3])
    print(nparam, rec_len, nrec, nlast)

    all_data = np.zeros((nparam, rec_len * nrec + nlast), dtype='float32')
    for i in range(nrec + 1):
        if i == nrec:
            if nlast == 0:
                break
            data_len = nlast
        else:
            data_len = rec_len
        bin_data = f.read(nparam * data_len * 4)
        fmt = f'{nparam * rec_len:d}f'
        data = struct.unpack(fmt, bin_data)
        for istep in range(data_len):
            all_data[:, istep] = data[istep * nparam : (istep + 1) * nparam]
    f.close()

    return all_data


def format_trans(data, nout):
    nparam, rec_len = data.shape
    batchsize = int(nparam / nout)
    idx = 0
    ret = np.zeros((nout, rec_len * batchsize))
    for irec in range(rec_len):
        for ib in range(batchsize):
            ret[:, idx] = data[ib * nout: ib * nout + nout, irec]
            idx += 1
    return ret


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='')
    parser.add_argument('-o', '--output_file', type=str, default='')
    parser.add_argument('-t', '--trans', action='store_true')
    parser.add_argument('-n', '--nclasses', type=int, default=10)

    args, _ = parser.parse_known_args()
    
    print(args.input_file, args.output_file)
    rec_data = read_data_from_file(args.input_file)
    if args.trans:
        save_data = format_trans(rec_data, args.nclasses)
    else:
        save_data = rec_data
    print(rec_data.shape, rec_data.min(), rec_data.max())
    print(save_data.shape, save_data.min(), save_data.max())
    np.save(args.output_file, save_data)
