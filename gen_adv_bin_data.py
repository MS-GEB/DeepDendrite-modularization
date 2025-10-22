import os
import struct
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source-folder', type=str, default='./adversarial/mnist')
    parser.add_argument('-d', '--dest-folder', type=str, default='./datasets/mnist')

    args, _ = parser.parse_known_args()

    source_filename_list = os.listdir(args.source_folder)
    source_filename_list.sort()
    for source_filename in source_filename_list:
        source_file = np.load(os.path.join(args.source_folder, source_filename))
        x_test, y_test = source_file['x_test'], source_file['y_test']
        print(x_test.shape, y_test.shape)

        assert (x_test.max() <= 1. and x_test.min() >= 0.)
        x_test = x_test.transpose((0, 3, 1, 2))     # channel first
        x_test = x_test.reshape((x_test.shape[0], -1))
        y_test = y_test.reshape((-1,))
        x_test = x_test.astype('float64')
        y_test = y_test.astype('int32')
        print(x_test.shape, y_test.shape)
        
        dest_filename = source_filename.split('_')[0] + '_test'
        interval_test = 50 / (x_test.flatten() + 0.01)
        pack_test = np.stack((interval_test, 90 + interval_test), axis=-1).flatten()
        if not os.path.isdir(args.dest_folder):
            os.makedirs(args.dest_folder)
        with open(os.path.join(args.dest_folder, dest_filename), "wb") as f:
            f.write(struct.pack("ii", x_test.shape[0], x_test.shape[1] * 2))
            f.write(pack_test.tobytes())
            f.write(y_test.tobytes())
