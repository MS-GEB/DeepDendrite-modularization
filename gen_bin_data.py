import os
import struct
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mnist')
    parser.add_argument('-f', '--folder', type=str, default='./datasets/mnist')
    parser.add_argument('-e', '--epochs', type=int, default=1)

    args, _ = parser.parse_known_args()

    if args.dataset == 'mnist':
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

    if args.dataset == 'mnist':
        dest_dir = args.folder
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        
        interval_train, interval_test = 50 / (x_train.flatten() + 0.01), 50 / (x_test.flatten() + 0.01)
        pack_train = np.stack((interval_train, 90 + interval_train), axis=-1).flatten()
        pack_test = np.stack((interval_test, 90 + interval_test), axis=-1).flatten()
        with open(os.path.join(dest_dir, "stim_img_train"), "wb") as f:
            f.write(struct.pack("ii", x_train.shape[0], x_train.shape[1] * 2))
            f.write(pack_train.tobytes())
            f.write(y_train.tobytes())
        with open(os.path.join(dest_dir, "stim_img_test"), "wb") as f:
            f.write(struct.pack("ii", x_test.shape[0], x_test.shape[1] * 2))
            f.write(pack_test.tobytes())
            f.write(y_test.tobytes())
        
        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")
