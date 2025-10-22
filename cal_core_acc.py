import sys
import numpy as np


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='')
    parser.add_argument('-d', '--dataset', type=str, default='mnist')
    parser.add_argument('-l', '--testlen', type=int, default=10000)

    args, _ = parser.parse_known_args()

    core_output = np.load(args.output)
    if args.dataset == 'mnist':
        with np.load('./datasets/mnist.npz', allow_pickle=True) as f:
            y_test = f['y_test']
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")
    
    nimg = min(args.testlen, core_output.shape[1])
    core_res = np.argmax(core_output, axis=0)[:nimg]
    labels = y_test[:nimg].flatten()
    ncorrect = np.where(core_res == labels)[0].shape[0]
    print(f"nimg:{nimg:d} acc:{float(ncorrect) / nimg * 100:.4f}%")
