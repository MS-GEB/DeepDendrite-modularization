import os
import numpy as np
import tensorflow as tf
from resnet20 import make_resnet
import foolbox

dataset = 'mnist'
learning_rate = 0.01
lr_str = '1e-2'
batch_size = 64
epochs = 60
load_trained = False
attack_method = 'fgsm'


def attack_model(model, method, bounds, x_test, y_test, epsilons):
    fmodel = foolbox.TensorFlowModel(model, bounds)
    clean_acc = foolbox.accuracy(fmodel, x_test, y_test)
    print(f"clean acc:{clean_acc:f}")

    if method == 'fgsm':
        attack = foolbox.attacks.FGSM()
    elif method == 'pgd':
        attack = foolbox.attacks.PGD()
    else:
        raise NotImplementedError(f'Unknown attack method: {method}')
    raw_advs, clipped_advs, success = attack(fmodel, x_test, y_test, epsilons=epsilons)
    robust_acc = 1 - success.numpy().mean(axis=-1)
    for eps, acc in zip(epsilons, robust_acc):
        print(f"eps:{eps:f} acc:{acc:f}")

    return clipped_advs


def prepare_data(dataset):
    if dataset == 'mnist':
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        y_train = y_train.astype('int32')
        x_test = x_test.astype('float32') / 255
        y_test = y_test.astype('int32')
        x_train = np.expand_dims(x_train, -1)
        y_train = y_train.flatten()
        x_test = np.expand_dims(x_test, -1)
        y_test = y_test.flatten()
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")
    
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = prepare_data(dataset)

    model = make_resnet(x_train.shape[1:], 10, 3, 1e-5)
    # model.summary()
    optim = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optim, metrics=["acc"])
    if not load_trained:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
        model.save_weights(f"resnet20_{dataset}_{epochs:d}epochs_sgd{lr_str}.h5")
    else:
        model.load_weights(f"resnet20_{dataset}_{epochs:d}epochs_sgd{lr_str}.h5")
    model.evaluate(x_test, y_test, batch_size=batch_size)

    epsilons = [0.02 * (i + 1) for i in range(20)]
    adv_img_list = attack_model(model, attack_method, (0, 1), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test), epsilons)

    if not os.path.isdir(f"./{dataset}"):
        os.makedirs(f"./{dataset}")
    for adv_img, eps in zip(adv_img_list, epsilons):
       save_name = f"./{dataset}/{attack_method}{eps:.3f}_test.npz"
       np.savez(save_name, x_test=adv_img.numpy(), y_test=y_test)
