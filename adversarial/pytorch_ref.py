import os
import torch
import numpy as np
import pickle
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn

torch.manual_seed(0)

dataset = 'cifar10'     # 'mnist'
nettype = 'fc'          # 'tinyfc', 'conv'
batch_size = 2          # 4
epoch_num = 60
optimtype = 'sgd'       # 'adam'
learning_rate = 1e-4    # 5e-3
lr_str = '1e-4'         # '5e-3'
load_trained = True    # False
w_save_fn = f"{dataset}_{nettype}net_b{batch_size}_{epoch_num}epochs_{optimtype}_lr{lr_str}_weights.pkl"
# w_load_fn = f"{dataset}_{nettype}net_b{batch_size}_{epoch_num}epochs_{optimtype}_lr{lr_str}_weights.pkl"
w_load_fn = f"weights_ddn_{dataset}_{nettype}net_b{batch_size}_{epoch_num}epochs_{optimtype}_lr{lr_str}.pkl"
print(f"bs: {batch_size}, lr: {learning_rate}")

class MnistTinyFCNet(nn.Module):
    def __init__(self):
        super(MnistTinyFCNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1*28*28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.flatten(x)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class MnistFCNet(nn.Module):
    def __init__(self):
        super(MnistFCNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1*28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 10)

        # He uniform for w, 0 for bias
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc5.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc6.weight, nonlinearity='relu')

        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)
        init.constant_(self.fc3.bias, 0)
        init.constant_(self.fc4.bias, 0)
        init.constant_(self.fc5.bias, 0)
        init.constant_(self.fc6.bias, 0)
    
    def forward(self, x):
        out = self.flatten(x)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.fc6(out)
        return out

class MnistConvNet(nn.Module):
    def __init__(self):
        super(MnistConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 0)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), 2, 0)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64*6*6, 1024)
        self.fc2 = nn.Linear(1024, 10)

        # He uniform for w, 0 for bias
        init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

        init.constant_(self.conv1.bias, 0)
        init.constant_(self.conv2.bias, 0)
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.flat(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class Cifar10FCNet(nn.Module):
    def __init__(self):
        super(Cifar10FCNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 10)

        # He uniform for w, 0 for bias
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')

        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)
        init.constant_(self.fc3.bias, 0)
        init.constant_(self.fc4.bias, 0)
    
    def forward(self, x):
        out = self.flatten(x)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class Cifar10ConvNet(nn.Module):
    def __init__(self):
        super(Cifar10ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (5, 5), 2, 0)
        self.conv2 = nn.Conv2d(64, 128, (5, 5), 2, 0)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), 1, 0)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256*3*3, 1024)
        self.fc2 = nn.Linear(1024, 10)

        # He uniform for w, 0 for bias
        init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

        init.constant_(self.conv1.bias, 0)
        init.constant_(self.conv2.bias, 0)
        init.constant_(self.conv3.bias, 0)
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.flat(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

if dataset == 'mnist':
    if nettype == 'tinyfc':
        MyNet = MnistTinyFCNet
    elif nettype == 'fc':
        MyNet = MnistFCNet
    elif nettype == 'conv':
        MyNet = MnistConvNet
    else:
        raise NotImplementedError(f'Unknown nettype: {nettype}')
elif dataset == 'cifar10':
    if nettype == 'fc':
        MyNet = Cifar10FCNet
    elif nettype == 'conv':
        MyNet = Cifar10ConvNet
    else:
        raise NotImplementedError(f'Unknown nettype: {nettype}')
else:
    raise NotImplementedError(f'Unknown dataset: {dataset}')

if dataset == 'mnist':
    ###-----MNIST-----###
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, batch_size, 1, 28, 28))
    y_train = y_train.reshape((-1, batch_size))
    x_test = x_test.reshape((-1, batch_size, 1, 28, 28))
    y_test = y_test.reshape((-1, batch_size))

    x_train = x_train / 255
    x_test = x_test / 255

    # x_train = (x_train - 0.131) / 0.308
    # x_train = (x_train + 1) / 2
    # x_test = (x_test - 0.131) / 0.308
    # x_test = (x_test + 1) / 2
    ###-----END OF MNIST-----###
elif dataset == 'cifar10':
    ###-----CIFAR10-----###
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if x_train.shape == (50000, 32, 32, 3):
        x_train = x_train.transpose((0, 3, 1, 2))   # channel first
        x_test = x_test.transpose((0, 3, 1, 2))
    x_train = x_train.reshape((-1, batch_size, 3, 32, 32))
    y_train = y_train.reshape((-1, batch_size))
    x_test = x_test.reshape((-1, batch_size, 3, 32, 32))
    y_test = y_test.reshape((-1, batch_size))

    x_train = x_train / 255
    x_test = x_test / 255

    # x_train = (x_train - np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))) / np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    # x_train = (x_train + 1) / 2
    # x_test = (x_test - np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))) / np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    # x_test = (x_test + 1) / 2
    ###-----END OF CIFAR10-----###
else:
    raise NotImplementedError(f'Unknown dataset: {dataset}')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MyNet().to(device)

loss_fc = nn.CrossEntropyLoss()
if optimtype == 'sgd':
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif optimtype == 'adam':
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    raise NotImplementedError(f'Unknown optimizer: {optimtype}')

# # same initialization
# for name, parameter in model.named_parameters():
#     if 'weight' in name:
#         parameter.data = torch.tensor(parameter.detach().cpu().numpy() * np.sqrt(6)).detach().to(device)
#     elif 'bias' in name:
#         parameter.data = torch.tensor(parameter.detach().cpu().numpy() * 0).detach().to(device)
#     else:
#         raise NotImplementedError

class AverageMeter(object):
    # computes and stores the average and current value
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    # computes the accuracy over the k top predictions for the specified values of k
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if not load_trained:
    for epoch in range(epoch_num):
        losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Acc@1', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(zip(x_train, y_train)):
            inputs, targets = torch.tensor(inputs).to(device), torch.tensor(targets).to(device)
            outputs = model(inputs)
            loss = loss_fc(outputs, targets)

            # measure accuracy and record loss
            acc1 = accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            train_acc.update(acc1[0].item(), inputs.size(0))

            optim.zero_grad()
            loss.backward()
            optim.step()
            if batch_idx % 10000 == 0:
                print(f"epoch [{epoch + 1}/{epoch_num}], step [{batch_idx + 1}/{len(x_train)}], "
                    f"outputs max: {outputs.max().item():.4f}, min: {outputs.min().item():.4f}, loss: {loss.item():.4f}")
        
        with torch.no_grad():
            test_acc = AverageMeter('Acc@1', ':6.2f')
            for batch_idx, (inputs, targets) in enumerate(zip(x_test, y_test)):
                inputs, targets = torch.tensor(inputs).to(device), torch.tensor(targets).to(device)
                outputs = model(inputs)
                
                # measure accuracy and record loss
                acc1 = accuracy(outputs, targets)
                test_acc.update(acc1[0].item(), inputs.size(0))
            
            print(f"train acc {train_acc.avg:.4f}% loss {losses.avg:.4f}, test acc {test_acc.avg:.4f}%")
    
    params = {}
    for name, parameter in model.named_parameters():
        print(name, ':', parameter.size())
        params[name] = parameter.detach().cpu().numpy()

    with open(w_save_fn, 'wb') as f:
        pickle.dump(params, f)
    
else:
    with open(w_load_fn, 'rb') as f:
        params = pickle.load(f)
        for name, parameter in model.named_parameters():
            parameter.data = torch.tensor(params[name]).detach().to(device)
    
    with torch.no_grad():
        test_acc = AverageMeter('Acc@1', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(zip(x_test, y_test)):
            inputs, targets = torch.tensor(inputs).to(device), torch.tensor(targets).to(device)
            outputs = model(inputs)
            
            # measure accuracy and record loss
            acc1 = accuracy(outputs, targets)
            test_acc.update(acc1[0].item(), inputs.size(0))
        
        print(f"test acc {test_acc.avg:.4f}%")

###-----adversarial attack-----###
###-----white box attack-----###
# import foolbox
# def attack_model(model, bounds, x_test, y_test, epsilons):
#     fmodel = foolbox.PyTorchModel(model, bounds)
#     clean_acc = foolbox.accuracy(fmodel, x_test, y_test)
#     print(f"clean acc:{clean_acc:f}")

#     attack = foolbox.attacks.PGD()
#     raw_advs, clipped_advs, success = attack(fmodel, x_test, y_test, epsilons=epsilons)
#     robust_acc = 1 - success.detach().cpu().numpy().mean(axis=-1)
#     for eps, acc in zip(epsilons, robust_acc):
#         print(f"eps:{eps:f} acc:{acc:f}")

#     return clipped_advs

# epsilons = [0.02 * (i + 1) for i in range(10)]
# model.eval()
# adv_img_list = attack_model(model, (0, 1), torch.from_numpy(x_test.reshape((-1, 1, 28, 28))).to(device), torch.from_numpy(y_test.reshape((-1,))).to(device), epsilons)
###-----END OF white box attack-----###

###-----transfer attack-----###
adv_imgs_path = f"E:/baai_backup/core_modularization/adv_imgs_v2/{dataset}"
source_filename_list = os.listdir(adv_imgs_path)
source_filename_list.sort()
with torch.no_grad():
    for source_filename in source_filename_list:
        source_file = np.load(os.path.join(adv_imgs_path, source_filename))
        x_test, y_test = source_file['x_test'], source_file['y_test']
        x_test = x_test.transpose((0, 3, 1, 2))     # channel first
        if dataset == 'mnist':
            x_test = x_test.reshape((-1, batch_size, 1, 28, 28))
        elif dataset == 'cifar10':
            x_test = x_test.reshape((-1, batch_size, 3, 32, 32))
        else:
            raise NotImplementedError(f'Unknown dataset: {dataset}')
        y_test = y_test.reshape((-1, batch_size))
        x_test = x_test.astype('float32')
        y_test = y_test.astype('int64')

        test_acc = AverageMeter('Acc@1', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(zip(x_test, y_test)):
            inputs, targets = torch.tensor(inputs).to(device), torch.tensor(targets).to(device)
            outputs = model(inputs)
            
            # measure accuracy and record loss
            acc1 = accuracy(outputs, targets)
            test_acc.update(acc1[0].item(), inputs.size(0))
        
        print(f"{source_filename.split('_')[0]} acc {test_acc.avg:.4f}%")
###-----END OF transfer attack-----###
###-----END OF adversarial attack-----###
