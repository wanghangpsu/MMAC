from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from src.resnet import ResNet18
import copy



parser = argparse.ArgumentParser(description='Reverse engineer backdoor pattern')
parser.add_argument('--model_dir', default='model0', help='model path')
parser.add_argument('--attack_dir', default='attack0', help='attack path')
#parser.add_argument('--data_path', '-d', required=True, help='data path')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NC = 10
NSTEP = 100 # maximum steps of the maximize margin optimization


class CleanNet(nn.Module):

    def __init__(self):
        super(CleanNet, self).__init__()
        model = ResNet18()
        model = model.to(device)
        model.load_state_dict(torch.load('./'+args.model_dir+'/model.pth'))
        model.eval()
        self.model = model
        self.mask_0 = torch.ones([64, 1, 1]).to(device) + 5
        self.mask_1_0 = torch.ones([64, 1, 1]).to(device)+ 5
        self.mask_1_1 = torch.ones([64, 1, 1]).to(device)+ 5
        self.mask_1_2 = torch.ones([64, 1, 1]).to(device)+ 5
        self.mask_1_3 = torch.ones([64, 1, 1]).to(device)+ 5

        self.mask_2_0 = torch.ones([128, 1, 1]).to(device) + 5
        self.mask_2_1 = torch.ones([128, 1, 1]).to(device) + 5
        self.mask_2_2 = torch.ones([128, 1, 1]).to(device) + 5
        self.mask_2_3 = torch.ones([128, 1, 1]).to(device) +  5

        self.mask_3_0 = torch.ones([256, 1, 1]).to(device) + 5
        self.mask_3_1 = torch.ones([256, 1, 1]).to(device)+ 5
        self.mask_3_2 = torch.ones([256, 1, 1]).to(device)+ 5
        self.mask_3_3 = torch.ones([256, 1, 1]).to(device)+ 5

        self.mask_4_0 = torch.ones([512, 1, 1]).to(device)+ 5
        self.mask_4_1 = torch.ones([512, 1, 1]).to(device)+ 5
        self.mask_4_2 = torch.ones([512, 1, 1]).to(device)+ 5
        self.mask_4_3 = torch.ones([512, 1, 1]).to(device)+ 5
        self.ml1 = [self.mask_1_0, self.mask_1_1, self.mask_1_2, self.mask_1_3]
        self.ml2 = [self.mask_2_0, self.mask_2_1, self.mask_2_2, self.mask_2_3]
        self.ml3 = [self.mask_3_0, self.mask_3_1, self.mask_3_2, self.mask_3_3]
        self.ml4 = [self.mask_4_0, self.mask_4_1, self.mask_4_2, self.mask_4_3]

    def layer_forward(self, layer_name, x, mask_list):
        out = F.relu(layer_name[0].bn1(layer_name[0].conv1(x)))
        out = torch.min(mask_list[0] , out)
        out = layer_name[0].bn2(layer_name[0].conv2(out))
        out += layer_name[0].shortcut(x)
        out = F.relu(out)
        out = torch.min(mask_list[1],  out)
        x = out
        out = F.relu(layer_name[1].bn1(layer_name[1].conv1(out)))
        out = torch.min(mask_list[2], out)
        out = layer_name[1].bn2(layer_name[1].conv2(out))
        out += layer_name[1].shortcut(x)
        out = F.relu(out)
        out = torch.min(mask_list[3], out)
        return out

    def forward(self, x):
        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = torch.min(out, self.mask_0)
        out = self.layer_forward(self.model.layer1, out, self.ml1)
        out = self.layer_forward(self.model.layer2, out, self.ml2)
        out = self.layer_forward(self.model.layer3, out, self.ml3)
        out = self.layer_forward(self.model.layer4, out, self.ml4)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.model.linear(out)
        return out
model = CleanNet()
mask_dict = {"mask0": model.mask_0,
             "mask10": model.mask_1_0,
             "mask11": model.mask_1_1,
             "mask12": model.mask_1_2,
             "mask13": model.mask_1_3,
             "mask20": model.mask_2_0,
             "mask21": model.mask_2_1,
             "mask22": model.mask_2_2,
             "mask23": model.mask_2_3,
             "mask30": model.mask_3_0,
             "mask31": model.mask_3_1,
             "mask32": model.mask_3_2,
             "mask33": model.mask_3_3,
             "mask40": model.mask_4_0,
             "mask41": model.mask_4_1,
             "mask42": model.mask_4_2,
             "mask43": model.mask_4_3
             }

for key in mask_dict:
    mask_dict[key].requires_grad = True
network = model
network.to(device)
network = torch.load('./'+args.model_dir + '/repaired_model.pth')


transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
all_ind = []
for s in range(10):
    ind = [i for i, label in enumerate(testset.targets) if label == s]
    all_ind += ind[50:]
testset.data = testset.data[all_ind]
testset.targets = list(np.array(testset.targets)[all_ind])


testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True, num_workers=2)




trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
all_ind = []
for s in range(10):
    ind = [i for i, label in enumerate(trainset.targets) if label == s]
    all_ind += ind[:50]
trainset = torch.utils.data.Subset(trainset, all_ind)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=200, shuffle=True, num_workers=2)

class Defense_Net(nn.Module):
    def __init__(self, clean_net, threshold):
        super(Defense_Net, self).__init__()
        self.network = clean_net
        self.threshold = threshold
    def forward(self, x):
        o1 = self.network(x)
        o2 = self.network.model(x)
        _, pred = o2.max(1)
        onehot_label = F.one_hot(pred, num_classes=NC)
        margin1 = torch.sum((o1 * onehot_label), dim=1) \
                  - torch.max((1 - onehot_label) * o1 - 1000 * onehot_label, dim=1)[0]
        margin2 = torch.sum((o2 * onehot_label), dim=1) \
                  - torch.max((1 - onehot_label) * o2 - 1000 * onehot_label, dim=1)[0]
        diff = margin2 - margin1
        o2[diff > self.threshold] = o1[diff > self.threshold]
        #print(o2.size(), diff.size())
        final_out = o2 - 1000 * onehot_label * (diff > self.threshold).float().unsqueeze(1)
        return final_out



# learn the threshold
all_diff = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        o1 = network(inputs)
        o2 = network.model(inputs)
        _, pred = o2.max(1)
        onehot_label = F.one_hot(pred, num_classes=NC)

        margin1 = torch.sum((o1 * onehot_label), dim=1) \
                  - torch.max((1 - onehot_label) * o1 - 1000 * onehot_label, dim=1)[0]

        margin2 = torch.sum((o2 * onehot_label), dim=1) \
                  - torch.max((1 - onehot_label) * o2 - 1000 * onehot_label, dim=1)[0]
        all_diff.append(margin2 - margin1)

all_diff = torch.cat(all_diff, dim=0)

mean = torch.mean(all_diff)
std = torch.std(all_diff)

# p = 0.05 z = 1.645, p = 0.005, z = 2.576

print(mean + 1.645 * std)
print(mean + 2.576 * std)
threshold = mean + 2.576 * std


network = Defense_Net(network, threshold)



testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

all_ind = []
for s in range(10):
    ind = [i for i, label in enumerate(testset.targets) if label == s]
    all_ind += ind[50:]
testset.data = testset.data[all_ind]
testset.targets = list(np.array(testset.targets)[all_ind])


testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = network(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

acc = 100.*correct/total
print('Test ACC: %.3f' % acc)




test_attacks = torch.load('./'+args.attack_dir + '/test_attacks')
patterns = torch.load('./'+args.attack_dir + '/pattern')
test_images_attacks = test_attacks['image']
test_labels_attacks = test_attacks['label']

testset_attacks = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)
attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=100, shuffle=False, num_workers=2)

# Evaluate attack success rate
correct = 0
total = 0
corr = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(attackloader):
        inputs, targets = inputs.to(device).float(), targets.to(device)
        outputs = network(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #corr += predicted.eq(targets+1).sum().item()

acc = 100.*correct/total
print('Attack success rate: %.3f' % acc)

# evaluate pacc


print(args.model_dir)


def mask_craft(pattern):
    mask = (pattern > 0.0).float()

    return mask


def embed_backdoor(image, pattern, mask):
    image = image * (1 - mask) + pattern * mask
    image.clamp(0, 1)

    return image


mask = mask_craft(patterns)
correct = 0
total = 0
target_class = test_labels_attacks[0].item()
ind_train = torch.load('./' + args.attack_dir + '/ind_train')
print(ind_train[0])
f = open('./' + args.attack_dir + '/attack_info.txt', "r")

source_class = int(f.readlines()[0].split(" ")[2][:-1])
print(source_class)
f.close()
ind_test = [i for i, label in enumerate(testset.targets) if label!=target_class]
testset.data = testset.data[ind_test]
testset.targets = list(np.array(testset.targets)[ind_test])


with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = embed_backdoor(inputs, patterns.unsqueeze(0), mask.unsqueeze(0)).float()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = network(inputs)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

acc = 100.*correct/total
print('poisoned ACC: %.3f' % acc)
