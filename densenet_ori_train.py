import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List

#added
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from load_utils import load_state_dict_from_url
from cub_voc import CUB_VOC
import os
from tqdm import tqdm
import shutil
import numpy as np
from newPad2d import newPad2d
#from torch.autograd import Variable


MEMORY_EFFICIENT = True
IS_TRAIN = 0        # 0/1
IS_MULTI = 0        # 0/1
LAYERS = '121'
DATANAME = 'bird' # bird/cat/.../cub/helen/voc_multi
NUM_CLASSES =6 if IS_MULTI else 2
cub_file = '/data/sw/dataset/frac_dataset'
voc_file = '/data/sw/dataset/VOCdevkit/VOC2010/voc2010_crop'
log_path = '/data/fjq/iccnn/densenet/' # for model
save_path = '/data/fjq/iccnn/basic_fmap/densenet/'  # for get_feature
acc_path = '/data/fjq/iccnn/basic_fmap/densenet/acc/'

dataset = '%s_densenet_%s_ori' % (LAYERS, DATANAME)
log_path = log_path + dataset + '/'
pretrain_model = log_path + 'model_2000.pth'
BATCHSIZE = 1
LR = 0.00001
EPOCH = 1000

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=MEMORY_EFFICIENT):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=0, #new padding
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
        self.pad2d_1 = newPad2d(1) #nn.ReplicationPad2d(1)#new padding

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.pad2d_1(self.relu2(self.norm2(bottleneck_output))))#new padding
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=MEMORY_EFFICIENT):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=MEMORY_EFFICIENT):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=0, bias=False)), # new padding
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)), # new padding
        ]))

        self.pad2d_1 = newPad2d(1)#nn.ZeroPad2d(1) #new padding
        self.pad2d_3 = newPad2d(3)#nn.ZeroPad2d(3) #new padding


        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.features):
            if i == 0:
                x = self.pad2d_3(x) # new padding
            if i == 3:
                x = self.pad2d_1(x) # new padding
            x = layer(x)
        out = F.relu(x, inplace=True)
        f_map = out.detach()    # get_feature
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out, f_map #out

def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        # print(key)
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    pretrained_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)


def _densenet(arch, growth_rate, block_config, num_init_features, num_class, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_class, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    else:
        if pretrain_model is not None:
            device = torch.device("cuda")
            model = nn.DataParallel(model).to(device)
            model.load_state_dict(torch.load(pretrain_model))
        else:
            print('Error: pretrain_model == None')
    return model


def densenet121(num_class, pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, num_class, pretrained, progress,
                     **kwargs)


def densenet161(num_class, pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, num_class, pretrained, progress,
                     **kwargs)


def densenet169(num_class, pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, num_class, pretrained, progress,
                     **kwargs)


def densenet201(num_class, pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, num_class, pretrained, progress,
                     **kwargs)

def get_Data(is_train, dataset_name, batch_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    voc_helen = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'helen', 'voc_multi']
    ##cub dataset###
    label = None if is_train else 0
    if not is_train:
        batch_size = 1
    if dataset_name == 'cub':
        trainset = CUB_VOC(cub_file, dataset_name, 'ori', train=True, transform=transform, is_frac=label)
        testset = CUB_VOC(cub_file, dataset_name, 'ori', train=False, transform=val_transform, is_frac=label)
    ###cropped voc dataset###
    elif dataset_name in voc_helen:
        trainset = CUB_VOC(voc_file, dataset_name, 'ori', train=True, transform=transform, is_frac=label)
        testset = CUB_VOC(voc_file, dataset_name, 'ori', train=False, transform=val_transform, is_frac=label)
    ###celeb dataset###
    #elif dataset_name == 'celeb':
    #    trainset = Celeb(training = True, transform=None)
    #    testset = Celeb(training = False, transform=None)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def net_train():
    trainset_loader, testset_loader = get_Data(IS_TRAIN, DATANAME, BATCHSIZE)
    if os.path.exists(log_path):
        shutil.rmtree(log_path);os.makedirs(log_path)
    else:
        os.makedirs(log_path)
    device = torch.device("cuda")
    net = None
    if LAYERS == '121':
        net = densenet121(num_class=NUM_CLASSES, pretrained=True)
    if LAYERS == '161':
        net = densenet161(num_class=NUM_CLASSES, pretrained=True)
    net = nn.DataParallel(net).to(device)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.module.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.6)

    # Train the model
    best_acc = 0.0; save_loss = []; test_loss = []; train_acc = []; test_acc = [];
    for epoch in range(EPOCH+1):
        scheduler.step()
        net.train()
        total_loss = 0.0; correct = .0; total = .0;
        for batch_step, input_data in tqdm(enumerate(trainset_loader,0),total=len(trainset_loader),smoothing=0.9):
            inputs, labels = input_data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output, _ = net(inputs)
            #print(output)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum()
            total += labels.size(0)
            loss = criterion(output, labels)
            #print(module.features.conv0.weight)
            loss.backward()
            #if batch_step>0:
            #    return
            #for name, parms in net.named_parameters():
            #   print('after* name:', name, 'grad_value:',parms.grad)
            optimizer.step()
            total_loss += loss.item()
        total_loss = float(total_loss) / (batch_step+1)
        correct = float(correct) / total
        testacc, testloss = test(net, testset_loader)
        save_loss.append(total_loss); train_acc.append(correct);
        test_loss.append(testloss); test_acc.append(testacc);
        np.savez(log_path+'loss.npz', train_loss=np.array(save_loss), test_loss=np.array(test_loss),\
                                    train_acc=np.array(train_acc), test_acc=np.array(test_acc))
        print('Epoch', epoch, 'train loss: %.4f' % total_loss, 'train accuracy:%.4f' % correct, \
                'test loss: %.4f' % testloss, 'test accuracy:%.4f' % testacc)
        if epoch % 50 == 0:
            torch.save(net.state_dict(), log_path+'model_%.3d.pth' % epoch)
        if epoch % 1 == 0:
            if testacc > best_acc:
                best_acc = testacc
                torch.save(net.state_dict(), log_path+'model_%.3d_%.4f.pth' % (epoch, best_acc))
    print('Finished Training')
    return net

def get_feature():
    print('pretrain_model:', pretrain_model)
    _, testset_test = get_Data(True, DATANAME, BATCHSIZE)
    _, testset_feature = get_Data(False, DATANAME, BATCHSIZE)
    device = torch.device("cuda")
    net = None
    if LAYERS == '121':
        net = densenet121(num_class=NUM_CLASSES, pretrained=False)
    if LAYERS == '161':
        net = densenet161(num_class=NUM_CLASSES, pretrained=False)
    net = nn.DataParallel(net).to(device)
    # Test the model
    acc, _ = test(net, testset_test)
    f = open(acc_path+dataset+'_test.txt', 'w+')
    f.write('%s\n' % dataset)
    f.write('acc:%f\n' % acc)
    print('test acc:', acc)
    all_feature = []
    testset = testset_test if DATANAME == 'voc_multi' else testset_feature
    for batch_step, input_data in tqdm(enumerate(testset,0),total=len(testset),smoothing=0.9):
        inputs, labels = input_data
        inputs, labels = inputs.to(device), labels.to(device)
        net.eval()
        output, f_map = net(inputs)
        all_feature.append(f_map.cpu().numpy())
    all_feature = np.concatenate(all_feature,axis=0)
    f.write('sample num:%d' % (all_feature.shape[0]))
    f.close()
    print(all_feature.shape)
    np.savez_compressed(save_path+LAYERS+'_densenet_'+DATANAME+'_ori.npz', f_map=all_feature[...])
    print('Finished Operation!')
    return net


def test(net, testdata):
    criterion = nn.CrossEntropyLoss()
    correct, total = .0, .0
    total_loss = .0
    for batch_step, input_data in tqdm(enumerate(testdata,0),total=len(testdata),smoothing=0.9):
        inputs, labels = input_data
        inputs, labels = inputs.cuda(), labels.cuda()
        net.eval()
        outputs, _ = net(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    total_loss = float(total_loss)/(batch_step+1)
    return float(correct)/total, total_loss

def densenet_ori_train():
    if IS_TRAIN == 1:
        net = net_train()
    elif IS_TRAIN == 0:
        net = get_feature()
