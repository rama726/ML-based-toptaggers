import torch
from torch import nn, optim
import argparse, json, time
import utils
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import datetime
import random

from types import SimpleNamespace

from torch.utils.data import Dataset
import logging
from torch.utils.data.distributed import DistributedSampler

import h5py, glob
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Top tagging')
parser.add_argument('--exp_name', type=str, default='', metavar='N',
                    help='experiment_name')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help = 'test best model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',   
                    help='input batch size for training')
parser.add_argument('--num_train', type=int, default=-1, metavar='N',
                    help='number of training samples')
parser.add_argument('--epochs', type=int, default=35, metavar='N',
                    help='number of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='number of warm-up epochs')
parser.add_argument('--c_weight', type=float, default=5e-3, metavar='N',
                    help='weight of x model')                 
parser.add_argument('--seed', type=int, default=99, metavar='N',
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--val_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before validation')
parser.add_argument('--datadir', type=str, default='./data', metavar='N',
                    help='data dir')
parser.add_argument('--logdir', type=str, default='./logs', metavar='N',
                    help='folder to output logs')
parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                    help='dropout probability')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--num_block',nargs='+', type=int, default=[3,3,3], metavar='N',
                    help='dropout probability')
parser.add_argument('--hidden',nargs='+', type=int, default=[16,32,64], metavar='N',
                    help='dim of latent space')
parser.add_argument('--n_layers', type=int, default=6, metavar='N',
                    help='number of LGEBs')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--weight_decay', type=float, default=1e-2, metavar='N',
                    help='weight decay')

parser.add_argument('--local_rank', type=int, default=os.environ['LOCAL_RANK'])

class JetDataset(Dataset):
    """
    PyTorch dataset.
    """
    def __init__(self, data, num_pts=-1, shuffle=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['image'])
        else:
            if num_pts > len(data['image']):
                logging.warn('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['image'])))
                self.num_pts = len(data['image'])
            else:
                self.num_pts = num_pts
        if shuffle:
            self.perm = torch.randperm(len(data['image']))[:self.num_pts]
        else:
            self.perm = None


    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}

class ResNetBlock(nn.Module):

    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in
            
        # Network representing F
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        
        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out
        
class PreActResNetBlock(nn.Module):

    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in
            
        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        )
        
        # 1x1 convolution can apply non-linearity as well, but not strictly necessary
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False)
        ) if subsample else None

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out           
resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock
}  

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU
}  

class ResNet(nn.Module):

    def __init__(self, num_classes=2, num_blocks=[3,3,3], c_hidden=[16,32,64], act_fn_name="relu", block_name="ResNetBlock", **kwargs):
        """
        Inputs: 
            num_classes - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        super().__init__()
        assert block_name in resnet_blocks_by_name
        self.hparams = SimpleNamespace(num_classes=num_classes, 
                                       c_hidden=c_hidden, 
                                       num_blocks=num_blocks, 
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name],
                                       block_class=resnet_blocks_by_name[block_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        
        # A first convolution on the original image to scale up the channel size
        if self.hparams.block_class == PreActResNetBlock: # => Don't apply non-linearity on output
            self.input_net = nn.Sequential(
                nn.Conv2d(2, c_hidden[0], kernel_size=3, padding=1, bias=False)
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(2, c_hidden[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_hidden[0]),
                self.hparams.act_fn()
            )
        
        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
                blocks.append(
                    self.hparams.block_class(c_in=c_hidden[block_idx if not subsample else (block_idx-1)],
                                             act_fn=self.hparams.act_fn,
                                             subsample=subsample,
                                             c_out=c_hidden[block_idx])
                )
        self.blocks = nn.Sequential(*blocks)
        
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x




def run(epoch, loader, partition):
    if partition == 'train':
        train_sampler.set_epoch(epoch)
        model.train()
    else:
        model.eval()

    res = {'time':0, 'correct':0, 'loss': 0, 'counter': 0, 'acc': 0,
           'loss_arr':[], 'correct_arr':[],'label':[],'score':[]}

    tik = time.time()
    loader_length = len(loader)

    for i, data in enumerate(loader):
        if partition == 'train':
            optimizer.zero_grad()

        batch_size,_,_,_=data['image'].size()
        imgs = data["image"].to(device)
        label = data["is_signal"].to(device, dtype).long()
        pred = model(imgs)
                
        predict = pred.max(1).indices
        correct = torch.sum(predict == label).item()
        loss = loss_fn(pred, label)
        
        if partition == 'train':
            loss.backward()
            optimizer.step()
        elif partition == 'test': 
            # save labels and probilities for ROC / AUC
            score = torch.nn.functional.softmax(pred, dim = -1)
            res['label'].append(label)
            res['score'].append(score)

        res['time'] = time.time() - tik
        res['correct'] += correct
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        res['correct_arr'].append(correct)

        if i != 0 and i % args.log_interval == 0:
            running_loss = sum(res['loss_arr'][-args.log_interval:])/len(res['loss_arr'][-args.log_interval:])
            running_acc = sum(res['correct_arr'][-args.log_interval:])/(len(res['correct_arr'][-args.log_interval:])*batch_size)
            avg_time = res['time']/res['counter'] * batch_size
            tmp_counter = utils.sum_reduce(res['counter'], device = device)
            tmp_loss = utils.sum_reduce(res['loss'], device = device) / tmp_counter
            tmp_acc = utils.sum_reduce(res['correct'], device = device) / tmp_counter
            if (args.local_rank == 0):
                print(">> %s \t Epoch %d/%d \t Batch %d/%d \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                     (partition, epoch + 1, args.epochs, i, loader_length, running_loss, running_acc, tmp_acc, avg_time))

    torch.cuda.empty_cache()
    # ---------- reduce -----------
    if partition == 'test':
        res['label'] = torch.cat(res['label']).unsqueeze(-1)
        res['score'] = torch.cat(res['score'])
        res['score'] = torch.cat((res['label'],res['score']),dim=-1)
    res['counter'] = utils.sum_reduce(res['counter'], device = device).item()
    res['loss'] = utils.sum_reduce(res['loss'], device = device).item() / res['counter']
    res['acc'] = utils.sum_reduce(res['correct'], device = device).item() / res['counter']
    return res

def train(res):
    ### training and validation
    for epoch in range(0, args.epochs):
        train_res = run(epoch, dataloaders['train'], partition='train')
        print("Time: train: %.2f \t Train loss %.4f \t Train acc: %.4f" % (train_res['time'],train_res['loss'],train_res['acc']))
        if epoch % args.val_interval == 0:
            if (args.local_rank == 0):
                torch.save(model.state_dict(), f"{args.logdir}/{args.exp_name}/checkpoint-epoch-{epoch}.pt")
            dist.barrier() # wait master to save model
            with torch.no_grad():
                val_res = run(epoch, dataloaders['valid'], partition='valid')
            if (args.local_rank == 0): # only master process save
                res['lr'].append(optimizer.param_groups[0]['lr'])
                res['train_time'].append(train_res['time'])
                res['val_time'].append(val_res['time'])
                res['train_loss'].append(train_res['loss'])
                res['train_acc'].append(train_res['acc'])
                res['val_loss'].append(val_res['loss'])
                res['val_acc'].append(val_res['acc'])
                res['epochs'].append(epoch)

                ## save best model
                if val_res['acc'] > res['best_val']:
                    print("New best validation model, saving...")
                    torch.save(model.state_dict(), f"{args.logdir}/{args.exp_name}/best-val-model.pt")
                    res['best_val'] = val_res['acc']
                    res['best_epoch'] = epoch

                print("Epoch %d/%d finished." % (epoch, args.epochs))
                print("Train time: %.2f \t Val time %.2f" % (train_res['time'], val_res['time']))
                print("Train loss %.4f \t Train acc: %.4f" % (train_res['loss'], train_res['acc']))
                print("Val loss: %.4f \t Val acc: %.4f" % (val_res['loss'], val_res['acc']))
                print("Best val acc: %.4f at epoch %d." % (res['best_val'],  res['best_epoch']))

                json_object = json.dumps(res, indent=4)
                with open(f"{args.logdir}/{args.exp_name}/train-result.json", "w") as outfile:
                    outfile.write(json_object)

        ## adjust learning rate
        if epoch < 4:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.5
        elif epoch > 3 and epoch < 30:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.9
        else :
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.1
        dist.barrier()

def test(res):
    ### test on best model
    best_model = torch.load(f"{args.logdir}/{args.exp_name}/best-val-model.pt", map_location=device)
    model.load_state_dict(best_model)
    with torch.no_grad():
        test_res = run(0, dataloaders['test'], partition='test')

    pred = [torch.zeros_like(test_res['score']) for _ in range(dist.get_world_size())]
    dist.all_gather(pred, test_res['score'] )
    pred = torch.cat(pred).cpu()

    if (args.local_rank == 0):
        np.save(f"{args.logdir}/{args.exp_name}/score.npy",pred)
        fpr, tpr, thres, eB, eS  = utils.buildROC(pred[...,0], pred[...,2])
        auc = utils.roc_auc_score(pred[...,0], pred[...,2])

        metric = {'test_loss': test_res['loss'], 'test_acc': test_res['acc'],
                  'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1]}
        res.update(metric)
        print("Test: Loss %.4f \t Acc %.4f \t AUC: %.4f \t 1/eB 0.3: %.4f \t 1/eB 0.5: %.4f"
               % (test_res['loss'], test_res['acc'], auc, 1./eB[0], 1./eB[1]))
        json_object = json.dumps(res, indent=4)
        with open(f"{args.logdir}/{args.exp_name}/test-result.json", "w") as outfile:
            outfile.write(json_object)

if __name__ == "__main__":
    ### initialize args
    args = parser.parse_args()
    utils.args_init(args)

    ### set random seed
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    ### initialize cuda
    dist.init_process_group("gloo")
    device = torch.device("cuda:{}".format(args.local_rank))
    dtype = torch.float32
    splits = ['train', 'test', 'valid'] 
    patterns = {'train':'train', 'test':'test', 'valid':'val'} 
    
    files = glob.glob(args.datadir + '/*.h5')
    datafiles = {split:[] for split in splits}
    for file in files:
        for split,pattern in patterns.items():
            if pattern in file: datafiles[split].append(file)
    nfiles = {split:len(datafiles[split]) for split in splits}
    
    num_pts={'train':-1,'test':-1,'valid':-1}
    
    num_pts_per_file = {}
    for split in splits:
        num_pts_per_file[split] = []
        
        if num_pts[split] == -1:
            for n in range(nfiles[split]): num_pts_per_file[split].append(-1)
        else:
            for n in range(nfiles[split]): num_pts_per_file[split].append(int(np.ceil(num_pts[split]/nfiles[split])))
            num_pts_per_file[split][-1] = int(np.maximum(num_pts[split] - np.sum(np.array(num_pts_per_file[split])[0:-1]),0))
    
    
    datasets = {}
    for split in splits:
        datasets[split] = []
        for file in datafiles[split]:
            with h5py.File(file,'r') as f:
                datasets[split].append({key: torch.from_numpy(val[:]) for key, val in f.items()})

    torch_datasets = {split: ConcatDataset([JetDataset(data, num_pts=num_pts_per_file[split][idx]) for idx, data in enumerate(datasets[split])]) for split in splits}

    train_sampler = DistributedSampler(torch_datasets['train'])                                    
    dataloaders = {split: DataLoader(dataset,
                                 batch_size=args.batch_size, # prevent CUDA memory exceeded
                                 sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False),
                                 pin_memory=True,
                                 persistent_workers=True,
                                 drop_last= True if (split == 'train') else False,
                                 num_workers=args.num_workers)
                                 for split, dataset in torch_datasets.items()}        

    ### create parallel model
    model2 = ResNet(num_classes=2, num_blocks=args.num_block, c_hidden=args.hidden, act_fn_name="relu", block_name="ResNetBlock")

    model2 = model2.to(device)
    model = DistributedDataParallel(model2, device_ids=[args.local_rank])

    ### print model and data information
    if (args.local_rank == 0):
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Network Size:", pytorch_total_params)
        for (split, dataloader) in dataloaders.items():
            print(f" {split} samples: {len(dataloader.dataset)}")

    ### optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    ### loss function
    loss_fn = nn.CrossEntropyLoss()

    ### initialize logs
    res = {'epochs': [], 'lr' : [],
           'train_time': [], 'val_time': [],  'train_loss': [], 'val_loss': [],
           'train_acc': [], 'val_acc': [], 'best_val': 0, 'best_epoch': 0}

    if not args.test_mode:
        ### training and testing
        train(res)
        test(res)
    else:
        ### only test on best model
        test(res)
