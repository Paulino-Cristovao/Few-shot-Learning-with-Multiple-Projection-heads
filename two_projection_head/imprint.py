import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
from tqdm import tqdm_notebook as tqdm
import loader
#from DataLoader import load_torch_data
from helper_functions import*
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import gzip, pickle
import matplotlib.cm as cm
import random
import math
from tsne import bh_sne



parser = argparse.ArgumentParser(description='PyTorch Omniglot Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data', metavar='DIR', default='/home/aca10537zf/exp_thesis/layerSelection/data/CUB_200_2011',
                    help='path to dataset')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-c', '--checkpoint', default='imprint_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_checkpoint)')
parser.add_argument('-d', '--save', default='savefiles_imprint', type=str, metavar='PATH',
                    help='path to save files (default: save_files)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--num-sample', default=1, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
parser.add_argument('--aug', action='store_true', help='whether use data augmentation during training')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--latent', '--latent-size', default=256, type=int,
                    metavar='N', help='latent size (default: 256)')

best_prec1 = 0

# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    cm_avg1 = []
    cm_avg1_ = []
    cm_avg2 = []
    cm_avg2_ = []
    test_avg1 = []
    test_avg1_ = []
    test_avg2 = []
    test_avg2_ = []

    m = 200
    cm_avg1 = np.zeros((m, m))
    cm_avg1_ = np.zeros((m, m))
    cm_avg2 = np.zeros((m, m))
    cm_avg2_ = np.zeros((m, m))

    # save accuracy for later
    accuracy_save = []

    global args, best_prec1
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(str(args.latent)+'/'+args.save)
        #mkdir_p(args.save+'/'+args.checkpoint)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(str(args.latent)+'/'+args.checkpoint)

    model = models.Net(args.latent).to(device)

    # WI from fc1 before relu
    print(" ")
    print('==> Reading from model checkpoint... WI from fc1 before relu')
    print(args.model)
    print('************************************')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True


    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    novel_trasforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]) if not args.aug else transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    def novel_dataset():
        novel_dataset = loader.ImageLoader(
            args.data,
            novel_trasforms,
            train=True, num_classes=200, 
            num_train_sample=args.num_sample, 
            novel_only=True, aug=args.aug)
        return novel_dataset

    def novel_loader():
        novel_loader = torch.utils.data.DataLoader(
            novel_dataset(), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        return novel_loader

    
    def load_val():
        val_loader = torch.utils.data.DataLoader(
            loader.ImageLoader(args.data, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), num_classes=200, novel_only=args.test_novel_only),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        return val_loader

    
    #imprint(novel_loader, model, 0)

    # Data loading code # fc1 before relu
    for i in tqdm(range(10)):
        imprint(novel_loader(), model, 0)
        test_acc1, cm_test1, feature, = validate(load_val(), model, 0)
        test_avg1.append(test_acc1)
        cm_avg1 = cm_avg1 + cm_test1
    
    print(" ")
    # Accuracy- Test data [0-9]
    test_avg_acc1 = np.array(test_avg1).mean()
    test_avg_acc1 = str(round(test_avg_acc1,2))
    c_matrix1 = np.rint((cm_avg1 / 10))


    # from fc1 before relu
    feature_fc1 = feature[0].astype(np.float64)
    target_fc1 = feature[1]

    # feature
    feature_fc1 = bh_sne(feature_fc1[-5000:])
    target_fc1 = target_fc1[-5000:]

    plt.rcParams['figure.figsize'] = 20, 20
    plt.scatter(np.reshape(feature_fc1[:, 0], -1), np.reshape(feature_fc1[:, 1], -1), c=np.reshape(target_fc1,-1), cmap=plt.cm.get_cmap("jet", 200))
    plt.colorbar(ticks=range(200))
    plt.title('Testing: Latent Projection')
    plt.savefig(str(args.latent)+'/'+args.save +'/fc1.png', bbox_inches='tight')
    #plt.show()
    plt.clf()

    # plot confusion matrix
    plot_confusion_matric(c_matrix1, str(args.latent)+'/'+args.save, 'fc1_before_relu',test_avg_acc1)


    #####
    # WI from fc1 after relu
    model = models.Net(args.latent).to(device)

    print(" ")
    print('==> Reading from model checkpoint... WI from fc1 after relu')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True


    #imprint(novel_loader(), model, 1)

    # Data loading code # fc1 before relu
    for i in tqdm(range(10)):
        imprint(novel_loader(), model, 1)
        test_acc1_, cm_test1_,_ = validate(load_val(), model, 1)
        test_avg1_.append(test_acc1_)
        cm_avg1_ = cm_avg1_ + cm_test1_

    print(" ")
    # Accuracy- Test data [0-9]
    test_avg_acc1_ = np.array(test_avg1_).mean()
    test_avg_acc1_ = str(round(test_avg_acc1_))
    c_matrix1_ = np.rint((cm_avg1_ / 10))

    # plot confusion matrix
    plot_confusion_matric(c_matrix1_, str(args.latent)+'/'+args.save, 'fc1_after_relu',test_avg_acc1_)

    # From FC2
    model = models.Net(args.latent).to(device)

    print(" ")
    print('==> Reading from model checkpoint... WI before fc2 after relu')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True


    #imprint(novel_loader, model, 2)

    # Data loading code # fc1 before relu
    for i in tqdm(range(10)):
        imprint(novel_loader(), model, 2)
        test_acc2, cm_test2, feature2, = validate(load_val(), model, 2)
        test_avg2.append(test_acc2)
        cm_avg2 = cm_avg2 + cm_test2
    
    print(" ")
    # Accuracy- Test data [0-9]
    test_avg_acc2 = np.array(test_avg2).mean()
    test_avg_acc2 = str(round(test_avg_acc2,2))
    c_matrix2 = np.rint((cm_avg2 / 10))


    # from fc2 before relu
    feature_fc2 = feature2[0].astype(np.float64)
    target_fc2 = feature2[1]

    # feature
    feature_fc2 = bh_sne(feature_fc2[-5000:])
    target_fc2 = target_fc2[-5000:]

    plt.rcParams['figure.figsize'] = 20, 20
    plt.scatter(np.reshape(feature_fc2[:, 0], -1), np.reshape(feature_fc2[:, 1], -1), c=np.reshape(target_fc2,-1), cmap=plt.cm.get_cmap("jet", 200))
    plt.colorbar(ticks=range(200))
    plt.title('Testing: Latent Projection')
    plt.savefig(str(args.latent)+'/'+args.save +'/fc2.png', bbox_inches='tight')
    #plt.show()
    plt.clf()

    # plot confusion matrix
    plot_confusion_matric(c_matrix2, str(args.latent)+'/'+args.save, 'fc2_before_relu',test_avg_acc2)



    #####
    # WI from fc2 after relu
    model = models.Net(args.latent).to(device)

    print(" ")
    print('==> Reading from model checkpoint... WI from fc2 after relu')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True


    #imprint(novel_loader, model, 3)

    # Data loading code # fc1 before relu
    for i in tqdm(range(10)):
        imprint(novel_loader(), model, 3)
        test_acc2_, cm_test2_,_ = validate(load_val(), model, 3)
        test_avg2_.append(test_acc2_)
        cm_avg2_ = cm_avg2_ + cm_test2_

    print(" ")
    # Accuracy- Test data [0-9]
    test_avg_acc2_ = np.array(test_avg2_).mean()
    test_avg_acc2_ = str(round(test_avg_acc2_))
    c_matrix2_ = np.rint((cm_avg2_ / 10))

    # plot confusion matrix
    plot_confusion_matric(c_matrix2_, str(args.latent)+'/'+args.save, 'fc2_after_relu',test_avg_acc2_)


    accuracy_save.extend([args.latent,test_avg_acc1,test_avg_acc1_,test_avg_acc2,test_avg_acc2_])
    np.save(str(args.latent)+'/'+args.save+'_'+str(args.latent)+'_'+'acc.npy',accuracy_save)


    save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': test_acc2,
        }, checkpoint=str(args.latent)+'/'+args.checkpoint)


def imprint(novel_loader, model, select):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Imprinting', max=len(novel_loader))
    with torch.no_grad():
        for batch_idx, data in enumerate(novel_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = data[0].to(device)
            target = data[1].long().to(device)

            # compute output
            feature3,feature2,feature1,feature0 = model.helper_extract(input)

            if select==0:    feature = feature0 # before fc1 relu
            if select==1:    feature = feature1 # after fc1 relu
            if select==2:    feature = feature2 # before fc2 relu
            if select==3:    feature = feature3 # after fc2 relu
            
            output = F.normalize(feature,p=2,dim=feature.dim()-1,eps=1e-12)
            
            
            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=batch_idx + 1,
                        size=len(novel_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td
                        )
            bar.next()
        bar.finish()
      
    
    new_weight = torch.zeros(100, int(args.latent))

    for i in range(100):
        tmp = output_stack[target_stack == (i + 100)].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = torch.cat((model.classifier.fc.weight.data[:100], new_weight.to(device)))
    model.classifier.fc = nn.Linear(int(args.latent), 200, bias=False)
    model.classifier.fc.weight.data = weight
    


def validate(val_loader_test, model, select):
    top1 = AverageMeter()
    top5 = AverageMeter()

    m = 200
    class_correct = [0]*m
    class_total = [0]*m
    conf_matrix = np.zeros((m, m))

    latent_proj = Proj_Latent()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader_test):
            input = data[0].to(device)
            label = data[1].long().to(device)

            # compute output
            if select==0:   output,feature = model.forward_wi_fc1(input)
            if select==1:   output,feature = model.forward_wi_fc1_(input)
            if select==2:   output,feature = model.forward_wi_fc2(input)
            if select==3:   output,feature = model.forward_wi_fc2_(input)
            
            #predict --> convert output probabilities to predicted class
            pred = output.argmax(1)
            
                   
            # measure accuracy
            prec1, _ = accuracy(output, label, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))

            # compare predictions to true label
            correct_tensor = pred.eq(label.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())

            #conf_matrix = c_matrix(label, pred, correct)
            correct_tensor = pred.eq(label.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())


            for label_, pred_, correct_ in zip(label, pred, correct):
                class_correct[label_] += correct_
                class_total[label_] += 1 
                # update confusion matrix
                conf_matrix[label_][pred_] += 1

            # project latent representations
            fe = latent_proj.append(feature,label)

            
            # print progress
            if batch_idx == len(val_loader_test):
              print('({batch}/{size}) | top1: {top1: .4f} '.format(
                          batch=batch_idx + 1,
                          size=len(val_loader_test),
                          top1=top1.avg,
                          ))
              
    return top1.avg, conf_matrix, fe


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

if __name__ == '__main__':
    main()




