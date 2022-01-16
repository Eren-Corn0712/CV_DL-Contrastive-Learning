import os
import argparse
import math
import datetime

import torch.utils.data
import torch
import wandb
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import warnings

from torchvision.datasets import ImageFolder

from util import TwoTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, same_seeds
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aug_util import Resize_Pad, GaussianBlur
from pytorch_metric_learning.losses import SupConLoss, NTXentLoss

from model import CLRBackbone, CLRLinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

train_path_list = ['tumor_data/train']
test_path_list = ['tumor_data/test']

exp_time = datetime.datetime.today().date()
project = f'CLR_{exp_time}'
num_workers = int(os.cpu_count() / 4)
warnings.filterwarnings("ignore")


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred, normalize=True)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, precision, recall, f1


class SimCLRFTransform:
    def __init__(self, opt, eval_transform: bool = False, ) -> None:
        self.normalize = transforms.Normalize(mean=opt.mean, std=opt.std)

        if not eval_transform:
            data_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomChoice([
                    transforms.RandomEqualize(p=0.5),
                    transforms.RandomAutocontrast(p=0.5),
                    transforms.RandomAdjustSharpness(2, p=0.5),
                ]),
                Resize_Pad(opt.size)
            ]
        else:
            data_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                Resize_Pad(opt.size)
            ]

        if opt.gaussian_blur:
            kernel_size = int(0.1 * opt.size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))

        if self.normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose([transforms.ToTensor(), self.normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


def parse_option_bk():
    parser = argparse.ArgumentParser('argument for training')
    # Training Hyper-parameter
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=num_workers,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='400,450,500',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--head', type=str, default='mlp', choices=['mlp', 'mlp_bn', '2mlp_bn'])
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='tumor',
                        choices=['tumor', 'path'], help='dataset')

    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', type=bool, default=True,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--gaussian_blur', type=bool, default=True,
                        help='Gaussian_blur for DataAugmentation')

    opt = parser.parse_args()

    return opt


def parse_option_linear():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=num_workers,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='160,175,190',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='tumor',
                        choices=['tumor', 'path'], help='dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for Resize')
    # other setting
    parser.add_argument('--cosine', type=bool, default=True,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    # load pre-train model
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    parser.add_argument('--gaussian_blur', type=bool, default=False,
                        help='Gaussian_blur for DataAugmentation')

    opt = parser.parse_args()

    return opt


def set_folder_bk(opt):
    opt.model_path = './save_{}/{}_models'.format(exp_time, opt.dataset)
    opt.wandb_path = './save_{}/{}_wandb'.format(exp_time, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    # warm-up for large-batch training,
    if opt.batch_size > 32:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # If the pathname refers to an existing directory
    opt.wandb_folder = os.path.join(opt.wandb_path, opt.model_name)
    if not os.path.isdir(opt.wandb_folder):
        os.makedirs(opt.wandb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(name=opt.save_folder)

    return opt


def set_loader_bk(opt):
    # construct data loader
    if opt.dataset == 'tumor':
        mean = (0.1771, 0.1771, 0.1771)
        std = (0.1842, 0.1842, 0.1842)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.mean = mean
    opt.std = std

    train_transform = SimCLRFTransform(opt, eval_transform=False)
    test_transform = SimCLRFTransform(opt, eval_transform=True)

    opt.train_transform = list(train_transform.transform.transforms)
    opt.test_transform = list(test_transform.transform.transforms)

    if opt.dataset == 'tumor':
        opt.train_path_list = train_path_list
        opt.test_path_list = test_path_list
        train_dataset = torch.utils.data.ConcatDataset([
            ImageFolder(root,
                        TwoTransform(train_transform))
            for root in opt.train_path_list])
        test_dataset = torch.utils.data.ConcatDataset([
            ImageFolder(root,
                        TwoTransform(test_transform))
            for root in opt.test_path_list])
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def set_loader_linear(opt):
    # construct data loader
    if opt.dataset == 'tumor':
        mean = (0.1771, 0.1771, 0.1771)
        std = (0.1842, 0.1842, 0.1842)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.mean = mean
    opt.std = std

    train_transform = SimCLRFTransform(opt, eval_transform=True)
    test_transform = SimCLRFTransform(opt, eval_transform=True)

    opt.train_transform = list(train_transform.transform.transforms)
    opt.test_transform = list(test_transform.transform.transforms)

    if opt.dataset == 'tumor':
        opt.train_path_list = train_path_list
        opt.test_path_list = test_path_list
        train_dataset = torch.utils.data.ConcatDataset([
            ImageFolder(root,
                        train_transform)
            for root in opt.train_path_list])
        test_dataset = torch.utils.data.ConcatDataset([
            ImageFolder(root,
                        test_transform)
            for root in opt.test_path_list])

    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def set_folder_linear(opt):
    opt.model_path = './save_{}/cls_{}_models'.format(exp_time, opt.dataset)
    opt.wandb_path = './save_{}/cls_{}_wandb_path'.format(exp_time, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'tumor':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.wandb_folder = os.path.join(opt.wandb_path, opt.model_name)
    if not os.path.isdir(opt.wandb_folder):
        os.makedirs(opt.wandb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_backbone(opt):
    model = CLRBackbone(name=opt.model, head=opt.head, feat_dim=opt.feat_dim)
    if opt.method == 'SupCon':
        criterion = SupConLoss(temperature=opt.temp)
    elif opt.method == 'SimCLR':
        criterion = NTXentLoss(temperature=opt.temp)
    else:
        raise ValueError('contrastive method not supported: {}'.
                         format(opt.method))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train_backbone(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    losses = AverageMeter()

    process_bar = tqdm(train_loader, total=len(train_loader), ascii=True, position=0, leave=True)
    process_bar.set_description(f'TB epoch:{epoch}:{opt.model}')
    for idx, (images, labels) in enumerate(process_bar):

        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)  # (2B x 128)

        if opt.method == 'SupCon':
            labels = torch.cat([labels, labels], dim=0)  # (B,)-> (2B,)
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            labels = torch.arange(bsz)
            labels = torch.cat([labels, labels], dim=0)
            loss = criterion(features, labels)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        process_bar.set_postfix_str(
            'loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

    return losses.avg


def train_linear(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

    process_bar = tqdm(train_loader, total=len(train_loader), ascii=True, position=0, leave=True)
    process_bar.set_description(f'TC epoch:{epoch}:{opt.model}')
    for idx, (images, labels) in enumerate(process_bar):

        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)
        # update metric

        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(output, 1)

        labels = labels.view(-1).detach().cpu().numpy()
        predictions = predictions.view(-1).detach().cpu().numpy()
        acc, precision, recall, f1 = compute_metrics(y_true=labels, y_pred=predictions)

        top1.update(acc, n=1)
        precision_meter.update(precision, n=1)
        recall_meter.update(recall, n=1)
        f1_meter.update(f1, n=1)

        process_bar.set_postfix_str(
            'loss {loss.val:.3f} ({loss.avg:.3f}) Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))

    return losses.avg, top1.avg, precision_meter.avg, recall_meter.avg, f1_meter.avg


def validation_backbone(val_loader, model, criterion, opt):
    """one epoch training"""
    model.eval()

    losses = AverageMeter()

    with torch.no_grad():
        process_bar = tqdm(val_loader, total=len(val_loader), ascii=True, position=0, leave=True)
        process_bar.set_description(f'VB:{opt.model}')
        for idx, (images, labels) in enumerate(process_bar):
            images = torch.cat([images[0], images[1]], dim=0)

            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            bsz = labels.shape[0]

            # compute loss
            features = model(images)  # 2B x 128

            if opt.method == 'SupCon':
                labels = torch.cat([labels, labels], dim=0)  # (B,)-> (2B,)
                loss = criterion(features, labels)
            elif opt.method == 'SimCLR':
                labels = torch.arange(bsz)
                labels = torch.cat([labels, labels], dim=0)
                loss = criterion(features, labels)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

            # update metric
            losses.update(loss.item(), bsz)

            process_bar.set_postfix_str(
                'loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

    return losses.avg


def validate_linear(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

    with torch.no_grad():
        process_bar = tqdm(val_loader, total=len(val_loader), ascii=True, position=0, leave=True)
        process_bar.set_description(f'VC:{opt.model}')
        for idx, (images, labels) in enumerate(process_bar):
            images = images.float().cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward and freeze backbone network
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            _, predictions = torch.max(output, 1)

            labels = labels.view(-1).detach().cpu().numpy()
            predictions = predictions.view(-1).detach().cpu().numpy()
            acc, precision, recall, f1 = compute_metrics(y_true=labels, y_pred=predictions)

            top1.update(acc, n=1)
            precision_meter.update(precision, n=1)
            recall_meter.update(recall, n=1)
            f1_meter.update(f1, n=1)

            process_bar.set_postfix_str(
                'loss {loss.val:.3f} ({loss.avg:.3f}) '
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                'prec@ {prec.val:.3f} ({prec.avg:.3f}) '
                'recall@ {recall.val:.3f} ({recall.avg:.3f}) '
                'f1@ {f1.val:.3f} ({f1.avg:.3f})'.format(
                    loss=losses,
                    top1=top1,
                    prec=precision_meter,
                    recall=recall_meter,
                    f1=f1_meter))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg, precision_meter.avg, recall_meter.avg, f1_meter.avg


def set_model_linear(opt):
    classifier = CLRLinearClassifier(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = classifier.cuda()
    criterion = criterion.cuda()

    return classifier, criterion


def main():
    opt_bk = parse_option_bk()
    # random seed
    same_seeds(seed=2022)

    # create folder
    set_folder_bk(opt_bk)

    # build data loader
    train_loader, test_loader = set_loader_bk(opt_bk)

    # check_aug_data(train_loader)
    # build model and linear_criterion
    backbone, backbone_criterion = set_backbone(opt_bk)

    # build optimizer
    optimizer = set_optimizer(opt_bk, backbone)

    # wandb
    your_api_key = '6ca9efcfdd0230ae14f6160f01209f0ac93aff34'
    wandb.login(key=your_api_key)
    wandb_log = wandb.init(dir=opt_bk.wandb_folder,
                           config=vars(opt_bk),
                           project=project,
                           name=f'B_{opt_bk.model}_{opt_bk.method}_{opt_bk.trial}'
                           )
    wandb_log.watch(models=backbone,
                    criterion=backbone_criterion,
                    log_freq=100,
                    log_graph=True)

    for epoch in range(1, opt_bk.epochs + 1):
        adjust_learning_rate(opt_bk, optimizer, epoch)

        # train for one epoch
        loss = train_backbone(train_loader, backbone, backbone_criterion, optimizer, epoch, opt_bk)
        val_loss = validation_backbone(test_loader, backbone, backbone_criterion, opt_bk)

        # wandb logger
        wandb_log.log({'train_loss': loss, 'val_loss': val_loss, 'epoch': epoch}, )
        wandb_log.log({'learning_rate': optimizer.param_groups[0]['lr'], 'epoch': epoch})

    # save the last model
    save_file = os.path.join(opt_bk.save_folder, f'last.pth')

    save_model(backbone, optimizer, opt_bk, opt_bk.epochs, save_file)

    wandb_log.finish()

    # -------------------------------------------------------------------------
    best_acc = 0
    opt_linear = parse_option_linear()

    # create folder
    set_folder_linear(opt_linear)

    # build data loader
    train_loader, test_loader = set_loader_linear(opt_linear)

    # build model and linear_criterion
    classifier, linear_criterion = set_model_linear(opt_linear)

    # build optimizer
    optimizer = set_optimizer(opt_linear, classifier)

    # wandb
    wandb_log = wandb.init(dir=opt_linear.wandb_folder,
                           config=vars(opt_linear),
                           project=project,
                           name=f'L_{opt_linear.model}_{opt_linear.method}_{opt_linear.trial}',
                           )
    wandb_log.watch(models=classifier,
                    criterion=linear_criterion,
                    log_freq=100,
                    log_graph=False)

    # training routine
    for epoch in range(1, opt_linear.epochs + 1):
        adjust_learning_rate(opt_linear, optimizer, epoch)

        train_loss, train_acc, train_prec, train_recall, train_f1 = train_linear(train_loader, backbone, classifier,
                                                                                 linear_criterion,
                                                                                 optimizer, epoch, opt_linear)
        # eval for one epoch
        val_loss, val_acc, val_prec, val_recall, val_f1 = validate_linear(test_loader, backbone, classifier,
                                                                          linear_criterion, opt_linear)

        wandb_log.log({'Train_loss': train_loss, 'Val_loss': val_loss, 'epoch': epoch})
        wandb_log.log({'Train_acc': train_acc, 'Val_acc': val_acc, 'epoch': epoch})
        wandb_log.log({'Train_prec': train_prec, 'Val_prec': val_prec, 'epoch': epoch})
        wandb_log.log({'Train_recall': train_recall, 'Val_recall': val_recall, 'epoch': epoch})
        wandb_log.log({'Train_f1': train_f1, 'Val_f1': val_f1, 'epoch': epoch})
        wandb_log.log({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch})

        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))
    # save the last model
    save_file = os.path.join(opt_linear.save_folder, 'ckpt_cls_last.pth')
    save_model(classifier, optimizer, opt_linear, opt_linear.epochs, save_file)

    # Finish wandb
    wandb_log.finish()


if __name__ == '__main__':
    main()
