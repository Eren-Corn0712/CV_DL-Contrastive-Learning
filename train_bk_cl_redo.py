import os
import argparse
import math
import datetime
import pandas as pd
import torch.utils.data
import torch
import wandb
import wandb.sklearn

import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import warnings

from torchvision.datasets import ImageFolder

from util import AverageMeter, ContrastiveTransformations
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, same_seeds
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from aug_util import show_augmentation_image, test_weight_sampler
from aug_util import Erosion, Resize_Pad, GaussianBlur
from pytorch_metric_learning.losses import SupConLoss, NTXentLoss
from torch.utils.data import WeightedRandomSampler
from model import CLRBackbone, CLRLinearClassifier, CLRClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

train_path = 'tumor_data/train'
test_path = 'tumor_data/test'

exp_time = datetime.datetime.today().date()
project = f'CLR_Project_implicit_resnet18'
num_workers = int(os.cpu_count() / 4)
warnings.filterwarnings("ignore")


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


class SimCLRFTransform:
    def __init__(self, opt, eval_transform: bool = False, ) -> None:
        self.normalize = transforms.Normalize(mean=opt.mean, std=opt.std)
        s = 1.0
        self.color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.resize = transforms.Resize(size=(opt.size, opt.size))
        self.random_crop_resize = transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.))
        self.erosion = Erosion(p=0.5)
        if not eval_transform:
            data_transforms = [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomChoice([
                    transforms.RandomEqualize(p=0.5),
                    transforms.RandomAutocontrast(p=0.5),
                    transforms.RandomAdjustSharpness(2, p=0.5),
                ]),
                transforms.RandomGrayscale(p=0.2),
                Resize_Pad(opt.size)
            ]
        else:
            data_transforms = [
                # transforms.RandomHorizontalFlip(p=0.5),
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
            final_transform = transforms.Compose([transforms.ToTensor(), ])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


def parse_option_bk():
    parser = argparse.ArgumentParser('argument for training')
    # Training Hyper-parameter
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=num_workers,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='250,300,350',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='implicit_resnet18')
    parser.add_argument('--head', type=str, default='mlp_bn', choices=['mlp', 'mlp_bn', '2mlp_bn'])
    parser.add_argument('--feat_dim', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='tumor',
                        choices=['tumor', 'path'], help='dataset')

    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
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

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=num_workers,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='implicit_resnet18')
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
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    parser.add_argument('--gaussian_blur', type=bool, default=False,
                        help='Gaussian_blur for DataAugmentation')

    parser.add_argument('--classifier', type=str, default='ML',
                        choices=['ML', 'SL'])
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
        os.makedirs(opt.save_folder)

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
    test_transform = SimCLRFTransform(opt, eval_transform=False)

    opt.train_transform = list(train_transform.transform.transforms)
    opt.test_transform = list(test_transform.transform.transforms)

    if opt.dataset == 'tumor':
        opt.train_path = train_path
        opt.test_path = test_path
        train_dataset = ImageFolder(opt.train_path,
                                    ContrastiveTransformations(train_transform))
        test_dataset = ImageFolder(opt.test_path,
                                   ContrastiveTransformations(test_transform))
    else:
        raise ValueError(opt.dataset)

    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        sampler=sampler,
        pin_memory=True, )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True, )

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

    train_transform = SimCLRFTransform(opt, eval_transform=False)
    test_transform = SimCLRFTransform(opt, eval_transform=True)

    opt.train_transform = list(train_transform.transform.transforms)
    opt.test_transform = list(test_transform.transform.transforms)

    if opt.dataset == 'tumor':
        opt.train_path = train_path
        opt.test_path = test_path
        train_dataset = ImageFolder(opt.train_path,
                                    train_transform)
        test_dataset = ImageFolder(opt.test_path,
                                   test_transform)

    else:
        raise ValueError(opt.dataset)

    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
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

    return train_dataset, train_loader, test_dataset, test_loader


def set_folder_linear(opt):
    opt.model_path = './save_{}/cls_{}_models'.format(exp_time, opt.dataset)
    opt.wandb_path = './save_{}/cls_{}_wandb'.format(exp_time, opt.dataset)
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
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train_backbone(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    losses = AverageMeter()

    process_bar = tqdm(train_loader, total=len(train_loader), ascii=True, position=0, leave=True)
    process_bar.set_description(f'TB epoch:{epoch} {opt.model}')
    for idx, (images, labels) in enumerate(process_bar):
        # [AugmentImage1,AugmentImage2]
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)  # features size is 2B x 128

        if opt.method == 'SupCon':
            labels = torch.cat([labels, labels], dim=0).to(device=features.device)  # (B)-> (2B)
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            labels = torch.arange(bsz).to(device=features.device)
            labels = torch.cat([labels, labels], dim=0)  # (B) -> (2B)
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

    process_bar = tqdm(train_loader, total=len(train_loader), ascii=True, position=0, leave=True)
    process_bar.set_description(f'TC epoch:{epoch} {opt.model}')
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
        predictions = torch.argmax(output, 1)

        # update metric
        acc = accuracy_score(y_true=labels.view(-1).detach().cpu().numpy(),
                             y_pred=predictions.view(-1).detach().cpu().numpy(),
                             normalize=True)

        losses.update(loss.item(), bsz)
        top1.update(acc)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        process_bar.set_postfix_str(
            'loss {loss.val:.3f} ({loss.avg:.3f}) Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))

    return losses.avg, top1.avg


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

    y_pred = list()  # save predict label
    y_true = list()  # save ground truth

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
            predictions = torch.argmax(output, 1)

            # update metric
            acc = accuracy_score(y_true=labels.view(-1).detach().cpu().numpy(),
                                 y_pred=predictions.view(-1).detach().cpu().numpy(),
                                 normalize=True)
            losses.update(loss.item(), bsz)
            top1.update(acc)

            # update y_pred and y_true
            y_pred.extend(predictions.view(-1).detach().cpu().numpy())
            y_true.extend(labels.view(-1).detach().cpu().numpy())

            process_bar.set_postfix_str(
                'loss {loss.val:.3f} ({loss.avg:.3f}) Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses,
                                                                                                    top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg, y_pred, y_true


def set_model_linear(opt):
    if opt.classifier == 'SL':
        classifier = CLRLinearClassifier(name=opt.model, num_classes=opt.n_cls)
    elif opt.classifier == 'ML':
        classifier = CLRClassifier(name=opt.model, num_classes=opt.n_cls)
    else:
        raise ValueError('contrastive method not supported: {}'.
                         format(opt.classifier))
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

    show_augmentation_image(train_loader)
    test_weight_sampler(train_loader)

    # build model and linear_criterion
    backbone, backbone_criterion = set_backbone(opt_bk)

    # build optimizer
    optimizer = set_optimizer(opt_bk, backbone)

    # wandb
    your_api_key = '6ca9efcfdd0230ae14f6160f01209f0ac93aff34'
    wandb.login(key=your_api_key)
    wandb.init(dir=opt_bk.wandb_folder,
               config=vars(opt_bk),
               project=project,
               name=f'B_{opt_bk.model}_{opt_bk.method}_{opt_bk.trial}'
               )
    wandb.watch(models=backbone,
                criterion=backbone_criterion,
                log_freq=100,
                log_graph=True,
                log="all", )

    for epoch in range(1, opt_bk.epochs + 1):
        adjust_learning_rate(opt_bk, optimizer, epoch)

        # train for one epoch
        loss = train_backbone(train_loader, backbone, backbone_criterion, optimizer, epoch, opt_bk)
        val_loss = validation_backbone(test_loader, backbone, backbone_criterion, opt_bk)

        # wandb logger
        wandb.log({'train_loss': loss, 'val_loss': val_loss, 'epoch': epoch}, )
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr'], 'epoch': epoch})

    # save the last model
    save_file = os.path.join(opt_bk.save_folder, f'last.pth')

    save_model(backbone, optimizer, opt_bk, opt_bk.epochs, save_file)

    wandb.finish()

    # -------------------------------------------------------------------------
    best_acc = 0
    opt_linear = parse_option_linear()

    # create folder
    set_folder_linear(opt_linear)

    # build data loader
    train_dataset, train_loader, test_dataset, test_loader = set_loader_linear(opt_linear)

    # We need class name for plot confuse matrix
    class_names: list = train_dataset.classes

    # build model and linear_criterion
    classifier, linear_criterion = set_model_linear(opt_linear)

    # build optimizer
    optimizer = set_optimizer(opt_linear, classifier)

    # wandb
    wandb.init(dir=opt_linear.wandb_folder,
               config=vars(opt_linear),
               project=project,
               name=f'L_{opt_linear.model}_{opt_linear.method}_{opt_linear.trial}',
               )
    wandb.watch(models=classifier,
                criterion=linear_criterion,
                log_freq=100,
                log_graph=True,
                log="all")

    # training routine
    for epoch in range(1, opt_linear.epochs + 1):
        adjust_learning_rate(opt_linear, optimizer, epoch)

        train_loss, train_acc = train_linear(train_loader, backbone, classifier,
                                             linear_criterion,
                                             optimizer, epoch, opt_linear)
        # eval for one epoch
        val_loss, val_acc, y_pred, y_true = validate_linear(test_loader, backbone, classifier,
                                                            linear_criterion, opt_linear)

        wandb.log({'Train_Loss': train_loss, 'Val_Loss': val_loss, 'epoch': epoch})
        wandb.log({'Train_Acc': train_acc, 'Val_Acc': val_acc, 'epoch': epoch})
        wandb.log({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch})

        result = classification_report(y_pred=y_pred, y_true=y_true, target_names=class_names, output_dict=True)
        df = pd.DataFrame(result).transpose()

        if val_acc > best_acc:
            best_acc = val_acc
            wandb.run.summary["best_accuracy"] = best_acc
            # Create csv file to record experiment result
            save_file = os.path.join(opt_linear.save_folder, f'ckpt_cls_{best_acc:.2f}.pth')
            save_model(classifier, optimizer, opt_linear, opt_linear.epochs, save_file)
            csv_file = os.path.join(opt_linear.save_folder, f'{epoch}_{best_acc:.2f}.csv')
            df.to_csv(csv_file)
        if epoch == opt_linear.epochs:
            csv_file = os.path.join(opt_linear.save_folder, f'last.csv')
            df.to_csv(csv_file)

    print('best accuracy: {:.2f}'.format(best_acc))

    # save the last model
    save_file = os.path.join(opt_linear.save_folder, 'ckpt_cls_last.pth')
    save_model(classifier, optimizer, opt_linear, opt_linear.epochs, save_file)

    # Finish wandb
    wandb.finish()


if __name__ == '__main__':
    main()
