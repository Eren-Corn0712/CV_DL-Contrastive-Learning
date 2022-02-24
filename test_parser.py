# TODO: How to parse mutiple nested sub-commands using python argparse
# TODO: Please Try to merge two parser in CLR

import argparse


def parse_option_bk():
    parser = argparse.ArgumentParser('argument for training')
    # Training Hyper-parameter
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=5,
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


if __name__ == '__main__':
