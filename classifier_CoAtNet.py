import torch
import argparse
from tqdm import tqdm
from model.coatnet import coatnet_0


# TODO: Train Code and Test Code
# TODO: You need to think what is need in setting parse
def parse_option():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')

    # optimizer
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # wandb seeting
    return args


def train(args, epoch, model, train_loader, optimizer, criterion):
    model.train()
    progress_bar = tqdm(train_loader)
    progress_bar.set_description(f'{epoch}')
    for idx, (images, labels) in enumerate(progress_bar):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        batch_loss = criterion(outputs, labels)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        progress_bar.set_postfix_str(f'loss:{batch_loss}')

        if args.dry_run:
            break


def test(model, test_loader, ):
    pass

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    unpickle()
    args = parse_option()
