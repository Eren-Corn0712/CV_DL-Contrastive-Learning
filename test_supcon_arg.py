import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from networks.res_cct import SupConResNet, LinearClassifier
from util import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns


def parse_option():
    parser = argparse.ArgumentParser('Testing')

    # dataset setting
    parser.add_argument('--dataset_path', type=str, default='tumor_data/test',
                        help='Please use custom dataset path! We use ImageFolder to Read dataset')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')

    # model loading
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--backbone_ckpt', type=str,
                        default='save_2021-12-20/tumor_models/SupCon_tumor_resnet50_lr_0.0005_decay_0.0005_bsz_64_temp_0.1_Equ_Adj_trial_0/last.pth',
                        help='path to pre-trained model')
    parser.add_argument('--classifier_ckpt', type=str,
                        default='save_2021-12-21/cls_tumor_models/tumor_resnet50_lr_0.1_decay_0_bsz_64_3_ft/ckpt_cls_last.pth',
                        help='path to pre-trained classifier')

    opt = parser.parse_args()

    return opt


def set_transform():
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.2981, 0.2981, 0.2981),
            std=(0.1455, 0.1455, 0.1455)
        )
    ])

    return transform


def set_dataset(opt):
    dataset = datasets.ImageFolder(root=opt.dataset_path,
                                   transform=set_transform(),
                                   target_transform=None)

    return dataset


def set_loader(opt, dataset):
    data_loader = DataLoader(dataset,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             sampler=None,
                             pin_memory=True,
                             num_workers=4)

    return data_loader


def set_all_model(opt):
    model = SupConResNet(name=opt.model)
    classifier = LinearClassifier(name=opt.model, num_classes=2)  # default two classes

    # load weight
    model_checkpoint = torch.load(opt.backbone_ckpt)
    model.load_state_dict(state_dict=model_checkpoint['model'])

    cla_checkpoint = torch.load(opt.classifier_ckpt)
    classifier.load_state_dict(state_dict=cla_checkpoint['model'])

    # model to cuda
    model.cuda()
    classifier.cuda()

    return model, classifier


def test(model, classifier, data_loader):
    model.eval()
    classifier.eval()

    y_prediction = []
    y_label = []

    with torch.no_grad():
        for (images, label) in tqdm(data_loader):
            images, label = images.cuda(), label.cuda()

            outputs = model.encoder(images)
            outputs = classifier(outputs)

            _, predictions = torch.max(outputs, 1)

            y_label.extend(label.view(-1).detach().cpu().numpy())
            y_prediction.extend(predictions.view(-1).detach().cpu().numpy())

    return y_label, y_prediction


def show_confusion_matrix(y_label, y_prediction, class_names):
    cf_matrix = confusion_matrix(y_true=y_label,
                                 y_pred=y_prediction, )

    def_cm = pd.DataFrame(data=cf_matrix,
                          index=class_names,
                          columns=class_names)
    per_cls_acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1)
    print(class_names)
    print(per_cls_acc)  # 顯示每個class的Accuracy

    print("Plot confusion matrix")
    plt.figure()
    sns.heatmap(def_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig("confusion_matrix.png")


def classification_analysis(y_true, y_pred):
    print("Accuracy_score:{:.4f}".format(accuracy_score(y_true, y_pred)))
    print("precision_score:{:.4f}".format(precision_score(y_true, y_pred)))
    print("recall_score:{:.4f}".format(recall_score(y_true, y_pred)))
    print(f"f1_score:{f1_score(y_true, y_pred)}")


def main():
    opt = parse_option()

    dataset = set_dataset(opt)

    data_loader = set_loader(opt, dataset)

    model, classifier = set_all_model(opt)

    y_label, y_prediction = test(model, classifier, data_loader)

    class_names = [name for name in dataset.classes]

    show_confusion_matrix(y_label, y_prediction, class_names)

    classification_analysis(y_label, y_prediction)


if __name__ == '__main__':
    main()
