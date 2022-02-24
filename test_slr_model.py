import argparse
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from model import CLRBackbone, CLRClassifier, CLRLinearClassifier
from aug_util import Resize_Pad
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

bk_ckpt = 'save_2022-02-14/tumor_models/SupCon_tumor_implicit_resnet18_m_m_a_a_2_3_4_5_lr_0.5_decay_0.0005_bsz_512_temp_0.1_trial_5_warm/last.pth'
cl_ckpt = 'save_2022-02-14/cls_tumor_models/SupCon_tumor_implicit_resnet18_m_m_a_a_2_3_4_5_lr_0.1_decay_0_bsz_512_5_cosine/ckpt_cls_0.83.pth'


def parse_option():
    parser = argparse.ArgumentParser('Testing')

    # dataset setting
    parser.add_argument('--dataset_path', type=str, default='tumor_data/test',
                        help='Please use custom dataset path! We use ImageFolder to Read dataset')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch_size')
    # model loading
    parser.add_argument('--model', type=str, default='implicit_resnet18_m_m_a_a_2_3_4_5')
    parser.add_argument('--head', type=str, default='mlp_bn')
    parser.add_argument('--feat_dim', type=int, default=256)
    parser.add_argument('--backbone_ckpt',
                        type=str,
                        default=bk_ckpt,
                        help='path to pre-trained model')
    parser.add_argument('--classifier_ckpt',
                        type=str,
                        default=cl_ckpt,
                        help='path to pre-trained classifier')
    parser.add_argument('--n_cls', type=int, default=2)
    parser.add_argument('--classifier', type=str, default='ML')
    opt = parser.parse_args()
    return opt


def set_transform():
    transform = transforms.Compose([
        Resize_Pad(224),
        transforms.ToTensor(),
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
                             num_workers=os.cpu_count())
    return data_loader


def set_all_model(opt):
    model = CLRBackbone(name=opt.model, head=opt.head, feat_dim=opt.feat_dim)
    if opt.classifier == 'SL':
        classifier = CLRLinearClassifier(name=opt.model, num_classes=opt.n_cls)
    elif opt.classifier == 'ML':
        classifier = CLRClassifier(name=opt.model, num_classes=opt.n_cls)

    # load weight
    model_checkpoint = torch.load(opt.backbone_ckpt)
    model.load_state_dict(state_dict=model_checkpoint['model'], strict=False)

    cla_checkpoint = torch.load(opt.classifier_ckpt)
    classifier.load_state_dict(state_dict=cla_checkpoint['model'], strict=False)

    # model to cuda
    if torch.cuda.is_available():
        model.cuda()
        classifier.cuda()

    return model, classifier


def test(model, classifier, data_loader):
    model.eval()
    classifier.eval()

    y_prediction = []
    y_label = []

    # initial numpy array
    representation_list = []
    with torch.no_grad():
        for (images, label) in tqdm(data_loader):
            if torch.cuda.is_available():
                images, label = images.cuda(), label.cuda()

            # Vector
            representation = model.encoder(images)
            representation_list.extend(representation.cpu().numpy())

            outputs = classifier(representation)
            predictions = torch.argmax(outputs, 1)
            y_label.extend(label.view(-1).detach().cpu().numpy())
            y_prediction.extend(predictions.view(-1).detach().cpu().numpy())

    return y_label, y_prediction, representation_list


def show_confusion_matrix(y_label, y_prediction, class_names):
    report = classification_report(y_true=y_label,
                                   y_pred=y_prediction,
                                   target_names=class_names)
    print(report)

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
    plt.show()


def main():
    opt = parse_option()

    dataset = set_dataset(opt)

    data_loader = set_loader(opt, dataset)

    model, classifier = set_all_model(opt)

    y_label, y_prediction, representation_vector = test(model, classifier, data_loader)

    class_names = [name for name in dataset.classes]

    show_confusion_matrix(y_label, y_prediction, class_names)


if __name__ == '__main__':
    main()
