import argparse
import logging
import time
from torch.utils.data import DataLoader
from lion import Lion
import pingouin as pg
import seaborn as sns
import math
from torch import optim
from UNet import Unet, resnet34_unet, resnet101_unet,MTUnet,AAUnet
from attention_unet import AttU_Net
from channel_unet import myChannelUnet
from loss import *
from r2unet import R2U_Net
from segnet import SegNet
from unetpp import NestedUNet
from fcn import get_fcn8s
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot
from cenet_ocr import *
from cenet import CE_Net_,MTCE_Net
from cenetppp import *
from lib.loss.loss_contrast import *
from edge_loss import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from unet.UNet3Plus import UNet3Plus
from seg_model import deeplab,highresnet,miniseg,pspnet,efficientseg
import cv2

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="test")
    parse.add_argument("--epoch", type=int, default=100)
    parse.add_argument("--testepoch", type=int, default=54)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_unet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument('--dataset', default='mydataset',  # dsb2018_256
                       help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")

    args = parse.parse_args()
    return args


def getLog(args):
    dirname = os.path.join(args.log_dir, args.arch, str(args.batch_size), str(args.dataset), str(args.epoch))
    filename = dirname + '/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logging


def uncertainty_estimate(output, task):
    """
    使用贝叶斯估计方法计算输出结果的不确定性估计。

    Args:
        output: 模型输出的结果，shape 为 [batch_size, num_classes] 或 [batch_size, num_classes, height, width]，根据不同任务可能有不同尺寸。
        task: 当前任务的名称，可以是 "classification" 或 "segmentation"。

    Returns:
        uncertainty: 输出结果的不确定性估计，shape 为 [batch_size]。
    """
    with torch.no_grad():
        if task == "classification":
            # 对分类任务的输出进行 softmax 转换
            softmax_output = torch.softmax(output, dim=-1)
            max_probs, _ = torch.max(softmax_output, dim=-1)
            uncertainty = 1 - max_probs
        elif task == "segmentation":
            # 对分割任务的输出取平均值，并进行 softmax 转换
            softmax_output = torch.softmax(output, dim=1)
            mean_probs = torch.mean(softmax_output, dim=(1,2))
            max_probs, _ = torch.max(mean_probs, dim=-1)
            uncertainty = 1 - max_probs

    return uncertainty

def getModel(args):
    if args.arch == 'UNet':
        model = Unet(3, 2).to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(2, pretrained=False).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = False
        model = NestedUNet(args, 3, 2).to(device)
    return model


def getDataset(args):
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    if args.dataset == "mydataset":
        if args.arch=='cenet_sup':
            train_dataset = fzjDataset(r"train", transform=TwoCropTransform(x_transforms,ori_transforms), target_transform=y_transforms)
        else:
            train_dataset=fzjDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset,num_workers=8, batch_size=args.batch_size, shuffle=True)
        val_dataset = fzjDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = fzjDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders, val_dataloaders, test_dataloaders


def get_one_hot(label, N):
    label=label.type(torch.int)
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N).to(device)
    ones = ones.index_select(0, label)
    size.append(N)
    ones = ones.view(*size)
    ones = ones.transpose(2, 3)
    ones = ones.transpose(1, 2)
    return ones


def test(val_dataloaders, save_predict=False):

    logging.info('final test........')
    if save_predict == True:
        dir = os.path.join(r'/data2/save_predict/', str(args.arch), str(args.batch_size), str(args.testepoch),
                           str(args.dataset))
        dir2 = os.path.join(r'/data2/save_predict2/', str(args.arch), str(args.batch_size), str(args.testepoch),
                           str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(
         r'/data/save_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(
             args.testepoch) + '.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    # plt.ion() #开启动态模式
    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        dist_total = 0
        pixel_dist_error_total = 0
        spacings_dic={}
        true_label_list = []
        pred_list=[]
        pred_label_list = []
        pred_measure_list = []
        final_pred_list = []
        final_class1_list = []
        pred_pro1_list= []
        diff_list=[]
        ori_list=[]
        num = len(val_dataloaders)  # 验证集图片的总数
        with open("/data/data4/spacing.txt", 'r') as f:
            spacings = f.readlines()
            for i in range(len(spacings)):
                spacings_dic[spacings[i].split('|')[0]] = float(
                    spacings[i].split(' ')[1].strip('\n').strip('(').strip(','))
        for pic, y_hat, pic_path, mask_path, border, ma, dist,wl,prop,reg_label,ca,class_label,pix in val_dataloaders:
            pic = pic.to(device)
            start = time.time()
            predict = model(pic)
            end = time.time()
            print('time:', (end - start) )
            
            if isinstance(predict, tuple):
                predict=predict[0].argmax(dim=1)
            else:
                predict = predict.argmax(dim=1)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            predict = cv2.convertScaleAbs(predict)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(predict, connectivity=8)
            # for i in range(1, num_labels):
            #     if stats[i][4] < 400:
            #         labels[labels == i] = 0

            # 连通域个数大于两个（除背景外）时将连通域面积最大的两个之外的区域抑制
            areas = [stats[i][4] for i in range(1, num_labels)]
            sorted_areas = sorted(areas, reverse=True)
            if len(sorted_areas) > 2:
                threshold = sorted_areas[1]
                for i in range(1, num_labels):
                    if stats[i][4] < threshold:
                        labels[labels == i] = 0
            if len(sorted_areas) > 1:
                labels[labels > 1] = 1

            # labels[labels == 1] = 255
            predict = np.float32(labels)
            fold = mask_path[0].split('/')[-2]
            pix_size = pix/prop
            iou, diff,pixel_dist_error= get_iou(mask_path[0], predict, ma, dist,pix_size)
            diff_list.append(float(reg_label-diff))
            ori_list.append(float(reg_label))
            miou_total += iou  # 获取当前预测图的miou，并加到总miou中
            hd=get_hd(mask_path[0], predict, ma)
            hd_total += hd
            dice = get_dice(mask_path[0], predict, ma)
            print('dice:',dice)
            dice_total += dice
            dist_total += abs(diff)
            pixel_dist_error_total += abs(pixel_dist_error)
            with open('/data2/unet.txt', 'a') as f1:
                f1.write(mask_path[0] + ' ' + str(iou)+ ' '+str(dice)+ ' '+str(hd)+ ' '+str(diff)+ ' '+str(pixel_dist_error) + '\n')
            box=(int(border[0][0]),int(border[1][0]),int(border[2][0]),int(border[3][0]))
            # 将边缘画成红色
            edges1 = cv2.cvtColor(predict, cv2.COLOR_GRAY2BGR)
            edges1[np.where((edges1 == [255, 255, 255]).all(axis=2))] = [0, 255, 255]
            # 将边缘与原始图像合并
            result = cv2.addWeighted(img_ori, 1, edges1, 1, 0, dtype=cv2.CV_32F)
            img_ori = cv2.resize(result.astype('float32'), (int(border[2] - border[0]), int(border[3] - border[1])))
            cv2.imwrite(dir2 + '/' + mask_path[0].split('/')[-2] + '_' + mask_path[0].split('/')[-1],img_ori)
            if i < num: i += 1  # 处理验证集下一张图
        print('Miou=%f,aver_hd=%f,aver_dice=%f,aver_pixel_dist_error=%f,,aver_dist_error=%f mm ' % (
            miou_total / num, hd_total / num, dice_total / num, aver_pixel_dist_error, dist_total / num))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f,aver_pixel_dist_error=%f,,aver_dist_error=%f mm' % (
            miou_total / num, hd_total / num, dice_total / num, aver_pixel_dist_error, dist_total / num))



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    gpus=[1,2]
    x_transforms = transforms.Compose([
        # transforms.Resize([512,512]),

        transforms.ToTensor(),  # -> [0,1]
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])
    ori_transforms = transforms.Compose([
        # transforms.Resize([512,512]),

        transforms.ToTensor(),  # -> [0,1]
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])
    # mask只需要转换为tensor
    y_transforms = transforms.Compose([
        # transforms.Resize([512,512]),
        transforms.ToTensor()])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size, args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
                 (args.arch, args.epoch, args.batch_size, args.dataset))
    print('**************************')
    model = getModel(args)
    model=model.to(device)
    train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)
    criterion2 = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters())
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)
