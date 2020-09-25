import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as trns
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.autograd import Variable
from dataloader import RetinopathyDataset
from torch.utils.data import DataLoader
from models import ResNet
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix
import itertools



# use cuda or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BatchSize = 4

# create train/val transforms
train_transform = trns.Compose([
                      trns.RandomHorizontalFlip(p=0.4),
                      trns.RandomVerticalFlip(),
                      trns.ToTensor(),
                      trns.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                  ])
val_transform = trns.Compose([
                    trns.ToTensor(),
                    trns.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
                ])

# create train/val datasets
trainset = RetinopathyDataset(root='data/',
                              mode='train',
                              transform=train_transform)
valset = RetinopathyDataset(root='data/',
                            mode='test',
                            transform=val_transform)

# create train/val loaders
train_loader = DataLoader(dataset=trainset,
                          batch_size=BatchSize,
                          shuffle=True,
                          num_workers=0)
val_loader = DataLoader(dataset=valset,
                        batch_size=BatchSize,
                        shuffle=False,
                        num_workers=0)

'''
Testing function
args : 
    model: the network you built, e.g., EEGNet, DeepConvNet
    data_loader: propose (batch_size, (input, label)) data
'''
def evaluate(model, data_loader):
    model.eval()

    count = 0
    correct = 0
    for batch_idx, (X,Y) in enumerate(data_loader):
        inputs = X.to(device)
        labels = Y.to(device)
        # wrap them in Variable
        inputs = Variable(inputs.float())
        labels = Variable(labels.long())
        with torch.no_grad():
            output = model(inputs)
        pred = torch.max(output, 1)[1]
        correct += pred.eq(labels.data).cpu().sum()
        count += 1

    results = correct.cpu().numpy() * 100.0 / len(data_loader.dataset)

    return results


'''
Training function
args : 
    model: the network you built, e.g., EEGNet, DeepConvNet 
    optimizer: the optimizer you set, e.g., Adam, SGD.
    criterion: the loss function you set, e.g., CrossEntropyLoss
    model_save_path:  save file name string before "best.plk"
'''
def train(model, epochs, optimizer, scheduler, criterion, model_save_path):

    epoch_hist = []
    acc_train_hist = []
    acc_test_hist = []

    test_acc_max = 0.0
    if not os.path.exists('backup'):
        os.mkdir('backup')

    # training
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        for batch_idx, (X, Y) in enumerate(train_loader):
            inputs = X.to(device)
            labels = Y.to(device)
            # wrap them in Variable
            inputs, labels = Variable(inputs.float()), Variable(labels.long())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # scheduler.step()
        # Verification
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, val_loader)
        print('\nEpoch %d, Training Loss: %8f, Train Accuracy: %2f, Test Accuracy: %2f' % (epoch, running_loss, train_acc, test_acc))

        epoch_hist.append(epoch)
        acc_train_hist.append(train_acc)
        acc_test_hist.append(test_acc)

        if test_acc_max < test_acc:
            model_save_full_path = model_save_path + 'best.pkl'
            torch.save(model.state_dict(), model_save_full_path)
            test_acc_max = test_acc

    return epoch_hist, acc_train_hist, acc_test_hist

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          savepath='condusion_matrix.png',):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(-0.5, len(classes) - 0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(savepath)

def save_confusion_matrix(model, data_loader, savepath):
    model.eval()

    pred_result = np.zeros(len(data_loader.dataset))
    true_result = np.zeros(len(data_loader.dataset))

    count = 0
    correct = 0
    for batch_idx, (X, Y) in enumerate(data_loader):
        inputs = X.to(device)
        labels = Y.to(device)
        # wrap them in Variable
        inputs = Variable(inputs.float())
        labels = Variable(labels.long())
        with torch.no_grad():
            output = model(inputs)
        pred = torch.max(output, 1)[1]
        correct += pred.eq(labels.data).cpu().sum()
        count += 1

        for k in range(len(pred)):
            pred_result[batch_idx * BatchSize + k] = pred.cpu().numpy()[k]
        for k in range(len(Y)):
            true_result[batch_idx * BatchSize + k] = Y.cpu().numpy()[k]

    # calculate accuracy
    acc = correct.cpu().numpy() * 100.0 / len(data_loader.dataset)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(true_result, pred_result, labels=[0, 1, 2, 3, 4])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    # class_names = {'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'}
    class_names = [r'$0$', r'$1$', r'$2$', r'$3$', r'$4$']
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Confusion matrix, with normalization', savepath=savepath)
    # plt.show()
    plt.close()

    return acc


def demo_resnet18_training():

    lr = 1e-3
    weight_decay = 5e-4
    momentum = 0.9
    epochs_1 = 5
    epochs_2 = 10
    # max_num = 20655
    # # [ 1 / number of instances for each class]
    # weights = torch.from_numpy(
    #     np.array([max_num / 20655, max_num / 1955, max_num / 4210, max_num / 698, max_num / 581])).float()
    # class_weights = weights.to(device)

    resnet18 = ResNet(layer_num=18, pretrained=True, num_classes=5)
    resnet18_wo = ResNet(layer_num=18, pretrained=False, num_classes=5)

    net_ary = []
    net_ary.append(resnet18)
    net_ary.append(resnet18_wo)

    # for saving the best accuracy in each model
    train_max = np.zeros(shape=[len(net_ary)], dtype=float)
    test_max = np.zeros(shape=[len(net_ary)], dtype=float)

    plt.figure()
    for i in range(2):
        if i == 0:
            train_label_name = 'Train(with pretraining)'
            test_label_name = 'Test(with pretraining)'
            model_save_path = 'backup/resnet18_withp_'
            cm_save_path = 'backup/cm_resnet18_with_pretrain.png'
        else:
            train_label_name = 'Train(w/o pretraining)'
            test_label_name = 'Test(w/o pretraining)'
            model_save_path = 'backup/resnet18_wop_'
            cm_save_path = 'backup/cm_resnet18_wo_pretrain.png'

        net = net_ary[i]
        net = net.to(device)

        # First step: Fine-tune fc-layer
        for param in net.parameters():
            param.requires_grad = False
        net.classify.weight.requires_grad = True
        net.classify.bias.requires_grad = True
        summary(net, [(3, 512, 512)])
        optimizer = optim.SGD(net.classify.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # gamma = decaying factor
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        # criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        epoch_hist_1, acc_train_hist_1, acc_test_hist_1 = train(net, epochs_1, optimizer, scheduler, criterion,
                                                                model_save_path)

        # Second step: Fine-tune whole network
        for param in net.parameters():
            param.requires_grad = True
        summary(net, [(3, 512, 512)])
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        epoch_hist_2, acc_train_hist_2, acc_test_hist_2 = train(net, epochs_2, optimizer, scheduler, criterion,
                                                                model_save_path)

        # combine results in two steps
        epoch_hist = np.hstack((epoch_hist_1, epoch_hist_2))
        for k in range(epochs_1, epochs_1 + epochs_2, 1):
            epoch_hist[k] += epochs_1
        acc_train_hist = np.hstack((acc_train_hist_1, acc_train_hist_2))
        acc_test_hist = np.hstack((acc_test_hist_1, acc_test_hist_2))

        # find max accuracy
        train_max[i] = np.max(acc_train_hist)
        test_max[i] = np.max(acc_test_hist)

        # calculate and save confusion matrix
        _ = save_confusion_matrix(net, val_loader, cm_save_path)

        # plot accuracy curve
        plt.plot(epoch_hist, acc_train_hist, label=train_label_name)
        plt.plot(epoch_hist, acc_test_hist, label=test_label_name)

    print(train_max)
    print(test_max)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.title("Result Comparison (ResNet18)")
    plt.legend()
    plt.savefig("backup/accuracy_resnet18.jpg")
    plt.show()


def demo_resnet50_training():
    lr = 1e-3
    weight_decay = 5e-4
    momentum = 0.9
    epochs_1 = 5
    epochs_2 = 10

    # max_num = 20655
    # # [ 1 / number of instances for each class]
    # weights = torch.from_numpy(
    #     np.array([max_num / 20655, max_num / 1955, max_num / 4210, max_num / 698, max_num / 581])).float()
    # class_weights = weights.to(device)

    resnet50 = ResNet(layer_num=50, pretrained=True, num_classes=5)
    resnet50_wo = ResNet(layer_num=50, pretrained=False, num_classes=5)

    net_ary = []
    net_ary.append(resnet50)
    net_ary.append(resnet50_wo)

    # for saving the best accuracy in each model
    train_max = np.zeros(shape=[len(net_ary)], dtype=float)
    test_max = np.zeros(shape=[len(net_ary)], dtype=float)

    plt.figure()
    for i in range(2):
        if i == 0:
            train_label_name = 'Train(with pretraining)'
            test_label_name = 'Test(with pretraining)'
            model_save_path = 'backup/resnet50_withp_'
            cm_save_path = 'backup/cm_resnet50_with_pretrain.png'
        else:
            train_label_name = 'Train(w/o pretraining)'
            test_label_name = 'Test(w/o pretraining)'
            model_save_path = 'backup/resnet50_wop_'
            cm_save_path = 'backup/cm_resnet50_wo_pretrain.png'

        net = net_ary[i]
        net = net.to(device)

        # First step: Fine-tune fc-layer
        ct = 0
        for child in net.children():
            ct += 1
            if ct < 10:
                for param in child.parameters():
                    param.requires_grad = False
        summary(net, [(3, 512, 512)])
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # gamma = decaying factor
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        # criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        epoch_hist_1, acc_train_hist_1, acc_test_hist_1 = train(net, epochs_1, optimizer, scheduler, criterion, model_save_path)

        # Second step: Fine-tune whole network
        for param in net.parameters():
            param.requires_grad = True
        summary(net, [(3, 512, 512)])
        optimizer_2 = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # gamma = decaying factor
        scheduler_2 = StepLR(optimizer, step_size=100, gamma=0.1)
        # criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        criterion_2 = nn.CrossEntropyLoss().to(device)
        epoch_hist_2, acc_train_hist_2, acc_test_hist_2 = train(net, epochs_2, optimizer_2, scheduler_2, criterion_2,
                                                                model_save_path)

        # combine results in two steps
        epoch_hist = np.hstack((epoch_hist_1, epoch_hist_2))
        for k in range(epochs_1, epochs_1 + epochs_2, 1):
            epoch_hist[k] += epochs_1
        acc_train_hist = np.hstack((acc_train_hist_1, acc_train_hist_2))
        acc_test_hist = np.hstack((acc_test_hist_1, acc_test_hist_2))

        # find max accuracy
        train_max[i] = np.max(acc_train_hist)
        test_max[i] = np.max(acc_test_hist)

        # calculate and save confusion matrix
        _ = save_confusion_matrix(net, val_loader, cm_save_path)

        # plot accuracy curve
        plt.plot(epoch_hist, acc_train_hist, label=train_label_name)
        plt.plot(epoch_hist, acc_test_hist, label=test_label_name)

    print(train_max)
    print(test_max)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.title("Result Comparison (ResNet50)")
    plt.legend()
    plt.savefig("backup/accuracy_resnet50.jpg")
    plt.show()

def demo_from_best_model(resnet_layer, pretrained, num_classes, path):

    assert resnet_layer == 18 or resnet_layer == 50

    net_best = ResNet(layer_num=resnet_layer, pretrained=pretrained, num_classes=num_classes)
    net_best = net_best.to(device)
    net_best.load_state_dict(torch.load(path))
    net_best.eval()
    best_acc = save_confusion_matrix(net_best, val_loader, 'backup_demo/cm_best.png')
    print('test_best_accuracy = %.2f' % best_acc)


demo_resnet18_training()
demo_resnet50_training()
# demo_from_best_model(resnet_layer=18, pretrained=True, num_classes=5, path='backup_demo/resnet18_withp_best.pkl')
