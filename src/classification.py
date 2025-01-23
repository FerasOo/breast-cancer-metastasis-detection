from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
import torch.optim as optim
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from sklearn import metrics
import os
images_dir = "./data/images/"
# 1147
class Hyperparameters:
    batch_size = 32
    test_batch_size = 252
    epochs = 50
    lr = 0.0003
    image_size = 196
    stats_dir = "./stats/part1"
    models_dir = "./saved_models/part1"
    predictions_dir = "./predictions/part1"


# get parameters for training and testing
args = Hyperparameters()

# create stats_dir, models_dir, and predictions_dir if do not exist
if not os.path.exists(args.stats_dir):
    os.makedirs(args.stats_dir)

if not os.path.exists(args.models_dir):
    os.makedirs(args.models_dir)

if not os.path.exists(args.predictions_dir):
    os.makedirs(args.predictions_dir)

class CNN(nn.Module):
    def __init__(self, num_features=1, ):
        super().__init__()

        torch.manual_seed(29)
        self.linear_size = int(48 * args.image_size * args.image_size / 16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # Half the dimension size
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(self.linear_size, 1)
        # self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), self.linear_size)
        x = self.dropout(x)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        x = self.sigmoid(x)
        return x





# set the seed for reproducibility
torch.manual_seed(1)


class imageDataSet(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

        # we can call len(dataset) to return the size

    def __len__(self):
        return self.y_data.shape[0]


df = pd.read_csv('./data/train.csv')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(args.image_size),
   # transforms.RandomRotation(25),
   #  transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    # transforms.Normalize((0), (1))
    ])
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(args.image_size),
# transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

x_train = torch.zeros(1147, 3, args.image_size, args.image_size)
y_train = torch.zeros(1147, 1)
x_test = torch.zeros(252, 3, args.image_size, args.image_size)
y_test = torch.zeros(252, 1)
for index, row in df.iterrows():
    img = Image.open(images_dir+row['img_id']+".jpeg")
    label = row['label']
    #t = transform(img)

    #t = to_ten(img)
    if index < 1147:
        t = transform(img)
        x_train[index] = t
        y_train[index] = label
    else:
        t2 = transform2(img)
        x_test[index-1147] = t2
        y_test[index-1147] = label


train_dataset = imageDataSet(x_train, y_train)
test_dataset = imageDataSet(x_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

device = torch.device("cpu")

model = CNN().to(device)

# optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=0.005)
#
# scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
optimizer = torch.optim.Adam(model.parameters(), args.lr)
criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss(reduction="sum")

def train(model, device, train_loader, optimizer):
    num_predictions = 0
    running_loss = 0.0
    running_correct = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # step1 - forward pass
        output = model(data)

        # step2 - calculate loss
        #loss = F.nll_loss(output, target)
        #target = target.view(-1)
        loss = criterion(output, target)

        # step3 - clear the gradients
        optimizer.zero_grad()

        # step4 - backpropagation
        loss.backward()

        # step5 - update the weights
        optimizer.step()

        ##### collect stats #####
        num_predictions += target.size(0)
        predictions = (output > 0.5).long()
        running_correct += (predictions == target).sum()

        running_loss += loss.item() * target.size(0)

        # _, predicted = torch.max(output, 1)
        #predict = output.value()
        #correct = (predicted == target).sum().item()
        #running_correct += correct
        #########################

    ##### epoch stats #####
    train_loss = running_loss / num_predictions
    train_acc = running_correct / num_predictions
    #######################

    return train_loss, train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #target = target.view(-1)
            loss = criterion(output, target)
            test_loss += loss.item() * target.size(0)
            predictions = (output > 0.5).long()
            correct += (predictions == target).sum()
            try:
                auc = metrics.roc_auc_score(target, output)
            except ValueError:
                auc = 0
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    #####
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    #####

    return test_loss, test_acc, auc


df = pd.read_csv('./data/test.csv')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(args.image_size),
# transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    ])




def predict(model, device, predictions_filename):
    ##### prepare the file
    with open(predictions_filename, 'w') as f_predictions_filename:
        f_predictions_filename.write('img_id,cancer_score\n')


    ##### obtain predictions and write into text file
    model.eval()
    with torch.no_grad():
        for index, row in df.iterrows():
            img = Image.open(images_dir + row['img_id'] + ".jpeg")
            data = transform(img)
            t = torch.zeros(1, 3, args.image_size, args.image_size)
            t[0] = data
            output = model(t)
            with open(predictions_filename, 'a') as f_predictions_filename:
                f_predictions_filename.write(f"{row['img_id']},{output.item():.3f}\n")



stats_filename = "{}/mnist_mlp_loss_acc.txt".format(args.stats_dir)
print("Training ...")
best_test_loss = 1
best_test_auroc = 0
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer)
    test_loss, test_acc, auroc = test(model, device, test_loader)


    ##### write stats into file #####
    with open(stats_filename, 'a') as f_stat_filename:
        f_stat_filename.write('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(epoch, train_loss, train_acc, test_loss, test_acc, auroc))
    #################################

    print("Epoch={:d} ### train_loss={:.4f}, train_acc={:.4f}, test_loss={:.4f}, test_acc={:.4f}, AUROC={:.4f}".format(epoch, train_loss, train_acc, test_loss, test_acc, auroc), end="")

    msg = ""
    if epoch < 10:
        print(msg)
        continue
    if best_test_loss > test_loss:
        best_test_loss = test_loss
        msg += "\t BEST LOSS"
        predict(model, device, f"{args.predictions_dir}/predictions_best_loss_{epoch}.csv")

    else:
        if auroc > 0.92:
            predict(model, device, f"{args.predictions_dir}/predictions_best_auroc_{epoch}.csv")
        msg += "\t HIGH AUROC"
    print(msg)

FILE = f"{args.models_dir}/final_model.pth"
torch.save(model.state_dict(), FILE)

predict(model, device, f"{args.predictions_dir}/predictions_final.csv")


