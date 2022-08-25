import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from utils import save_history

def learning_scheduler(optimizer, epoch, lr=0.001, lr_decay_epoch=10):
    lr = lr * (0.5**(epoch // lr_decay_epoch))

    for param in optimizer.param_groups:
        param['lr'] = lr
    return optimizer


def train(model, train_set, val_set, args, trainlogfile, log_path):

    #init dataloaders
    if torch.cuda.is_available():
        device = 'cuda'
        trainloader = DataLoader(train_set,batch_size = args.batch_size, shuffle = True, drop_last = True, num_workers=6, pin_memory=True)
    else:
        device = 'cpu'
        trainloader = DataLoader(train_set,batch_size = args.batch_size, shuffle = True, drop_last = True)


    valloader = DataLoader(val_set, batch_size = 1, shuffle = True)

    #init datalogging
    history = dict(train = [],val = [])
    trainf = open(trainlogfile,'a')
    trainf.write("Logging Validation statistics below:\n")

    #init model
    untrained_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(untrained_weights)
    best_weights = copy.deepcopy(model.state_dict)


    #init training
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = 2000000000
    patience = args.patience
    early_stop_counter = 0

    # train model until validation loss stops converging
    for epoch in range(args.epochs):
        optimizer = learning_scheduler(optimizer, epoch, lr = args.lr, lr_decay_epoch=args.lr_decay)
        model = model.train().to(device)

        train_losses = []
        #forward, backward, log results
        for idx, (inputs,labels) in enumerate(trainloader):
            optimizer.zero_grad()

            inputs, labels = inputs.float().to(device), labels.to(device)


            out = model.forward(Variable(inputs))


            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())


        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(valloader):
                inputs, labels = inputs.float().to(device), labels.to(device)

                output = model.forward(Variable(inputs))
                loss = criterion(output,labels)
                val_losses.append(loss.item())


        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print("Epoch: {}, train loss {}, validation loss: {}".format(epoch,train_loss,val_loss))
        trainf.write("Epoch: {}, average train CE {:.6f}, average val CE {:.6f}\n".format(epoch,train_loss,val_loss))

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            model.load_state_dict(best_weights)
            print("early_stop_counter: ",early_stop_counter,"\n")
            if early_stop_counter >=patience:
                save_history(history,args.log_path,args.arch)
                trainf.close()
                return model
            else:
                save_history(history,args.log_path,args.arch)
                early_stop_counter+=1
    save_history(history,args.log_path,args.arch)
    trainf.close()
    return model

def test(model, test_set, args, testlogfile):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    testf = open(testlogfile, 'a')
    testloader = DataLoader(test_set,batch_size = 1, shuffle = True)

    corrects = 0
    for idx, (input,label) in enumerate(testloader):
        input, label = input.float().to(device), label.to(device)
        input, label = Variable(input), Variable(label)
        output = model.forward(input)
        _, predicted = torch.max(output.data,1)
        corrects += torch.sum(predicted==label.data).item()

    accuracy=corrects/test_set.__len__()

    testf.write("Accuracy: {:.6f}\n".format(accuracy))
    testf.close()
