import torch
from argparse import Namespace
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from utils2 import weights_init, compute_acc
from MCE_ST import *




from sklearn import metrics
from torchsummary import summary

torch.backends.cudnn.deterministic = True
#torch.manual_seed(254)
torch.manual_seed(999)



def train( n_band, data_loader, n_category, run, **hyper):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:

        data_loader: a PyTorch dataset loader

    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    opt = Namespace(**hyper)


    if opt.model =='MCE-ST':
        channel = 1
        netD = MCE_ST(n_category, n_band, opt.dmodel, opt.depth, opt.nhead)          
    
    
    #######print the model###############
    summary(netD.cuda(),(n_band,channel))
    
    class_criterion = nn.CrossEntropyLoss()



    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    epoch = opt.epoch

    if class_criterion is None:
        raise Exception("Missing criterion. You must specify a loss function for classification.")


    #tensor placeholders
    input = torch.FloatTensor(opt.batch_size,n_band)
    class_label = torch.LongTensor(opt.batch_size) #the label for classification

    #if using cuda
    if torch.cuda.is_available():
        netD.cuda()
        class_criterion.cuda()
        input = input.cuda()
        class_label = class_label.cuda()



    input = Variable(input)
    class_label = Variable(class_label)


    best_accuracy = 0.0

    min_train_loss = np.inf
    

    start.record()
    for e in range(epoch):
        # Run the training loop for one epoch
        num_correct = 0
        num_data = 0
        train_loss = 0.0
        

        for i, data in enumerate(data_loader):

            
            netD.zero_grad()
            real_cpu, label = data #what is real_cpu? is it real data?
            batch_size = real_cpu.size(0)

            #Load the data into the GPU if required
            real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            class_label.resize_(batch_size).copy_(label) # I assume that the class_label is label of classification
            class_output = netD.forward(input)

            class_errD_real = class_criterion(class_output, class_label)
            errD_real = class_errD_real
            errD = errD_real
            errD_real.backward() #back propagating the error through the model
            #compute the current classification accuracy
            accuracy, correct, length_data = compute_acc(class_output, class_label)

            optimizerD.step() #update the model to optimize loss
            train_loss += class_errD_real.item() 

            num_correct = num_correct + correct
            num_data = num_data+ length_data

        accuracy_iter = float(num_correct)*100 / float(num_data)
        print("iter: %d, total accuracy: %.4f" % (e, accuracy_iter))
        train_loss=train_loss / len(data_loader)
        print(f'Epoch {e+1} \t Training Loss: {train_loss}')
        if min_train_loss > train_loss:
            print(f'Training Loss Decreased({min_train_loss:.6f}--->{train_loss:.6f}) \t Saving The Model')
            min_train_loss = train_loss
            # Saving State Dict
            name = 'model/%s_%s_%d.pth'%(hyper["model"],hyper["dataset"],run)
            torch.save(netD, name)
                
    end.record()
    torch.cuda.synchronize()
    print("Time:" ,start.elapsed_time(end))



# evaluate the model my code
def evaluate_model(testdata_loader, model_path, model_name=None):

    #open the model
    
    device = torch.device("cuda")
    model = torch.load(model_path)
    #print(model)
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(testdata_loader):
        # evaluate the model on the test set
        inputs = inputs.to(device)
        yhat = model.forward(inputs.float())
        # retrieve numpy array


        yhat = yhat.cpu()
        yhat = yhat.detach().numpy()
        actual = targets.numpy()

        actual = actual.reshape((len(actual), 1))

        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate accuracy
    y_pred = predictions.argmax(axis=1)
    acc = metrics.accuracy_score(y_pred, actuals)
    print("accuracy 4: ", acc)
    return predictions, actuals

