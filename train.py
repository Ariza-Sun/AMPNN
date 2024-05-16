import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as U
import time
import datetime
import numpy as np
import copy

from torch.utils.tensorboard import SummaryWriter  

from utils.ReadFile import *
from utils.eval import *
from model.Generator import *
from model.Predictor import *



##########Parser################

### Task level
parser = argparse.ArgumentParser(description='PyTorch Implementation of AMPNN')
parser.add_argument('--data', type=str, default='SH_Park',
                    help='name of the dataset')


### Training Level
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--epochs',type=int,default=500,help='')

### Data Organization
parser.add_argument('--in_dim',type=int,default=1,help='dimension of node embeddings')
parser.add_argument('--seq_in_len',type=int,default=3,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1,help='how far future to predict')

### Model_level
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--edge_minibatch',type=int,default=8,help='how many edges to be updated at once for averagely one node')
parser.add_argument('--lr_gen',type=float,default=1, help='learning rate of generator')
parser.add_argument('--lr_pre',type=float,default=0.0001,help='learning rate of predictor')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--hidden_num',type=int,default=128, help='dimension of hidden state for gru')


args = parser.parse_args()
device = torch.device(args.device)



def train(train_loader,Gen,PreNet,op_gen,op_prenet,max_num,min_num):

    loss_fn = nn.MSELoss()
    loss_batch=[]


    # feed training set
    for idx, data in enumerate(train_loader):
        # data
        inp,labels=data
        # print(inp.shape)  # (batchsize, seq_len, node_num)
        # print(labels.shape) # (batchsize, pre_len, node_num)
        inp=inp.to(device)
        labels=labels.to(device)
        
        ##########
        # Forward
        ##########
        # Tune: Find how Gen.weight_mat change after BP
        # cache_mat=copy.deepcopy(Gen.weight_matrix)
        op_gen.zero_grad()
        op_prenet.zero_grad()

        # Sample Matrix
        partial_weight_matrix, partial_adj_matrix=Gen.sample_matrix()
        partial_weight_matrix.to(device)
        partial_adj_matrix.to(device)
   
        y_hat =PreNet(inp, partial_adj_matrix, partial_weight_matrix)  #(batchsize,pre_len,node_num)

        loss=loss_fn(y_hat,labels)
     
        ###########
        # Backward
        ###########
        loss.backward()
        # cut gradient in case nan shows up
        U.clip_grad_norm_(Gen.weight_matrix, 0.000075)
    
        op_prenet.step()
        op_gen.step()

        # Tune: Find how Gen.weight_mat change after BP
        # print(torch.sum(torch.abs(Gen.weight_matrix-cache_mat)).item())

        #ensure weight matrix to be [0,1]
        Gen.clamp_parameters()

        loss_batch.append(loss.item())
    
    return np.mean(loss_batch)



def test(val_loader,Gen,PreNet,max_num, min_num):
    
    # Prediction
    loss_batch=[]
    RMSE=[]
    MAE=[]
    Acc=[]

    loss_fn = nn.MSELoss()
    # feed validating set
    for idx, data in enumerate(val_loader):
        # data
        inp,labels=data
        # print(inp.shape)  # (batchsize, seq_len, node_num)
        # print(labels.shape) # (batchsize, pre_len, node_num)
        inp=inp.to(device)
        labels=labels.to(device)
       
        ##########
        # Forward
        ##########

        # Sample Matrix
        partial_weight_matrix, partial_adj_matrix=Gen.sample_matrix()
        partial_weight_matrix.to(device)
        partial_adj_matrix.to(device)
   
        y_hat =PreNet(inp, partial_adj_matrix, partial_weight_matrix)  #(batchsize,pre_len,node_num)

        loss=loss_fn(y_hat,labels)

        ##################
        # calculate metric 
        ##################
        rmse,mae,acc,r2,var=dyn_evaluator(y_hat.cpu(),labels.cpu(),labels.size(0), max_num, min_num)
        RMSE.append(rmse)
        MAE.append(mae)
        Acc.append(acc)

        loss_batch.append(loss.item())

    return np.mean(loss_batch),np.mean(RMSE),np.mean(MAE),np.mean(Acc),Gen.edge_coverage,Gen.edge_overlap_rate


if __name__ == "__main__":
    print(args)
    
    #Tensorboard
    writer = SummaryWriter('runs/'+args.data+'_'+str(datetime.datetime.now()))

    # Load data
    train_loader,val_loader, test_loader, node_num, max_num, min_num= Load_Dataset(batch_size=args.batch_size,seq_len=args.seq_in_len,pre_len=args.seq_out_len,dataset=args.data,norm=args.normalize)
    print('Successfully loading dataset')

    # Init Model
    Gen=Graph_Generator(node_num=node_num, edge_minibatch=args.edge_minibatch)
    PreNet=ATGCN(node_size=node_num,hidden_dim=args.hidden_num)

    Gen=Gen.to(device)
    PreNet=PreNet.to(device)

    # Init Optimizer
    Optim_Gen=optim.Adam(Gen.parameters(),lr=args.lr_gen)
    Optim_PreNet=optim.Adam(PreNet.parameters(),lr=args.lr_pre)


    best_val = 10000000
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss= train(train_loader,Gen,PreNet,Optim_Gen,Optim_PreNet,max_num, min_num)
            val_loss,val_rmse,val_mae,val_acc,edge_cover_rate,edge_overlap_rate= test(val_loader,Gen,PreNet,max_num, min_num)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} |  val_loss {:5.4f} | valid rmse {:5.4f} | valid mae {:5.4f} | valid acc  {:5.4f} | edge coverage {:5.4f} | overlap {:5.4f} |'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rmse, val_mae, val_acc,edge_cover_rate,edge_overlap_rate), flush=True)
            
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            writer.add_scalar("val_rmse",val_rmse,epoch)
            writer.add_scalar("val_mae",val_mae,epoch)
            writer.add_scalar("val_acc",val_acc,epoch)
            if epoch>0:
                writer.add_scalar("Coverage_Rate",edge_cover_rate,epoch)
                writer.add_scalar("Overlap_Rate",edge_overlap_rate,epoch)

            # Save the model if the validation loss is the best we've seen so far
            if val_loss < best_val:
                with open('save/'+args.data+'/Gen.pt', 'wb') as f:
                    torch.save(Gen, f)
                with open('save/'+args.data+'/PreNet.pt', 'wb') as f:
                    torch.save(PreNet, f)
                best_val = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print('begin testing')
    test_loss,tes_rmse,test_mae,test_acc,edge_cover_rate,edge_overlap_rate= test(test_loader,Gen,PreNet,max_num, min_num)
    print('test result: test_remse: {:5.4f} |  test_mae: {:5.4f} | test_acc: {:5.4f} )'.format(test_rmse,test_mae,test_acc),flush=True)































