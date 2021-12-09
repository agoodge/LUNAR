# imports
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import utils
import variables as var
from torch_geometric.nn import MessagePassing
from copy import deepcopy

# Message passing scheme
class GNN1(MessagePassing):
    def __init__(self,k):
        super(GNN1, self).__init__(flow="target_to_source")
        self.k = k
        self.hidden_size = 256
        self.network = nn.Sequential(
            nn.Linear(k,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,1),
            nn.Sigmoid()
            )

    def forward(self, x, edge_index, edge_attr):
        self.network = self.network.to(dtype = torch.float32)
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_attr, k = self.k, network=self.network)
        return out

    def message(self,x_i,x_j,edge_attr):
        # message is the edge weight
        return edge_attr

    def aggregate(self, inputs, index, k, network):
        # concatenate all k messages
        self.input_aggr = inputs.reshape(-1,k)
        # pass through network
        out = self.network(self.input_aggr)
        return out

# GNN
class GNN(torch.nn.Module):
    def __init__(self, k):
        super(GNN, self).__init__()
        self.k = k
        self.L1 = GNN1(self.k)
    def forward(self,data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        out = self.L1(self.x, self.edge_index, self.edge_attr)
        out = torch.squeeze(out,1)
        return out
    
def run(train_x,train_y,val_x,val_y,test_x,test_y,dataset,seed,k,samples,train_new_model):  

    # loss function
    criterion = nn.MSELoss(reduction = 'none')    

    # path to save model parameters
    model_path = 'saved_models/%s/%d/net_%d.pth' %(dataset,k,seed)
    if not os.path.exists(os.path.dirname(model_path)):
       os.makedirs(os.path.dirname(model_path)) 
    
    x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, idx = utils.negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, samples, var.proportion, var.epsilon)
    data = utils.build_graph(x, y, dist, idx)
        
    data = data.to(var.device)                                                                    
    torch.manual_seed(seed)
    net = GNN(k).to(var.device)
   
    if train_new_model == True:
      
        optimizer = optim.Adam(net.parameters(), lr = var.lr, weight_decay = var.wd)
   
        with torch.no_grad():
            
            net.eval()
            out = net(data)
            loss = criterion(out,data.y)

            val_loss = loss[val_mask == 1].mean()
            val_score = roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu())

            best_val_score = 0
           
        # training
        for epoch in range(var.n_epochs):
            net.train()
            optimizer.zero_grad()
            out = net(data)
            # loss for training data only
            loss = criterion(out[train_mask == 1],data.y[train_mask == 1]).sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                net.eval()
                out = net(data)
                loss = criterion(out,data.y)
                              
                val_loss = loss[val_mask == 1].mean()
                val_score = roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu())

                # if new model gives the best validation set score
                if val_score >= best_val_score:
                          
                    # save model parameters
                    best_dict = {'epoch': epoch,
                           'model_state_dict': deepcopy(net.state_dict()),
                           'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                           'val_loss': val_loss,
                           'val_score': val_score,
                           'k': k,}
                    
                    # save best model
                    #torch.save(best_dict, model_path)
                    
                    # reset best score so far
                    best_val_score = val_score
       
        # load best model
        net.load_state_dict(best_dict['model_state_dict'])
        
    # if not training a new model, load the saved model
    if train_new_model == False:
        
        load_dict = torch.load(model_path)
        net.load_state_dict(load_dict['model_state_dict'])
 
    # testing
    with torch.no_grad():
        net.eval()
        out = net(data)
        loss = criterion(out,data.y)
       
    # return output for test points
    return out[test_mask==1].cpu()