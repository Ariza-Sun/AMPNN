import argparse
import torch
import torch.nn as nn
import copy
from torch.nn.parameter import Parameter

# Define the condition
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Graph_Generator(nn.Module):
    def __init__(self, node_num, edge_minibatch):
        super(Graph_Generator, self).__init__()
        self.node_num=node_num
        self.edge_num=node_num*edge_minibatch
        self.weight_matrix=Parameter(torch.rand(node_num, node_num)) 
        self.partial_adj_matrix=None
        self.partial_weight_matrix=None

        # Tune
        self.chosen_edge_cache=torch.tensor(-1)
        self.chosen_edge_history=torch.tensor([])
        self.edge_coverage=None
        self.edge_overlap_rate =None

        
    def single_cal_row_col(self, index):
        # calculate row & col from index of flatten vector
        row=index//self.node_num
        col=index % self.node_num
        return row, col
    
    def cal_row_col(self, indices):
        # batch calculation
        rows = []
        cols = []
        for index in indices:
            row, col = self.single_cal_row_col(index)
            rows.append(row)
            cols.append(col)
        return torch.tensor(rows), torch.tensor(cols)

    def sample_matrix(self):
    # draw partial_matrix from weight_matrix
        weight_vector=self.weight_matrix.view(1,-1)
     
        # choose edges
        self.partial_adj_matrix=torch.zeros(self.node_num,self.node_num).to(device)
        # sorted_indices = torch.argsort(weight_vector, descending=True)
        # choosen_indices=sorted_indices[0, 0:self.edge_num]
        _, choosen_indices = torch.topk(weight_vector, k=self.edge_num)
        choosen_indices=torch.squeeze(choosen_indices,dim=0)

        rows, cols = self.cal_row_col(choosen_indices)
        self.partial_adj_matrix[rows, cols] = 1

        self.partial_weight_matrix=self.partial_adj_matrix.mul(self.weight_matrix)
 
        ##### 
        #Tune
        #####

        #Find how many chosen_edge change over training
        self.chosen_edge_cache=self.chosen_edge_cache.to(device)
        common_elements = torch.isin(choosen_indices, self.chosen_edge_cache)
        common_count = torch.sum(common_elements).item()
        self.edge_overlap_rate=common_count/len(choosen_indices)
        #print("Overlap Rate: "+str(self.edge_overlap_rate))
        self.chosen_edge_cache=copy.deepcopy(choosen_indices)

        # Find how mang edges has been chosen to update
        self.chosen_edge_history=self.chosen_edge_history.to(device)
        tmp=torch.cat([self.chosen_edge_history,choosen_indices])
        self.chosen_edge_history=torch.unique(tmp)
        self.edge_coverage=len(self.chosen_edge_history)/weight_vector.size(1)
       # print('Coverage: '+str(self.edge_coverage))

        return self.partial_weight_matrix, self.partial_adj_matrix
    
    def clamp_parameters(self):
        # set weight to be non-negative
        with torch.no_grad():
            self.weight_matrix.data.clamp_(min=0,max=1)

