import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# Define the condition
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class GCNConv(MessagePassing): # Graph Convolution Layer with Weight
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', node_dim=-2)  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, adj_mat, weight_mat):
        # x has shape [batchsize ,N, in_channels]

        # turn adj_mat into edge_index, has shape [2, E]
        edge_index = adj_mat.nonzero(as_tuple=False).t()
  

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        # Add self-loop manually to weight_matrix and turn weight_matrix into edge_weight has shape [E]
        weight_mat += torch.eye(x.size(1)).to(device)
        edge_weight = weight_mat[edge_index[0], edge_index[1]]
     
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x) # [batchsize, N, out_channels]

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(1), dtype=x.dtype)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        # norm = deg_inv_sqrt[row] *deg_inv_sqrt[col]

         
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        return norm.view(-1, 1) * x_j 
    


class TGCNGraphConvolution(nn.Module): # output: w[f(A,X), h]+ b 
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units  # hidden_dim
        self._output_dim = output_dim
        self._bias_init_value = bias
        # self.register_buffer(
        #     "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        # )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.GCN=GCNConv(in_channels=1,out_channels=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state, adj_mat, weight_mat):
        # inputs: (batchsize,node_num)
        # hidden_state: (batchsize, node_num, num_gru_units)

        batch_size, num_nodes = inputs.shape  
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # inputs (batch_size, num_nodes, 1)
        inputs= self.GCN(inputs,adj_mat, weight_mat)

        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        #[x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)

        # [x, h] (batch_size * num_nodes, self._num_gru_units + 1)
        concatenation = concatenation.reshape(
             (batch_size * num_nodes, self._num_gru_units + 1)
         )
        
        # W[x, h] + b (batch_size * num_nodes, output_dim)
        outputs = concatenation @ self.weights + self.biases
        # W[x, h] + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # W[x, h] + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs



class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int): #num_gru_units=hidden_dim
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        #self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )    # r, u 
        self.graph_conv2 = TGCNGraphConvolution(
             self._hidden_dim, self._hidden_dim
        )    # c

    def forward(self, inputs, hidden_state, adj_mat, weight_mat):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state, adj_mat, weight_mat)) # warning concate=nan


        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state, adj_mat, weight_mat))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
  
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state


class ATGCN(nn.Module):   # Wrapper
    def __init__(self,node_size, hidden_dim: int, output_dim=1): 
        super(ATGCN, self).__init__()
        self._input_dim = node_size
        self._hidden_dim = hidden_dim
        self._output_dim=output_dim  # 1
        #self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self._input_dim, self._hidden_dim)
        self.Linear=nn.Linear(self._hidden_dim,self._output_dim)

    def forward(self, inputs, adj_mat, weight_mat):  # inputs:(batch_size, seq_len, node_num)    
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None

        for i in range(seq_len):  # Through Mutiple cells
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state, adj_mat, weight_mat)   #(output=hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        output=self.Linear(output)
        output = output.reshape((batch_size, self._output_dim, num_nodes))
        return output
