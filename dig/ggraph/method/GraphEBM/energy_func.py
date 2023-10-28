import torch
from torch import nn
from torch.nn import functional as F
import networkx as nx
from torch_scatter import scatter_add
import numpy as np

def swish(x):
    return x * torch.sigmoid(x)



class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)
        else:
            weight_sn = weight / sigma
        return weight_sn, u
    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)
        module.register_forward_pre_hook(fn)
        return fn
    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)
        
def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()
    SpectralNorm.apply(module, 'weight', bound=bound)
    return module


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_edge_type, std, bound=True, add_self=False):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.linear_node = spectral_norm(nn.Linear(in_channels, out_channels), std=std, bound=bound)
        self.linear_edge = spectral_norm(nn.Linear(in_channels, out_channels * num_edge_type), std=std, bound=bound)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels
        eps = torch.tensor([0], dtype=torch.float32)
        self.eps = nn.Parameter(eps)
        self.W = nn.ModuleList([nn.Linear(self.in_ch, self.out_ch) for i in range(4)])
        self.a = nn.ModuleList([nn.Linear(2 * self.out_ch, 1) for i in range(4)])
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.linear_edge1 = nn.Linear(in_channels+1, out_channels * num_edge_type)
        self.linear_node1 = nn.Linear(in_channels, in_channels)
        self.linear_edge2 = nn.Linear(out_channels * num_edge_type, out_channels * num_edge_type)
        self.output = nn.Linear(self.out_ch, self.out_ch)

# GraphConv(1,1,4,1).to('cuda')

    def forward(self, adj, h, atype):
        # orginal
        if atype == 0:
            mb, node, _ = h.shape 
            if self.add_self:
                h_node = self.linear_node(h) 
            m = self.linear_edge(h)
            m = m.reshape(mb, node, self.out_ch, self.num_edge_type) 
            m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
            hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
            hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)
            if self.add_self:
                return hr+h_node  #
            else:
                return hr
        # RGCN
        elif atype == 1:
            mb, node, _ = h.shape # h: (batchsize, ch, in)
            h_node = self.linear_node(h) # h_node: (batchsize, ch, out)
            m = self.linear_edge(h) # h_node: (batchsize, ch, out*4)
            m = m.reshape(mb, node, self.out_ch, self.num_edge_type) # batch, ch, out, 4
            m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
            hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
            hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)
            return hr+h_node
        
        # GAT
        elif atype == 2:
            mb, node, _ = h.shape # h: (batchsize, ch, in)
            m = self.linear_edge(h) # h_node: (batchsize, ch, out*4)
            m = m.reshape(mb, node, self.out_ch, self.num_edge_type) # batch, ch, out, 4
            m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
            hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
            hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)

            feature = []
            for i in range(self.num_edge_type):
                s = self.W[i](h)
                batch_size, num_nodes, out_features = s.size()
                a_input = torch.cat([s.unsqueeze(1).expand(-1, num_nodes, -1, -1),
                                        s.unsqueeze(2).expand(-1, -1, num_nodes, -1)], dim=-1)
                e = self.leaky_relu(self.a[i](a_input).squeeze(-1))
                zero_vec = -9e15 * torch.ones_like(e)
                asd = adj[:, i, :, :]
                attention = torch.where(asd > 0, e, zero_vec)
                attention = F.softmax(attention, dim=2)
                h_prime = torch.bmm(attention, s)
                feature.append(h_prime)
            return sum(feature) + hr
        
        # GearNet
        elif atype == 3:
            mb, node, _ = h.shape # h: (batchsize, ch, in)
            h_node = self.linear_node(h) # h_node: (batchsize, ch, out)
            h_node = F.relu(h_node)
            m1 = self.linear_edge(h) # h_node: (batchsize, ch, out*4)
            m1 = F.relu(m1)
            m2 = self.linear_edge2(m1) # h_node: (batchsize, ch, out*4)
            m2 = m2+m1
            m2 = F.relu(m2)
            m2 = m2.reshape(mb, node, self.out_ch, self.num_edge_type) # batch, ch, out, 4
            m2 = m2.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
            hr = torch.matmul(adj, m2)  # hr: (batchsize, edge_type, node, ch)
            hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)

            return hr+h_node

        # GATv2
        elif atype == 4:
            mb, node, _ = h.shape # h: (batchsize, ch, in)
            m = self.linear_edge(h) # h_node: (batchsize, ch, out*4)
            m = m.reshape(mb, node, self.out_ch, self.num_edge_type) # batch, ch, out, 4
            m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
            hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
            hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)

            feature = []
            for i in range(self.num_edge_type):
                s = self.W[i](h)
                batch_size, num_nodes, out_features = s.size()
                a_input = self.leaky_relu(torch.cat([s.unsqueeze(1).expand(-1, num_nodes, -1, -1),
                                        s.unsqueeze(2).expand(-1, -1, num_nodes, -1)], dim=-1))
                e = self.a[i](a_input).squeeze(-1)
                zero_vec = -9e15 * torch.ones_like(e)
                asd = adj[:, i, :, :]
                attention = torch.where(asd > 0, e, zero_vec)
                attention = F.softmax(attention, dim=2)
                h_prime = torch.bmm(attention, s)
                feature.append(h_prime)
            return sum(feature) + hr

        # GSN
        elif atype == 5:
            # square
            # def count_squares(adjacency_matrix):
            #     num_nodes = len(adjacency_matrix)
            #     squares_count = [0] * num_nodes

            #     # Perform a depth-first search to count squares
            #     def dfs(node, start_node, depth, path):
            #         if depth == 0 and node == start_node:
            #             squares_count[start_node] += 1
            #             return
            #         if depth <= 0:
            #             return
            #         for neighbor in range(num_nodes):
            #             if adjacency_matrix[node][neighbor] == 1 and neighbor not in path:
            #                 dfs(neighbor, start_node, depth - 1, path + [neighbor])

            #     # Iterate over all nodes and start DFS from each node
            #     for node in range(num_nodes):
            #         for depth in range(1, 5):  # Change 5 to the desired square size
            #             dfs(node, node, depth, [node])

            #     return squares_count
            
            # triangle
            mb, node, _ = h.shape # h: (batchsize, ch, in)
            asd = sum([adj[:, r, :, :] for r in range(self.num_edge_type)])
            ls = []
            for i in asd:
                # squares_count(i)
                l = []
                for s in range(i.shape[0]):
                    # Find the neighbors of the current node
                    row_i = i[s, :]
                    num_triangles = (row_i @ i @ row_i) // 2
                    l.append(num_triangles)
                ls.append(l)
            b = torch.tensor(ls)[:,:,None]
            h = torch.cat((h,b), -1)
            m = self.linear_edge1(h) # h_node: (batchsize, ch, out*4)
            m = m.reshape(mb, node, self.out_ch, self.num_edge_type) # batch, ch, out, 4
            m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
            hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
            hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)
            return hr

        # PNA
        elif atype == 6:
            mb, node, _ = h.shape # h: (batchsize, ch, in)
            h_node = self.linear_node1(h)
            h_node = F.relu(h_node)
            h_node = self.linear_node(h_node) # h_node: (batchsize, ch, out)
            m = self.linear_edge(h) # h_node: (batchsize, ch, out*4)
            m = F.relu(m)
            m = self.linear_edge2(m)
            m = m.reshape(mb, node, self.out_ch, self.num_edge_type) # batch, ch, out, 4
            m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
            hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
            hr = hr.mean(dim=1)   # hr: (batchsize, node, ch)
            return hr+h_node

        # GIN
        elif atype == 7:
            mb, node, _ = h.shape 
            h_node = self.linear_node(h)
            h_node = F.relu(h_node)
            m = self.linear_edge(h)
            m = m.reshape(mb, node, self.out_ch, self.num_edge_type)
            m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
            hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
            hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)
            return self.output((1+self.eps)*hr + h_node)
        else:
            raise NotImplementedError
    

class EnergyFunc(nn.Module):
    def __init__(self, n_atom_type, hidden, num_edge_type=4, atype=0, swish=True, depth=2, add_self=False, dropout=0):
        super(EnergyFunc, self).__init__()
        self.depth = depth
        self.graphconv1 = GraphConv(n_atom_type, hidden, num_edge_type, std=1, bound=False, add_self=add_self)
        self.graphconv = nn.ModuleList(GraphConv(hidden, hidden, num_edge_type, std=1e-10, add_self=add_self) for i in range(self.depth))
        self.swish = swish
        self.dropout = dropout
        self.atype = atype
        self.linear = nn.Linear(hidden, 1)
# EnergyFunc(10,64).to('cuda')
    def forward(self, adj, h):
        h = h.permute(0, 2, 1)
        out = self.graphconv1(adj, h, self.atype)
        out = F.dropout(out, p=self.dropout, training=self.training)
        if self.swish:
            out = swish(out)
        else:
            out = F.leaky_relu(out, negative_slope=0.2)
        for i in range(self.depth):
            out = self.graphconv[i](adj, out, self.atype)    
            out = F.dropout(out, p=self.dropout, training=self.training)
            if self.swish:
                out = swish(out)
            else:
                out = F.relu(out)
        out = out.sum(1) # (batchsize, node, ch) --> (batchsize, ch)
        out = self.linear(out)
        return out # Energy value (batchsize, 1)
