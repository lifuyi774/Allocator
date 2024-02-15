import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv, GINConv, GATConv, Set2Set


class GNNm(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, n_conv_layers=3, dropout=0.1,
                 conv_type1="GIN",conv_type2="GIN",batch_norm=True,  batch_size=128):
        super(GNNm, self).__init__()

        #
        self.batch_size = batch_size
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.batch_norms1 = nn.ModuleList()
        self.batch_norms2 = nn.ModuleList()

        '''GNN1'''
        self.convs1.append(self.get_conv_layer(n_features, hidden_dim, conv_type=conv_type1))
        self.batch_norms1.append(nn.BatchNorm1d(hidden_dim))

        # for i in range(n_conv_layers - 1):
        #     self.convs1.append(self.get_conv_layer(hidden_dim, hidden_dim, conv_type=conv_type1))
        #     self.batch_norms1.append(nn.BatchNorm1d(hidden_dim))



        '''GNN2'''
        self.convs2.append(self.get_conv_layer(n_features, hidden_dim, conv_type=conv_type2))
        self.batch_norms2.append(nn.BatchNorm1d(hidden_dim))

        # for i in range(n_conv_layers - 1):
        #     self.convs2.append(self.get_conv_layer(hidden_dim, hidden_dim, conv_type=conv_type2))
        #     self.batch_norms2.append(nn.BatchNorm1d(hidden_dim))

        '''MLP'''
        self.mlp_cksnap = nn.Sequential(nn.Linear(96, 80, bias=True), 
                                        nn.Linear(80, 50, bias=True),
                                        nn.Linear(50, 128, bias=True) 
                                        )
        self.mlp_kmer = nn.Sequential(nn.Linear(1364, 800, bias=True), 
                                        nn.Linear(800, 300, bias=True),
                                        nn.Linear(300, 128, bias=True) 
                                        )
        
        self.cross_attention1 = nn.MultiheadAttention(hidden_dim*2, 4)
        self.cross_attention2 = nn.MultiheadAttention(hidden_dim*2,4)

        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128, 6)

        self.pooling1 = Set2Set(hidden_dim, processing_steps=10)
        self.pooling2 = Set2Set(hidden_dim, processing_steps=10)

        self.dropout = nn.Dropout(dropout)
        self.conv_type1 = conv_type1
        self.conv_type2 = conv_type2
        self.batch_norm = batch_norm


    def forward(self, data,data1): 

        g1, adj1, edge_attr1, batch1 = data.x, data.edge_index, data.edge_attr, data.batch
        g2, adj2, edge_attr2, batch2 = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        x_cksnap=data.cksnap
        x_kmer=data.kmer
        # GNN 1
        for i, con in enumerate(self.convs1):
            g1 = self.apply_conv_layer(con, g1, adj1, edge_attr1, conv_type=self.conv_type1)
            # g1 = self.batch_norms[i](g1) if self.batch_norm else g1
            # g1 = nn.functional.leaky_relu(g1)
            g1 = self.dropout(g1)


        g1 = self.pooling1(g1, batch1)
        g1 = self.dropout(g1)



        # GNN 2
        for i, con in enumerate(self.convs2):
            g2 = self.apply_conv_layer(con, g2, adj2, edge_attr2, conv_type=self.conv_type2)
            # g2 = self.batch_norms1[i](g2) if self.batch_norm else g2
            # g2 = nn.functional.leaky_relu(g2)
            g2 = self.dropout(g2)


        g2 = self.pooling2(g2, batch2)
        g2 = self.dropout(g2)

        ss=self.batch_size

        x1=x_cksnap.reshape((ss,-1))
        x1=self.mlp_cksnap(x1) 
        x1 = x1.unsqueeze(dim=1) 
        x1 = x1.permute(1, 0, 2)
        x1, _ = self.cross_attention1(x1,x1,x1) 
        x1 = x1.permute(1, 0, 2)
        x1=x1.squeeze(dim=1)




        
        
        x2=x_kmer.reshape((ss,-1))
        x2=self.mlp_kmer(x2)
        x2 = x2.unsqueeze(dim=1)
        x2 = x2.permute(1, 0, 2)
        x2, _ = self.cross_attention2(x2,x2,x2)
        x2 = x2.permute(1, 0, 2)
        x2=x2.squeeze(dim=1)


        x = torch.cat((g1,g2, x1, x2),dim=1)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output


    @staticmethod
    def get_conv_layer(n_input_features, n_output_features, conv_type="GCN"):
        if conv_type == "GCN":
            return GCNConv(n_input_features, n_output_features)
        elif conv_type == "GAT":
            return GATConv(n_input_features, n_output_features)
        elif conv_type == "MPNN":
            net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, n_input_features *
                                                                      n_output_features))
            return NNConv(n_input_features, n_output_features, net)
        elif conv_type == "GIN":
            net = nn.Sequential(nn.Linear(n_input_features, n_output_features), nn.ReLU(),
                                nn.Linear(n_output_features, n_output_features))
            return GINConv(net)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))

    @staticmethod
    def apply_conv_layer(conv, x, adj, edge_attr, conv_type="GCN"):
        if conv_type in ["GCN", "GAT", "GIN"]:
            return conv(x, adj)
        elif conv_type in ["MPNN"]:
            return conv(x, adj, edge_attr)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))