import re
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset


# Word to index
word_to_ix = {"X": 0, "A": 1, "G": 2, "C": 3, "T": 4, "U": 4}
word_to_ix1 = {"<PAD>": [0,0,0,0,0], "A": [0,1,0,0,0], "G": [0,0,1,0,0], "C": [0,0,0,1,0], "T": [0,0,0,0,1], "U": [0,0,0,0,1]}
# Index to word
ix_to_word = {v: k for k, v in word_to_ix.items()}

# Tag to index
tag_to_ix = {"<PAD>": 0, ".": 1, "(": 2, ")": 3}
# Index to tag
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

N_to_EIIP = {"<PAD>": 0, "A": 0.1260, "G": 0.0806, "C": 0.1340, "T": 0.1335, "U": 0.1335}
N_to_NCP = {"<PAD>": [0,0,0], "A": [1,1,1], "G": [1,0,0], "C": [0,1,0], "T": [0,0,1], "U": [0,0,1]}

label_names = ['Nucleus', 'Exosome', 'Cytosol', 'Ribosome', 'Membrane', 'ER']
# Labels to index
label_to_ix = {'Nucleus': 0, 'Exosome': 1, 'Cytosol': 2, 'Ribosome': 3, 'Membrane': 4, 'ER': 5}

def prepare_sequence(seq,to_ix):
    idxs=[]
    for j,char in enumerate(seq):
        ANF=[seq[0:j+1].count(seq[j])/(j+1)]
        subidx=to_ix[char]+[N_to_EIIP[char]]+N_to_NCP[char]+ANF
        idxs.append(subidx)

    return torch.tensor(idxs, dtype=torch.float)

def dotbracket_to_graph(dotbracket):
    G = nx.Graph()
    bases = []

    for i, c in enumerate(dotbracket):
        if c == '(':
            bases.append(i)
        elif c == ')':
            neighbor = bases.pop()
            G.add_edge(i, neighbor, edge_type='base_pair')
        elif c == '.':
            G.add_node(i)
        else:
            print("Input is not in dot-bracket notation!")
            return None

        if i > 0:
            G.add_edge(i, i - 1, edge_type='adjacent')
    return G

class mRNAGraphDataset(InMemoryDataset):
    def __init__(self, records, foldings,featuresDict,featuresDict1):

        super().__init__()
        data_list = []
 
        for row in records:
            # Get sequence string
            seq_str = str(row.seq).replace('U','T')
            # Sequence string to tensor embedding
            sequence = prepare_sequence(seq_str, word_to_ix1)
            # Get label string
            # label_embedded = str(row.id)[:6]
            
            # Label string to label embedding
            # label_embedded = list(label_embedded)
            # Convert to int list
            # for i in range(len(label_embedded)):
            #     label_embedded[i] = int(label_embedded[i])

            # Get dot bracket string through sequence string
            dot_bracket_string = foldings[seq_str][0]

            # Dot bracket to graph
            g = dotbracket_to_graph(dot_bracket_string)

            # Sequence embedding
            seq_str=re.sub('[^ACGTU-]', '-', seq_str.upper())
            seq_str=re.sub('U', 'T', seq_str)


            x=sequence
            # Label embedding
            # y = y.view(1,6)


            # Get edge_attr and edge_index
            edges = list(g.edges(data=True))
            # One-hot encoding of the edge type
            edge_attr = torch.Tensor([[0, 1] if e[2]['edge_type'] == 'adjacent' else [1, 0] for e in edges])
            edge_index = torch.LongTensor(list(g.edges())).t().contiguous()

            # Graph to Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)#, y=y
            data.cksnap = torch.tensor(featuresDict[seq_str],dtype=torch.float32)
            data.kmer = torch.tensor(featuresDict1[seq_str],dtype=torch.float32)
            # Append objects to list
            data_list.append(data)


        print(f"Number of data: {len(data_list)}")
            
        self.data, self.slices = self.collate(data_list)
class mRNAGraphDataset_1(InMemoryDataset):
    def __init__(self, records, foldings):

        super().__init__()

        data_list = []
        # label_embedded_list = []

        
        for row in records:
            # Get sequence string
            seq_str = str(row.seq)
            # Sequence string to tensor embedding

            sequence = prepare_sequence(seq_str, word_to_ix1)

            # Get label string
            # label_embedded = str(row.id)[:6]
            
            # # Label string to label embedding
            # label_embedded = list(label_embedded)
            # # Convert to int list
            # for i in range(len(label_embedded)):
            #     label_embedded[i] = int(label_embedded[i])


            # Get dot bracket string through sequence string
            dot_bracket_string = foldings[seq_str][0]

            # Dot bracket to graph
            g = dotbracket_to_graph(dot_bracket_string)
            
            x = sequence
            # Label embedding
            # y = torch.Tensor(label_embedded)
            # y = y.view(1,6)


            # Get edge_attr and edge_index
            edges = list(g.edges(data=True))
            # One-hot encoding of the edge type
            edge_attr = torch.Tensor([[0, 1] if e[2]['edge_type'] == 'adjacent' else [1, 0] for e in edges])
            edge_index = torch.LongTensor(list(g.edges())).t().contiguous()

            # Graph to Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)#, y=y

            # Append objects to list
            data_list.append(data)
            

        print(f"Number of data: {len(data_list)}")

        self.data, self.slices = self.collate(data_list)  
