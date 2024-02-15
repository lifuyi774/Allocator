import re, os,sys
import torch
import argparse
from Bio import SeqIO
import itertools
from collections import Counter
import pandas as pd
from model import GNNm
from graphData import mRNAGraphDataset,mRNAGraphDataset_1
from torch_geometric.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--input_fasta', default="test", help='input fasta')
parser.add_argument('--output_path', default="test", help='output path')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
opt = parser.parse_args()

n_classes = 6
device = opt.device #'cpu' 

def RNAFold_folding(fasta_file):
  # Compute structure
    os.system(f"RNAfold --noPS {fasta_file} > RNAfold_tmp_dot.fasta")
    file ='RNAfold_tmp_dot.fasta'
    with open(file) as f:
        records=f.read()

    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    # print(records)
    seq_dotbracket = {}
    for fasta in records:
        valueList=[]
        array = fasta.split('\n')
        sequence,dot_bracket =array[1].replace('U','T'),array[2]
        dot_bracket_list=dot_bracket.split()

        ev=float(dot_bracket_list[1].split('(')[1].split(')')[0]) #.replace('\U00002013', '-')
        valueList.append(dot_bracket_list[0])
        valueList.append(ev)
        seq_dotbracket[sequence]=valueList
  
    return seq_dotbracket

def Linear_folding(fasta_file):
  out_fasta_name = "Linearfold_dot"
  with open(fasta_file, "r") as handle:
    records = list(SeqIO.parse(handle, "fasta"))
  for row in records:
    # Get sequence string
    seq = str(row.seq)
    lnc=str(row.description)
    # Write a one-sequence fasta
    with open("LinearFold_tmp.fasta", "w") as ofile: 
      ofile.write(f">{lnc}\n{seq}\n")
    # Compute structure
    tmp_name='LinearFold_tmp.fasta'
    # os.system(f"cat {tmp_name} | /your path/LinearFold/bin/linearfold_v > LinearFold_tmp.dot")
    os.system(f"cat {tmp_name} | /home/xudongguo/Projects/LinearFold/bin/linearfold_v > LinearFold_tmp.dot")
    # Clean output
    out_file_name = "clean_tmp.dot"
    in_lines = open("LinearFold_tmp.dot","r").readlines()
    with open(out_file_name,"w") as out_file:
      for line in in_lines:
        if ">" in line:
          out_file.write(':'.join(line.split(":")[1:]).strip() + "\n")
        else:
          out_file.write(line)

    os.system("cat " + out_file_name + " >> " + out_fasta_name + ".fasta") 
  file ='Linearfold_dot.fasta'
  with open(file) as f:
      records1=f.read()

  if re.search('>', records1) == None:
      print('Error: the input file %s seems not in FASTA format!' % file)
      sys.exit(1)
  records1 = records1.split('>')[1:]
  # print(records)
  seq_dotbracket = {}
  for fasta in records1:
      valueList=[]
      array = fasta.split('\n')
      sequence,dot_bracket =array[1],array[2]
      dot_bracket_list=dot_bracket.split()

      ev=float(dot_bracket_list[1].split('(')[1].split(')')[0]) #.replace('\U00002013', '-')
      valueList.append(dot_bracket_list[0])
      valueList.append(ev)
      seq_dotbracket[sequence]=valueList
  return seq_dotbracket

def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen
def read_nucleotide_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[1] if len(header_array) >= 2 else '0'
        label_train = header_array[2] if len(header_array) >= 3 else 'training'
        sequence = re.sub('U', 'T', sequence) 
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences
def CKSNAP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = kw['order'] if kw['order'] != None else 'ACGT'

    encodings = {}
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    # header = ['#', 'label']
    # for g in range(gap + 1):
    #     for aa in aaPairs:
    #         header.append(aa + '.gap' + str(g))
    # encodings.append(header)

    for i in fastas:
        seq_key=i[1]
        name, sequence, label = i[0], i[1], i[2]
        # code = [name, label]
        code=[]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        # encodings.append(code)
        encodings[seq_key]=code
    return encodings
def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer
def Kmer(fastas, k=2, type="DNA", upto=False, normalize=True, **kw):
    # encoding = []
    encoding = {}
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        # encoding.append(header)
        for i in fastas:
            seq_key=i[1]
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            # code = [name, label]
            code=[]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding[seq_key]=code
            # encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        # encoding.append(header)
        for i in fastas:
            seq_key=i[1]
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            # code = [name, label]
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            # encoding.append(code)
            encoding[seq_key]=code
    return encoding


def predictor(model,test_loader,test_loader1,test_records,outputpath):
    model.eval()

    with torch.no_grad():

        y_pred_list2 = []
        t = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).to(device)
        dataloader_iterator = iter(test_loader1)
        for i, data in enumerate(test_loader):

            data.x=data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device)
            data.batch = data.batch.to(device)
            data.cksnap=data.cksnap.to(device)
            data.kmer=data.kmer.to(device)
            # targets = data.y.to(device)
            try:
                data1 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(test_loader1)
                data1 = next(dataloader_iterator)
            data1.x=data1.x.to(device)  
            data1.edge_index = data1.edge_index.to(device)
            data1.edge_attr = data1.edge_attr.to(device)
            data1.batch = data1.batch.to(device)
            
            out = model(data, data1)

            pred = (out > t).float() * 1
            # y_score_list2 += list(out)
            y_pred_list2 += list(pred)

        # y_score_tensor2 = torch.stack(y_score_list2)
        y_pred_tensor2 = torch.stack(y_pred_list2)
        # y_predScore=y_score_tensor2.cpu().numpy()
        y_predClass=y_pred_tensor2.cpu().numpy()
        resultDF=pd.DataFrame(y_predClass,columns=['Nucleus', 'Exosome', 'Cytosol', 'Ribosome', 'Membrane', 'ER'])
        resultDF=resultDF.astype(bool).astype(str)
        seqDes=[str(record.description) for record in test_records]
        resultDF.insert(0,'SequenceID',seqDes)
        resultDF.to_csv(outputpath+'/results.txt',index=False) 


def main(): 
   
    model = GNNm(n_features=10, hidden_dim=64, n_classes=6,
                    n_conv_layers=3,conv_type1="GIN",conv_type2="GIN",
                    dropout=0.1, batch_norm=True, batch_size=1)
    model.load_state_dict(torch.load('model/model.pth',map_location=device))
    model.to(device)
    with open(opt.input_fasta, "r") as handle1:
        test_records = list(SeqIO.parse(handle1, "fasta"))
    RNAfoldDict=RNAFold_folding(opt.input_fasta)   
    os.system(f"rm RNAfold_tmp_dot.fasta")
    LinearfoldDict=Linear_folding(opt.input_fasta)
    os.system(f"rm LinearFold_tmp.fasta")
    os.system(f"rm Linearfold_dot.fasta")
    os.system(f"rm clean_tmp.dot")
    os.system(f"rm LinearFold_tmp.dot")


    fastas= read_nucleotide_sequences(opt.input_fasta)
    kw = {'order': 'ACGT'}
    CKSNAP_Dict=CKSNAP(fastas=fastas,gap=5,**kw)
    Kmer_Dict=Kmer(fastas, k=5, type="RNA", upto=True, normalize=True, **kw)
    # fastas_encodings
    # path = 'Features_5mer_dict.pkl'
    # with open(path, 'wb') as handle:
    #     pickle.dump(Kmer_Dict, handle)
    # path1 = 'Features_CKSNAP_dict.pkl'
    # with open(path1, 'wb') as handle:
    #     pickle.dump(CKSNAP_Dict, handle)
    test_set = mRNAGraphDataset(test_records, RNAfoldDict,CKSNAP_Dict,Kmer_Dict)
    test_set1 = mRNAGraphDataset_1(test_records, LinearfoldDict)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,drop_last=True)
    test_loader1 = DataLoader(test_set1, batch_size=1, shuffle=False,drop_last=True)
    
    predictor(model,test_loader,test_loader1,test_records,opt.output_path)

if __name__ == "__main__":
    
    main()     

    