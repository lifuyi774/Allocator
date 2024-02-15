# Allocator: a graph neural network-based framework for mRNA subcellular localization prediction.
## Introduction

The asymmetrical distribution of expressed mRNAs tightly controls the precise synthesis of proteins with-in human cells. This non-uniform distribution, a cornerstone of developmental biology, plays a pivotal role in numerous cellular processes. To advance our comprehension of gene regulatory networks, it is essential to develop computational tools for accurately identifying the subcellular localizations of mRNAs. Howev-er, considering multi-localization phenomena remains limited in existing approaches, with none consider-ing the influence of RNA’s secondary structure. In this study, we introduce ‘Allocator’, a deep learning-based model that seamlessly integrates both sequence-level and structure-level information, significantly enhancing the prediction of mRNA multi-localization. Allocator equips four efficient feature extractors, each designed to handle different inputs. Two are tailored for sequence-based inputs, incorporating MLP, and self-attention mechanisms. The other two are specialized in processing structure-based inputs, employing graph neural networks. Benchmarking results underscore Allocator’s superiority over state-of-the-art methods, showcasing its strength in revealing intricate localization associations. Furthermore, we have developed a user-friendly web server for Allocator, enhancing its accessibility. You can explore Allo-cator through our web server, available at http://allocator.unimelb-biotools.cloud.edu.au/.

## Environment
* Anaconda
* python 3.7.13

## Dependency

* torch   1.12.1
* torch-cluster   1.6.0
* torch-geometric   2.1.0.post1
* torch-scatter   2.0.9
* torch-sparse    0.6.15
* numpy		1.21.6
* biopython	1.81
* RNAfold
* LinearFold

## Installation of RNAfold(Linux)

'''
wget -q https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_5_x/ViennaRNA-2.5.0.tar.'''
'''tar xfz ViennaRNA-2.5.0.tar.gz'''

'''cd /content/ViennaRNA-2.5.0'''

'''./configure '''

'''make'''

'''sudo make install
'''

## Installation of LinearFold(Linux)
'''
git clone https://github.com/LinearFold/LinearFold.git
'''
'''
cd LinearFold
'''
'''
make
'''

## Usage

To get the information the user needs to enter for help, run:
    python Allocator.py --help
 or
    python Allocator.py -h

as follows:

>python Allocator.py -h
usage: it's usage tip.
optional arguments:
  -h, --help            show this help message and exit
  --input_path          input fasta.
  --output_path         output path.
  --device              cpu or cuda

## Examples:

### Prediction:
```python Allocator.py --input_path input.fasta --output_path results --device cpu```
