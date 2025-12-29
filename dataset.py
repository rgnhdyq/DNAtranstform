import torch
from torch.utils.data import Dataset
import pandas as pd

def seq_to_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

class KMerTokenizer:
    def __init__(self, k=3):
        self.k = k
        self.vocab = {}
        self.build_vocab()

    def build_vocab(self):
        bases = ['A','C','G','T']
        from itertools import product
        kmers = [''.join(p) for p in product(bases, repeat=self.k)]
        self.vocab = {kmer:i+1 for i,kmer in enumerate(kmers)}  # 0 for padding

    def encode(self, seq, max_len):
        kmers = seq_to_kmers(seq, self.k)
        ids = [self.vocab.get(km, 0) for km in kmers]
        ids = ids[:max_len]
        return ids + [0]*(max_len-len(ids))

class EPIDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=200):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        enh = self.tokenizer.encode(row['enhancer_frag_seq'], self.max_len)
        pro = self.tokenizer.encode(row['promoter_frag_seq'], self.max_len)

        return {
            "enhancer": torch.tensor(enh, dtype=torch.long),
            "promoter": torch.tensor(pro, dtype=torch.long),
            "distance": torch.tensor(row['enhancer_distance_to_promoter'], dtype=torch.float),
            "label": torch.tensor(row['label'], dtype=torch.float)
        }
