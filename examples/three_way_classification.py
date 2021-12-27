
from typing import Union, List, Tuple, Any, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pickle

from tape.datasets import LMDBDataset, pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from tape import ProteinBertForSequenceClassification

@registry.register_task('three_way_classification', num_labels=3)
class ThreeWayClassificationDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 data_label_set = None,
                 data_fold = None,
                 in_memory: bool = False):

        if split not in ('train', 'valid'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        self.data_path = data_path
        with open(data_path / 'label.txt', 'r') as f:
            labels = f.readlines()
        
        limited_names = pickle.load(open(data_path / 'names.pkl', 'rb'))
        self.labels = []
        self.labels_int = []

        if data_label_set is not None:
            limited_names = pickle.load(open(data_label_set, 'rb'))
            limited_names = list(map(lambda x:x.split("/")[-2], limited_names))
        print(limited_names)
                    
        
        for label in labels:
            name, label_int = label.split()
            if name in limited_names:
                self.labels.append(name)
                self.labels_int.append(int(label_int))
        #print(self.labels)
        self.ptms = {}
        with open(data_path / 'pm.out', 'r') as f:
            ptms = f.readlines()
        res = {self.labels[i]: self.labels_int[i] for i in range(len(self.labels))}
        for ptm in ptms:
            self.ptms[ptm.split()[0]] = float(ptm.split()[1].strip())

        for label in list(res.keys()):
            if self.ptms[label] <= 0.8:
                del res[label]
        
        self.labels = list(res.keys())
        self.labels_int = list(res.values())

        from sklearn.model_selection import train_test_split
        if data_fold is not None:
            if split == "train":
                fold = pickle.load(open(data_fold, 'rb'))["train_ids"]
                print(len(fold))
                print(len(self.labels))
                print(len(self.labels_int))
                self.labels = [self.labels[i] for i in fold]
                self.labels_int = [self.labels_int[i] for i in fold]
            else:
                fold = pickle.load(open(data_fold, 'rb'))["test_ids"]
                self.labels = [self.labels[i] for i in fold]
                self.labels_int = [self.labels_int[i] for i in fold]
        else:
            if split == 'train':
                self.labels, _, self.labels_int, _ = train_test_split(self.labels, self.labels_int, train_size=0.8, random_state=42)
            else:
                _, self.labels, _, self.labels_int = train_test_split(self.labels, self.labels_int, train_size=0.8, random_state=42)        

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        name = self.labels[index]
        labels = self.labels_int[index]
        fasta_path = self.data_path / 'seqs' / f"{name}.fasta"
        with open(fasta_path, 'r') as f:
            fasta = f.readlines()[1:]
            fasta = ''.join(fasta).replace("\n", "")
        token_ids = self.tokenizer.encode(fasta)
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        #labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)
        
        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0, max_length=1280))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0, max_length=1280))
        #ss_label = torch.from_numpy(pad_sequences(ss_label, -1))
        ss_label = torch.from_numpy(np.asarray(ss_label))
        
        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output

registry.register_task_model(
    'three_way_classification', 'transformer', ProteinBertForSequenceClassification, force_reregister=True)


if __name__ == '__main__':
    """ To actually run the task, you can do one of two things. You can
    simply import the appropriate run function from tape.main. The
    possible functions are `run_train`, `run_train_distributed`, and
    `run_eval`. Alternatively, you can add this dataset directly to
    tape/datasets.py.
    """
    from tape.main import run_train
    run_train()