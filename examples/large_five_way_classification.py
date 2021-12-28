
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

@registry.register_task('large_five_way_classification', num_labels=5)
class LargeFiveWayClassificationDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 data_fold=None,
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
        
        for label in labels:
            name, label_int = label.split()
            if name in limited_names:
                self.labels.append(name)
                self.labels_int.append(int(label_int))
        #print(self.labels)
        self.ptms = {}
        
        np.random.seed(42)
        order = np.random.permutation(len(self.labels))
        data_fold = int(data_fold)
        interval = len(self.labels) // 10
        test_sample = order[data_fold * interval:(data_fold + 1) * interval]

        from sklearn.model_selection import train_test_split
        
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        for i in range(len(self.labels)):
            if i in test_sample:
                test_data.append(self.labels[i])
                test_labels.append(self.labels_int[i])
            else:
                train_data.append(self.labels[i])
                train_labels.append(self.labels_int[i])


        if split == 'train':
            self.labels = train_data
            self.labels_int = train_labels
        else:
            self.labels = test_data
            self.labels_int = test_labels     

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
    'large_five_way_classification', 'transformer', ProteinBertForSequenceClassification, force_reregister=True)


if __name__ == '__main__':
    """ To actually run the task, you can do one of two things. You can
    simply import the appropriate run function from tape.main. The
    possible functions are `run_train`, `run_train_distributed`, and
    `run_eval`. Alternatively, you can add this dataset directly to
    tape/datasets.py.
    """
    from tape.main import run_train
    run_train()