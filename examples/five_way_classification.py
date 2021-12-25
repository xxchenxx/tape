
from typing import Union, List, Tuple, Any, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

from tape.datasets import LMDBDataset, pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from tape import ProteinBertForSequenceClassification

@registry.register_task('five_way_classification', num_labels=5)
class FiveWayClassificationDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
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
        self.labels = []
        self.labels_int = []

        from sklearn.model_selection import train_test_split
        if split == 'train':
            labels, _ = train_test_split(labels, train_size=0.75, random_state=42)
        else:
            _, labels = train_test_split(labels, train_size=0.75, random_state=42)

        for label in labels:
            print(label)
            self.labels.append(label.split()[0])
            self.labels_int.append(int(label.split()[1].strip()))

        self.ptms = {}
        with open(data_path / 'pm.out', 'r') as f:
            ptms = f.readlines()
        res = {self.labels[i]: self.labels_int[i] for i in range(len(self.labels))}
        for ptm in ptms:
            self.ptms[ptm.split()[0]] = float(ptm.split()[1].strip())

        #for label in list(res.keys()):
        #    if self.ptms[label] <= 0.8:
        #        del res[label]
        
        self.labels = list(res.keys())
        self.labels_int = list(res.values())

        
        

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
        labels = np.asarray(labels, np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, -1))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output

registry.register_task_model(
    'five_way_classification', 'transformer', ProteinBertForSequenceClassification)


if __name__ == '__main__':
    """ To actually run the task, you can do one of two things. You can
    simply import the appropriate run function from tape.main. The
    possible functions are `run_train`, `run_train_distributed`, and
    `run_eval`. Alternatively, you can add this dataset directly to
    tape/datasets.py.
    """
    from tape.main import run_train
    run_train()