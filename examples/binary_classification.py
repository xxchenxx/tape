
from typing import Union, List, Tuple, Any, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pickle
import os
from tape.datasets import LMDBDataset, pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from tape import ProteinBertForSequenceClassification

@registry.register_task('binary_classification', num_labels=5)
class BinaryClassificationDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 fasta_root = None,
                 split_file = None,
                 in_memory: bool = False):

        if split not in ('train', 'valid'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        self.data_path = data_path
        with open(data_path / split_file, 'rb') as f:
            split_file = pickle.load(f)
        
        if split == 'train':
            names = split_file['train_names']
            labels = split_file['train_labels']
        else:
            names = split_file['test_names']
            labels = split_file['test_labels']

        fasta_file_paths = []
        for name in names: 
            fasta_file_path = os.path.join(fasta_root, name + ".fasta")                
            fasta_file_paths.append(fasta_file_path)
            
        self.labels = labels
        self.fasta_file_paths = fasta_file_paths
                             
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        fasta_path = self.fasta_file_paths[index]
        with open(fasta_path, 'r') as f:
            fasta = f.readlines()[1:]
            fasta = ''.join(fasta).replace("\n", "")
        token_ids = self.tokenizer.encode(fasta)
        input_mask = np.ones_like(token_ids)
        
        return token_ids, input_mask, int(label)

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
    'binary_classification', 'transformer', ProteinBertForSequenceClassification, force_reregister=True)
from tape import UniRepForSequenceClassification
from tape import ProteinResNetForSequenceClassification
registry.register_task_model(
    'binary_classification', 'unirep', UniRepForSequenceClassification, force_reregister=True)
registry.register_task_model(
    'binary_classification', 'resnet', ProteinResNetForSequenceClassification, force_reregister=True)

if __name__ == '__main__':
    """ To actually run the task, you can do one of two things. You can
    simply import the appropriate run function from tape.main. The
    possible functions are `run_train`, `run_train_distributed`, and
    `run_eval`. Alternatively, you can add this dataset directly to
    tape/datasets.py.
    """
    from tape.main import run_train
    run_train()