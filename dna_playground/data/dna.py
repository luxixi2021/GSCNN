from typing import List, Tuple, Callable, Optional
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import re
import random


class DNADatasetImpl(Dataset):
    def __init__(
        self,
        npz_path: Path,
        repeat: int = 1,
        augmentation: bool = False,
    ):
        super().__init__()

        self.repeat = repeat
        self.augmentation = augmentation

        data = np.load(npz_path)

        self.seq_ids = data["seq_ids"]
        self.seqs = data["seqs"]
        self.type_ids = data["type_ids"]
        self.tmp_ids = data["tmp_ids"]
        self.annos = data["annos"]

    def __len__(self):
        return len(self.seq_ids) * self.repeat

    def __getitem__(self, index: int):
        index = index % len(self.seq_ids)
        seq = self.seqs[index]
        type_ids = self.type_ids[index]
        annos = self.annos[index]

        end_indices = np.where(type_ids == 5)[0].tolist()
        end_indices = [item+1 for item in end_indices]
        start_indices = [0] + end_indices
        lengths = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            lengths.append(end_idx - start_idx)
        max_length = max(*lengths)
        h_seqs = []
        h_type_ids = []

        for start_idx, end_idx in zip(start_indices, end_indices):
            length = end_idx - start_idx
            h_seqs.append(np.pad(
                seq[start_idx:end_idx], (1, max_length - length),
                mode="constant",
                constant_values=0,
            ))
            h_type_ids.append(np.pad(
                type_ids[start_idx:end_idx], (1, max_length - length),
                mode="constant",
                constant_values=0,
            ))


        h_seqs = np.stack(h_seqs, axis=0)
        h_type_ids = np.stack(h_type_ids, axis=0)
        tmp = self.tmp_ids[index]
        h_seqs = torch.from_numpy(h_seqs)
        h_type_ids = torch.from_numpy(h_type_ids)
        h_seqs = h_seqs.float()
        h_type_ids = h_type_ids.float()                
        tmp = torch.tensor(tmp).float()
        if self.augmentation:
            indices = np.random.permutation(h_seqs.shape[0])
            h_seqs = h_seqs[indices]
            h_type_ids = h_type_ids[indices]

        return h_seqs, h_type_ids, tmp, annos

class DNADataset(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 16,
        num_workers: int = 4,
        repeat: int = 1,
        use_seq: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root_path = Path(root_path)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.repeat = repeat
        self.use_seq = use_seq
        self.train_npz_path = self.root_path / "train.npz"
        self.val_npz_path = self.root_path / "val.npz"

        
    def prepare_data(self):
        if self.train_npz_path.exists() and self.val_npz_path.exists():
            return
        seed = 0 #0-9
        with open(self.root_path / "Soyseq.csv") as f:
            lines = f.readlines()
        lines = map(lambda x: x.strip(), lines)
        lines = filter(lambda x: len(x) > 0, lines)
        lines = map(lambda x: x.split(","), lines)
        lines = list(lines)

        annos = pd.read_csv(self.root_path / "SoyBean_pheno.csv", index_col=0)
        annoIndex=annos.index
        annos = annos.to_numpy(dtype=np.float32)

        for col in range(annos.shape[1]):
            scaler = StandardScaler()
            annos[:, col:col + 1] = scaler.fit_transform(annos[:, col:col + 1])

        assert len(lines) == len(annos), (len(lines), len(annos))

        vocabs = [
            f"{a}{b}"
            for a in ["H", "L", "M"]
            for b in ["K", "W", "J", "Y", "N"]
        ]

        #vocabs = ["H", "L", "M"]
        vocabs = {
            k: i + 1
            for i, k in enumerate(vocabs)
        }

        type_vocabs = ["K", "W", "J", "Y", "N"]
        #type_vocabs = ["J", "N"]
        type_vocabs = {
            k: i + 1
            for i, k in enumerate(type_vocabs)
        }

        (
            train_lines, val_lines, train_annos_index,  val_annos_index,
            train_annos, val_annos,
        ) = train_test_split(lines, annoIndex, annos, test_size=0.2, random_state=seed)
        print(val_annos_index)

        train_seq_ids = []
        train_seqs = []
        train_type_ids = []
        train_tmp_ids = []
        aug_train_annos=[]
        for aug in range(2):
            for (train_seq_id, train_raw_seq), train_anno_id, train_anno in zip(train_lines, train_annos_index, train_annos):
                #print('seq',train_raw_seq)
                assert train_seq_id == train_anno_id, (train_seq_id, train_anno_id)
                assert len(train_raw_seq) % 2 == 0
                train_raw_seq[-1] == "N"
                train_seq_ids.append(train_seq_id+str(aug))
                aug_train_annos.append(train_anno)
                train_seq = []
                train_type_id = []
                train_tmp_ids.append(int(train_anno_id.split('_')[-1]))
                
                if aug == 0:
                    for i in range(0, len(train_raw_seq), 2):
                        train_sub = train_raw_seq[i:i + 2]
                        train_seq.append(vocabs[train_sub[:2]])
                        train_type_id.append(type_vocabs[train_sub[1:]])
                    train_seqs.append(train_seq)
                    train_type_ids.append(train_type_id)
                else:
                    seq_list = re.split(r'([N])',train_raw_seq)
                    seq_list.append('')
                    seq_list=[''.join(i) for i in zip(seq_list[0::2],seq_list[1::2])][0:-1]
                    seq_aug=seq_list[aug:aug+1]
                    seq_list.pop(aug)
                    newseq_list=seq_aug+seq_list
                    train_new_seq = ''.join(newseq_list)                    
                    for i in range(0, len(train_new_seq), 2):
                        train_sub = train_new_seq[i:i + 2]
                        train_seq.append(vocabs[train_sub[:2]])
                        train_type_id.append(type_vocabs[train_sub[1:]])
                    train_seqs.append(train_seq)
                    train_type_ids.append(train_type_id)

        train_seq_ids = np.asarray(train_seq_ids).reshape(-1,1)
        train_seqs = np.asarray(train_seqs)
        train_type_ids = np.asarray(train_type_ids)
        aug_train_annos = np.asarray(aug_train_annos)

        val_seq_ids = []
        val_seqs = []
        val_type_ids = []
        val_tmp_ids = []

        for (val_seq_id, val_raw_seq), val_anno_id in zip(val_lines, val_annos_index):
            assert val_seq_id == val_anno_id, (val_seq_id, val_anno_id)
            assert len(val_raw_seq) % 2 == 0
            val_raw_seq[-1] == "N"
            val_seq_ids.append(val_seq_id)
            val_tmp_ids.append(int(val_anno_id.split('_')[-1]))
            val_seq = []
            val_type_id = []
            for i in range(0, len(val_raw_seq), 2):
                val_sub = val_raw_seq[i:i + 2]
                val_seq.append(vocabs[val_sub[:2]])
                val_type_id.append(type_vocabs[val_sub[1:]])

            val_seqs.append(val_seq)
            val_type_ids.append(val_type_id)

        val_seqs = np.asarray(val_seqs)
        val_type_ids = np.asarray(val_type_ids)

        np.savez_compressed(
            self.train_npz_path,
            seq_ids=train_seq_ids,
            seqs=train_seqs,
            type_ids=train_type_ids,
            tmp_ids=train_tmp_ids,
            annos=aug_train_annos,
        )

        np.savez_compressed(
            self.val_npz_path,
            seq_ids=val_seq_ids,
            seqs=val_seqs,
            type_ids=val_type_ids,
            tmp_ids=val_tmp_ids,
            annos=val_annos,
        )

    def train_dataloader(self):
        if self.use_seq:
            dataset = SeqDNADatasetImpl(self.train_npz_path)
        else:
            dataset = DNADatasetImpl(
                self.train_npz_path,
                augmentation=False,#online_aug=True; offline_aug=False
                repeat=self.repeat,
            )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.use_seq:
            dataset = SeqDNADatasetImpl(self.val_npz_path)
        else:
            dataset = DNADatasetImpl(
                self.val_npz_path,
                augmentation=False,
                repeat=self.repeat,
            )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    from tqdm import tqdm

    dataset = DNADataset("data")
    dataset.prepare_data()

    for batch in tqdm(dataset.train_dataloader()):
        ...
    for batch in tqdm(dataset.val_dataloader()):
        ...


if __name__ == "__main__":
    main()
