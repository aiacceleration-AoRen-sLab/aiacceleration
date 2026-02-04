import os
import random
from typing import List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
import os


class PruneDataset:
    '''
    1. load and process calibration dataset
    2. load and process evaluation dataset
    '''
    def __init__(self, args, dataset_name, tokenizer, seq_len,
                 local_path=None, eval_mode=False):
        '''
        :param args: -->SimpleNamespace
        :param dataset_name: c4
        :param local_path: load data from local directory
        param tokenizer & seq_len: parameters to get dataloader
        '''
        self.args = args
        self.logger = args.logger
        assert dataset_name is not None
        self.dataset_name = dataset_name.lower()  # c4, wikitext2
        self.local_path = local_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eval_mode = eval_mode
        self.data_cache_dir = self.args.data_cache_dir
        if not os.path.exists(self.data_cache_dir):
            os.makedirs(self.data_cache_dir)
        # tokenize data
        self.dataloader = self.get_dataloader(self.tokenizer, self.seq_len)

    def load_data(self):
        """
        :return:eval_data if eval_mode else train_data
        """
        # Check if local path is set, if so, prioritize using local path
        if self.local_path:
            # Convert relative path to absolute path
            if not os.path.isabs(self.local_path):
                local_path_abs = os.path.abspath(self.local_path)
            else:
                local_path_abs = self.local_path
            
            # Check if local path exists
            if os.path.exists(local_path_abs):
                self.logger.info(f'Loading data from local directory: {local_path_abs}')
                try:
                    if self.dataset_name in ["c4"]:
                        if self.eval_mode:
                            return load_dataset(local_path_abs, split='validation')
                        return load_dataset(local_path_abs, split='train')
                    elif self.dataset_name in ["wikitext2", "wikitext"]:
                        # For wikitext dataset, check if there are subdirectories
                        wikitext_path = local_path_abs
                        # Check if there are subdirectories under wikitext directory, such as wikitext-2-raw-v1
                        if os.path.isdir(local_path_abs):
                            for item in os.listdir(local_path_abs):
                                item_path = os.path.join(local_path_abs, item)
                                if os.path.isdir(item_path) and 'wikitext' in item.lower():
                                    wikitext_path = item_path
                                    self.logger.info(f'Found wikitext subdirectory: {wikitext_path}')
                                    break
                        if self.eval_mode:
                            return load_dataset(wikitext_path, 'wikitext-2-raw-v1', split='test')
                        else:
                            return load_dataset(wikitext_path, 'wikitext-2-raw-v1', split='train')
                    else:
                        raise NotImplementedError(f'Dataset {self.dataset_name} not supported')
                except Exception as e:
                    self.logger.warning(f'Failed to load from local path {local_path_abs}: {e}')
                    self.logger.info('Falling back to remote dataset download')
            else:
                self.logger.warning(f'Local path does not exist: {local_path_abs}, falling back to remote dataset')
        
        # Support Huggingface mirror sites (for remote dataset download)
        # Check HF_MIRROR and HF_ENDPOINT environment variables
        hf_mirror = os.getenv('HF_MIRROR', None)
        hf_endpoint = os.getenv('HF_ENDPOINT', None)
        original_hf_endpoint = os.environ.get('HF_ENDPOINT')
        
        # Set HuggingFace mirror (prioritize using HF_MIRROR, if not available then use HF_ENDPOINT)
        if hf_mirror:
            # Set HF_ENDPOINT environment variable to let datasets library use mirror
            os.environ['HF_ENDPOINT'] = hf_mirror
            self.logger.info(f'Using HuggingFace mirror from HF_MIRROR: {hf_mirror}')
        elif hf_endpoint:
            # If HF_ENDPOINT is already set, use it
            self.logger.info(f'Using HuggingFace endpoint: {hf_endpoint}')
        
        try:
            # Try to download dataset from HuggingFace
            self.logger.info(f'Downloading {self.dataset_name} dataset from HuggingFace...')
            
            if self.dataset_name in ["c4"]:
                if self.eval_mode:
                    dataset = load_dataset('allenai/c4',
                                         data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                         split='validation')
                else:
                    dataset = load_dataset('allenai/c4',
                                          data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                                          split='train')
            elif self.dataset_name in ["wikitext2", "wikitext"]:
                if self.eval_mode:
                    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                else:
                    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            else:
                raise NotImplementedError(f'Dataset {self.dataset_name} not supported')
            
            self.logger.info(f'Successfully loaded {self.dataset_name} dataset')
            return dataset
            
        except Exception as e:
            self.logger.error(f'Failed to download dataset from HuggingFace: {e}')
            if hf_mirror:
                self.logger.error(f'Mirror {hf_mirror} may not be accessible or dataset download failed')
            raise
        finally:
            # Restore original HF_ENDPOINT setting
            if original_hf_endpoint is not None:
                os.environ['HF_ENDPOINT'] = original_hf_endpoint
            elif 'HF_ENDPOINT' in os.environ and hf_mirror:
                # If mirror was set before, delete it now
                del os.environ['HF_ENDPOINT']

    def get_dataloader(self, tokenizer, seq_len) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''
        segment and tokenize data
        '''
        model_name = tokenizer.name_or_path.split("/")[-1]
        if self.eval_mode:
            cache_dataloader = f'{self.data_cache_dir}/eval_{self.dataset_name}_{model_name}_seqlen{seq_len}.cache'
            if os.path.exists(cache_dataloader):
                self.logger.info(f"load eval processed data from {cache_dataloader}")
                return torch.load(cache_dataloader, weights_only=True)
        else:
            cache_dataloader = f'{self.data_cache_dir}/cali_{self.dataset_name}_{model_name}_seqlen{self.seq_len}_nsamples{self.args.nsamples}.cache'
            # cache_dataloader = f'{self.data_cache_dir}/cali_{self.dataset_name}_{model_name}_seqlen{self.seq_len}_nsamples{self.args.nsamples}_seed{self.args.seed}.cache'
            if os.path.exists(cache_dataloader):
                self.logger.info(f"load calibration processed data from {cache_dataloader}")
                return torch.load(cache_dataloader, weights_only=True)  # single-->calibration processed data

        data = self.load_data()
        # process
        if self.eval_mode:
            return self.process_eval_data(data, tokenizer, seq_len, cache_dataloader)
        return self.process_calibration_data(data, tokenizer, seq_len, cache_dataloader)

    def process_calibration_data(self, train_data, tokenizer, seq_len, cache_dataloader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        random.seed(self.args.seed)
        train_loader = []
        if self.dataset_name == "c4":
            for _ in tqdm(range(self.args.nsamples), desc="Processing calibration data"):
                while True:
                    i = random.randint(0, len(train_data) - 1)  # i-th sentence

                    train_enc = tokenizer(train_data[i]['text'], return_tensors='pt')

                    if train_enc['input_ids'].shape[1] > seq_len:  # trainenc
                        break
                    else:
                        continue

                j = random.randint(0, train_enc['input_ids'].shape[1] - seq_len - 1)
                inp = train_enc.input_ids[:, j:(j + seq_len)]  # Get token sequence

                tar = inp.clone()
                tar[:, :-1] = -100
                train_loader.append((inp, tar))
            self.logger.info(f"processing {self.dataset_name} calibration data finished")
            try:
                torch.save(train_loader, cache_dataloader)
            except:
                pass
            return train_loader

        elif self.dataset_name in ["wikitext2", "wikitext"]:
            train_enc = tokenizer(" ".join(train_data['text']), return_tensors='pt')
            for i in tqdm(range(self.args.nsamples), desc="Processing calibration data"):
                j = random.randint(0, train_enc['input_ids'].shape[1] - seq_len - 1)
                inp = train_enc.input_ids[:, j:(j + seq_len)]
                tar = inp.clone()
                tar[:, :-1] = -100
                train_loader.append((inp, tar))
            self.logger.info(f"processing {self.dataset_name} calibration data finished")
            try:
                torch.save(train_loader, cache_dataloader)
            except:
                pass
            return train_loader
        else:
            raise NotImplementedError

    def process_eval_data(self, test_data, tokenizer, seq_len, cache_dataloader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''
        nsamples: test_data samples // seq_len, may not be self.args.nsamples (128)
        '''
        self.logger.info("start loading test loader")
        if self.dataset_name == "c4":
            test_enc = tokenizer("\n\n".join(test_data[:1100]['text']), return_tensors='pt')
        elif self.dataset_name in ["wikitext2", "wikitext"]:
            test_enc = tokenizer("\n\n".join(test_data['text']), return_tensors='pt')
        else:
            raise NotImplementedError

        # tokenize data
        nsamples = test_enc.input_ids.numel() // seq_len
        test_loader = []
        for i in tqdm(range(nsamples), desc="Processing eval data"):
            j = i + 1
            inp = test_enc.input_ids[:, (i * seq_len):(j * seq_len)]
            tar = inp.clone()
            tar[:, :-1] = -100
            test_loader.append((inp, tar))
        self.logger.info(
            f"{self.dataset_name} testenc numel: {test_enc.input_ids.numel()}, seq_len: {seq_len}, test_loader length/nsamples: {nsamples}")
        self.logger.info(f"processing {self.dataset_name} test loader finished")
        try:
            torch.save(test_loader, cache_dataloader)
        except:
            pass
        return test_loader

    def stack_loaders(self, dataloader: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        '''
        :param dataloader: # [(loader,tar), ..., (loader,tar)]-->list len: nsamples(128)
        :return: torch.Tenser: (128,1,seqlen)
        '''
        return torch.stack([loader[0] for loader in dataloader])

