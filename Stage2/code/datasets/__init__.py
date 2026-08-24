from header import *
from .samplers import DistributedBatchSampler
from .sft_dataset import *
from .mvtec import *
from .visa import VisaDataset
from . import all_supervised_with_cn
from .mixedwm38 import *
from .WM811k import *

'''
def get_tokenizer(model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    tokenizer.bos_token_id, tokenizer.eos_token_id = 1, 2
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
'''
def _build_distributed_loader(dataset, args, shuffle, drop_last, num_workers=8):
    # NOTE: Use RandomSampler for training and SequentialSampler for validation
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # NOTE: Keep your original "global batch size" logic for DistributedBatchSampler
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']

    batch_sampler = DistributedBatchSampler(
        sampler,
        batch_size,
        drop_last,
        rank,
        world_size
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=dataset.collate,
        pin_memory=False
    )
    return sampler, loader



def load_sft_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = SupervisedDataset(args['data_path'], args['image_root_path'])

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=1,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler

def load_mvtec_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = MVtecDataset('../data/mvtec_anomaly_detection')

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=8,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler

##### Add NEW with MixedWM38 dataset #####
def load_MixedWM38_datasets(args):
    # Annotation files for MixedWM38 dataset
    train_json = os.path.join(args.get('data_root', "../data"), "train.json")
    val_json = os.path.join(args.get('data_root', "../data"), "val.json")

    train_data = MixedWM38Dataset(train_json)
    val_data = MixedWM38Dataset(val_json)

    # Train: shuffle=True, drop_last=True
    train_sampler, train_iter = _build_distributed_loader(
        train_data, args, shuffle=True, drop_last=True, num_workers=0
    )

    # Val: shuffle=False, drop_last=False
    val_sampler, val_iter = _build_distributed_loader(
        val_data, args, shuffle=False, drop_last=False, num_workers=0
    )

    return train_data, train_iter, train_sampler, val_data, val_iter, val_sampler

def load_WM811k_datasets(args):
    # Annotation files for WM811k dataset
    train_json = os.path.join(args.get('data_root', "../data/WM-811k"), "train.json")
    val_json = os.path.join(args.get('data_root', "../data/WM-811k"), "val.json")

    train_data = WM811kDataset(train_json)
    val_data = WM811kDataset(val_json)

    # Train: shuffle=True, drop_last=True
    train_sampler, train_iter = _build_distributed_loader(
        train_data, args, shuffle=True, drop_last=True, num_workers=8
    )

    # Val: shuffle=False, drop_last=False
    val_sampler, val_iter = _build_distributed_loader(
        val_data, args, shuffle=False, drop_last=False, num_workers=8
    )

    return train_data, train_iter, train_sampler, val_data, val_iter, val_sampler

def load_visa_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = VisaDataset('../data/VisA')

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=8,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler


def load_supervised_dataset_with_cn(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = all_supervised_with_cn.SupervisedDataset('../data/all_anomalygpt')

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=1,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler