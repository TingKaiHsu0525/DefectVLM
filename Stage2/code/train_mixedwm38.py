from header import *
from datasets import *
from model import *
from config import *

from torch.utils.tensorboard import SummaryWriter


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)

    # model configurations
    parser.add_argument('--imagebind_ckpt_path', type=str)
    parser.add_argument('--vicuna_ckpt_path', type=str)
    parser.add_argument('--delta_ckpt_path', type=str)
    parser.add_argument('--lora_ckpt_path', type=str)

    # ImageBind-LoRA
    parser.add_argument('--imagebind_lora_ckpt_dir', type=str, default=None)
    parser.add_argument('--imagebind_lora_rank', type=int, default=4)
    parser.add_argument('--imagebind_lora_modalities', nargs='+', type=str, default=['vision','text'],
                        choices=['vision','text','audio','thermal','depth','imu'])
    parser.add_argument('--imagebind_lora_layer_idxs_vision', nargs='+', type=int, default=None)
    parser.add_argument('--imagebind_lora_layer_idxs_text', nargs='+', type=int, default=None)
    parser.add_argument('--imagebind_lora_postfix', type=str, default="_mixed_last")
    parser.add_argument('--imagebind_load_heads', action='store_true')
    parser.add_argument('--imagebind_load_postprocessors', action='store_true')


    parser.add_argument('--max_tgt_len', type=int)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_root_path', type=str)

    # validation configurations
    parser.add_argument('--do_eval', action='store_true', help='Enable validation')
    parser.add_argument('--eval_every', type=int, default=1, help='Run validation every N epochs')
    parser.add_argument('--save_best', action='store_true', help='Save best checkpoint based on val metric')

    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))

    local_rank_env = os.getenv('LOCAL_RANK', None)
    if local_rank_env is not None:
        args['local_rank'] = int(local_rank_env)
    else:
        args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()

    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])


def build_directory(path):
    if os.path.exists(path):
        return
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def evaluate(agent, val_iter):
    # agent.eval_model() should return (loss, token_acc) for a batch
    agent.model.eval()

    loss_sum = 0.0
    gen_loss_sum = 0.0
    pixel_loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0

    for batch in val_iter:
        loss, gen_loss, loss_pixel, token_acc = agent.eval_model(batch)
        gen_loss_sum += float(gen_loss)
        pixel_loss_sum += float(loss_pixel)
        loss_sum += float(loss)
        acc_sum += float(token_acc)
        n_batches += 1
        del batch

    # Aggregate across all ranks
    device = torch.device('cuda', torch.cuda.current_device())
    t = torch.tensor([loss_sum, gen_loss_sum, pixel_loss_sum, acc_sum, n_batches], device=device, dtype=torch.float32)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    # Global averages
    loss_avg = (t[0] / torch.clamp(t[4], min=1.0)).item()
    gen_loss_avg = (t[1] / torch.clamp(t[4], min=1.0)).item()
    pixel_loss_avg = (t[2] / torch.clamp(t[4], min=1.0)).item()
    acc_avg = (t[3] / torch.clamp(t[4], min=1.0)).item()
    return loss_avg, gen_loss_avg, pixel_loss_avg, acc_avg


def main(**args):
    config_env(args)
    args['ds_config_path'] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )

    # TensorBoard 
    is_rank0 = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
    writer = SummaryWriter(args['log_path']) if (args['log_path'] and is_rank0) else None

    # load datasets 
    train_data, train_iter, train_sampler, val_data, val_iter, val_sampler = load_MixedWM38_datasets(args)

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps

    agent = load_model(args)
    torch.distributed.barrier()

    best_val_acc = 0

    # begin to train
    pbar = tqdm(total=length)
    current_step = 0

    for epoch_i in tqdm(range(args['epochs'])):
        # set epoch for distributed sample
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch_i)
        if val_sampler is not None and hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch_i)

        agent.model.train()
        for batch in train_iter:
            loss, gen_loss, loss_pixel, metrics = agent.train_model(
                batch,
                current_step=current_step,
                pbar=pbar
            )

            train_loss = float(loss)
            train_token_acc = float(metrics)

            if writer is not None:
                writer.add_scalar("train/loss", train_loss, current_step)
                writer.add_scalar("train/gen_loss", float(gen_loss), current_step)
                writer.add_scalar("train/pixel_loss", float(loss_pixel), current_step)
                writer.add_scalar("train/token_acc", train_token_acc, current_step)

            del batch
            current_step += 1

        # validation
        if args.get('do_eval', False) and ((epoch_i + 1) % int(args.get('eval_every', 1)) == 0):
            torch.distributed.barrier()
            val_loss, val_gen_loss, val_pixel_loss, val_acc = evaluate(agent, val_iter)

            if is_rank0:
                logging.info(f"[Eval] epoch={epoch_i+1} Total val_loss={val_loss:.6f} val_gen_loss={val_gen_loss:.6f} val_pixel_loss={val_pixel_loss:.6f} val_token_acc={val_acc:.6f}")
                if writer is not None:
                    # Log validation metrics using global step to align with training curves.
                    writer.add_scalar("val/loss", float(val_loss), current_step)
                    writer.add_scalar("val/gen_loss", float(val_gen_loss), current_step)
                    writer.add_scalar("val/pixel_loss", float(val_pixel_loss), current_step)
                    writer.add_scalar("val/token_acc", float(val_acc), current_step)
                    writer.flush()

            # Save best
            if args.get('save_best', False) and (val_acc > best_val_acc):
                best_val_acc = val_acc
                torch.distributed.barrier()
                if is_rank0:
                    agent.save_model(args['save_path'], 0)
                    logging.info(f"[Best] epoch={epoch_i+1} best_val_token_acc={best_val_acc:.6f} saved")

        # save at the end of each epoch
        torch.distributed.barrier()
        if is_rank0 and (not args.get('save_best', False)):
            agent.save_model(args['save_path'], 0)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    args['layers'] = [7, 15, 23, 31]
    main(**args)
