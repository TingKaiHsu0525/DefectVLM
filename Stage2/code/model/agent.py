from header import *
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

class DeepSpeedAgent:
    
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        self.load_stage_1_parameters(args["delta_ckpt_path"])

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.image_decoder.named_parameters():
            param.requires_grad = True

        for name, param in self.model.prompt_learner.named_parameters():
            param.requires_grad = True
        
        # soft prompt
        for name, param in self.model.soft_prompt.named_parameters():
            param.requires_grad = True
        
        # llama proj
        for name, param in self.model.llama_proj.named_parameters():
            param.requires_grad = True

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
            model=self.model, 
            model_parameters=self.model.parameters(),
            config_params=ds_params, 
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        # original
        loss, gen_loss, loss_pixel, mle_acc = self.ds_engine(batch)
        # modify
        # loss, gen_loss, loss_pixel, loss_coop, mle_acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        # original
        pbar.set_description(f'[!] Total loss: {round(loss.item(), 4)}; Gen loss: {round(gen_loss.item(), 4)}; Pixel loss: {round(loss_pixel.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        # modify
        # pbar.set_description(f'[!] Total loss: {round(loss.item(), 4)}; Gen loss: {round(gen_loss.item(), 4)}; Pixel loss: {round(loss_pixel.item(), 4)}; CoOP loss: {round(loss_coop.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            # original
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; Total loss: {round(loss.item(), 4)}; Gen loss: {round(gen_loss.item(), 4)}; Pixel loss: {round(loss_pixel.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
            # modify
            # logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; Total loss: {round(loss.item(), 4)}; Gen loss: {round(gen_loss.item(), 4)}; Pixel loss: {round(loss_pixel.item(), 4)}; CoOP loss: {round(loss_coop.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')

        mle_acc *= 100
        # original
        return loss, gen_loss, loss_pixel, mle_acc
        # modify
        # return loss, gen_loss, loss_pixel, loss_coop, mle_acc

    # evaluate model
    @torch.no_grad()
    def eval_model(self, batch):
        self.ds_engine.module.eval()

        # original
        loss, gen_loss, loss_pixel, mle_acc = self.ds_engine(batch)
        # modify
        # loss, gen_loss, loss_pixel, loss_coop, mle_acc = self.ds_engine(batch)

        mle_acc *= 100
        # original
        return loss, gen_loss, loss_pixel, mle_acc
        # modify
        # return loss, gen_loss, loss_pixel, loss_coop, mle_acc
    
    def save_model(self, path, current_step):
        # Save only trainable parameters to keep checkpoint small and picklable
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        trainable_ckpt = OrderedDict()

        # Gather each trainable param to rank0 and save as a plain CPU tensor
        for name, param in self.ds_engine.module.named_parameters():
            if not param.requires_grad:
                continue

            # Gather the full parameter on rank0
            with GatheredParameters([param], modifier_rank=0):
                if rank == 0:
                    trainable_ckpt[name] = param.detach().cpu().clone()

        if rank == 0:
            os.makedirs(path, exist_ok=True)
            torch.save(trainable_ckpt, f"{path}/trainable_params.pt")

            # Save tokenizer and config 
            self.model.llama_tokenizer.save_pretrained(path)
            self.model.llama_model.config.save_pretrained(path)

        print(f"[!] save trainable params into {path}/trainable_params.pt", flush=True)


    def load_stage_1_parameters(self, path):
        delta_ckpt = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)
