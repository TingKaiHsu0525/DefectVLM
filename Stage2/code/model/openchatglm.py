from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .modeling_chatglm import ChatGLMForConditionalGeneration
from .AnomalyGPT_models import LinearLayer, PromptLearner
from utils.loss import FocalLoss, BinaryDiceLoss

import torch

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer

from .learnable_context import SoftPrompt_IB
from .ImageBind.models import lora as IBLoRA
from .ImageBind.models.imagebind_model import load_module

# Defect type classes
NORMAL_TEXT     = ["normal wafer map"]
ABNORMAL_TEXT   = ["center defect type wafer map", "donut defect type wafer map","loc defect type wafer map"
                    , "edge loc defect type wafer map", "edge ring defect type wafer map", "scratch defect type wafer map"]


class OpenChatGLMPEFTModel(nn.Module):
    '''
    ChatGLM-6B counterpart of OpenLLAMAPEFTModel (model/openllama.py). Same ImageBind vision and text encoder, soft-prompt, and pixel-level Focal+Dice anomaly
    loss, only change the LLM backbone.
    '''

    # 1 image token (from ImageBind global feature) + 18 PromptLearner tokens
    # (9 learnable base prompts + 9 image-conditioned prompts from the anomaly map).
    N_VISUAL_TOKENS = 19

    def __init__(self, **args):
        super(OpenChatGLMPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        chatglm_ckpt_path = args['vicuna_ckpt_path']
        max_tgt_len = args['max_tgt_len']

        self.device = torch.cuda.current_device()

        print(f'Initializing visual encoder from {imagebind_ckpt_path} ...')

        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)

        # soft prompt (CoOP-style learnable per-class text context)
        self.soft_prompt = SoftPrompt_IB(num_classes=7, embed_dim=1024, n_ctx=8).to(self.device)

        imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)

        # load stage1 fine-tuned imagebind model
        lora_ckpt_dir = args["imagebind_lora_ckpt_dir"]
        lora_postfix  = args.get("imagebind_lora_postfix", "_mixed_last")
        lora_rank     = args.get("imagebind_lora_rank", 4)
        print(f"Loading ImageBind-LoRA adapter from {lora_ckpt_dir} ...")
        print(f"LoRA rank: {lora_rank}, postfix: {lora_postfix}")

        self.visual_encoder.modality_trunks.update(
            IBLoRA.apply_lora_modality_trunks(
                self.visual_encoder.modality_trunks,
                rank=lora_rank,
                modality_names=[ModalityType.VISION, ModalityType.TEXT],
            )
        )

        IBLoRA.load_lora_modality_trunks(
            self.visual_encoder.modality_trunks,
            checkpoint_dir=lora_ckpt_dir,
            postfix=lora_postfix,
        )

        load_module(
            self.visual_encoder.modality_postprocessors,
            module_name="postprocessors",
            checkpoint_dir=lora_ckpt_dir,
            postfix=lora_postfix,
        )
        load_module(
            self.visual_encoder.modality_heads,
            module_name="heads",
            checkpoint_dir=lora_ckpt_dir,
            postfix=lora_postfix,
        )
        print("Loaded ImageBind-LoRA fine-tuned encoder.")

        self.image_decoder = LinearLayer(1280, 1024, 3)
        self.prompt_learner = PromptLearner(2, 4096)

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        # freeze vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Visual encoder initialized.')

        print(f'Initializing language decoder from {chatglm_ckpt_path} ...')

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=['query_key_value'],
        )

        # Uses the local, patched ChatGLMForConditionalGeneration (model/modeling_chatglm.py)
        self.llama_model = ChatGLMForConditionalGeneration.from_pretrained(chatglm_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = AutoTokenizer.from_pretrained(chatglm_ckpt_path, trust_remote_code=True)

        print('Language decoder initialized.')

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )

        self.max_tgt_len = max_tgt_len

    # vision encoding (unchanged from OpenLLAMAPEFTModel)

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0]     # bsz x 1024
            patch_features = embeddings['vision'][1]   # bsz x h*w x 1280
            del patch_features[1]
        patch_tokens = self.image_decoder(patch_features)   # bsz x h*w x 1024
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1)  # bsz x 1 x hidden
        return inputs_llama, patch_tokens

    def encode_image_from_tensor(self, image_tensors):
        if isinstance(image_tensors, list):
            image_tensors = torch.stack(image_tensors, dim=0)
        inputs = {ModalityType.VISION: image_tensors.to(self.device)}
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0]
            patch_features = embeddings['vision'][1]
            del patch_features[1]
        patch_tokens = self.image_decoder(patch_features)
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1)  # bsz x 1 x hidden
        return inputs_llama, patch_tokens

    # learnable context prompt for each class (identical to OpenLLAMAPEFTModel.encode_text_soft)
    def encode_text_soft(self, normal_texts, abnormal_texts):
        """
        Return:
            text_features: (1, 7, 1024)
            index 0: normal
            index 1~6: [center, donut, loc, edge_loc, edge_ring, scratch]
        """

        def encode_one(text_list, class_idx):
            token_ids = data.load_and_transform_text(text_list, self.device)

            prep = self.visual_encoder.modality_preprocessors[ModalityType.TEXT]
            prep_out = prep(text=token_ids)

            tokens = prep_out["trunk"]["tokens"]
            attn_mask = prep_out["trunk"].get("attn_mask", None)
            head_inputs = prep_out["head"]

            if tokens.ndim == 4:
                tokens = tokens.mean(dim=1)

            B = tokens.shape[0]

            prompt = self.soft_prompt.get_prompt(class_idx, B)
            tokens = torch.cat([prompt, tokens], dim=1)

            if attn_mask is not None:
                old_L = attn_mask.shape[0]
                n_ctx = self.soft_prompt.n_ctx
                pad_top  = torch.zeros(n_ctx, old_L, device=self.device, dtype=attn_mask.dtype)
                pad_left = torch.zeros(old_L + n_ctx, n_ctx, device=self.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([pad_top, attn_mask], dim=0)
                attn_mask = torch.cat([pad_left, attn_mask], dim=1)

            trunk = self.visual_encoder.modality_trunks[ModalityType.TEXT]
            hidden, _ = trunk(tokens, attn_mask=attn_mask, out_layers=self.visual_encoder.out_layers)

            head_inputs["seq_len"] = torch.full(
                (B,), tokens.shape[1] - 1, dtype=torch.long, device=self.device
            )

            head = self.visual_encoder.modality_heads[ModalityType.TEXT]
            feat = head(hidden, **head_inputs)

            post = self.visual_encoder.modality_postprocessors[ModalityType.TEXT]
            feat = post(feat)

            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            return feat

        normal_all = encode_one(normal_texts, class_idx=0)
        normal_feat = normal_all.mean(dim=0, keepdim=True)
        normal_feat = normal_feat / (normal_feat.norm(dim=-1, keepdim=True) + 1e-6)

        ab_feats = []
        for i, t in enumerate(abnormal_texts):
            ab_i = encode_one([t], class_idx=i + 1)
            ab_feats.append(ab_i)
        ab_feats = torch.cat(ab_feats, dim=0).unsqueeze(0)

        text_features = torch.cat([normal_feat.unsqueeze(1), ab_feats], dim=1)
        return text_features

    def _patch_tokens_to_anomaly_maps(self, patch_tokens, feats_text_tensor):
        """patch_tokens (per ImageBind layer) x text prototypes -> per-layer (B,2,224,224) binary prob maps."""
        anomaly_maps_bin = []
        for layer in range(len(patch_tokens)):
            pt = patch_tokens[layer]
            pt = pt / (pt.norm(dim=-1, keepdim=True) + 1e-6)

            logits = 100.0 * (pt @ feats_text_tensor.transpose(-2, -1))  # (B,L,C)
            B, L, C = logits.shape
            H = int(np.sqrt(L))

            logits = logits.permute(0, 2, 1).contiguous().view(B, C, H, H)
            logits = F.interpolate(logits, size=224, mode="bilinear", align_corners=True)

            normal_score = logits[:, 0, :, :]
            abnormal_score = torch.logsumexp(logits[:, 1:, :, :], dim=1)

            logits_bin = torch.stack([normal_score, abnormal_score], dim=1)
            prob_bin = torch.softmax(logits_bin, dim=1)
            anomaly_maps_bin.append(prob_bin)
        return anomaly_maps_bin

    # ChatGLM token/placeholder construction 

    def _build_context_ids(self, prompt_text):
        """
        Tokenizes "[Round 0]\\n问：<img>{N_VISUAL_TOKENS placeholders}</img>{prompt}\\n答："
        Returns (context_ids: List[int], visual_position: int) -- visual_position is the
        offset of the placeholder span within context_ids (and thus within the final
        input_ids, since context_ids is always its prefix).
        """
        tok = self.llama_tokenizer
        prefix_ids = tok("[Round 0]\n问：<img>", add_special_tokens=False).input_ids
        suffix_ids = tok(f"</img>{prompt_text}\n答：", add_special_tokens=False).input_ids
        visual_position = len(prefix_ids)
        context_ids = prefix_ids + [tok.pad_token_id] * self.N_VISUAL_TOKENS + suffix_ids
        return context_ids, visual_position

    def _build_training_sample(self, prompt_text, answer_text):
        """Returns (input_ids: List[int], labels: List[int], visual_position: int)."""
        tok = self.llama_tokenizer
        context_ids, visual_position = self._build_context_ids(prompt_text)
        answer_ids = tok(answer_text, add_special_tokens=False).input_ids

        # ChatGLM appends [gMASK, bos] itself for generation-style inputs
        head_ids = context_ids + [tok.gmask_token_id, tok.bos_token_id]
        input_ids = head_ids + answer_ids + [tok.eos_token_id]
        labels = [-100] * len(head_ids) + answer_ids + [tok.eos_token_id]
        return input_ids, labels, visual_position

    def forward(self, inputs):
        if 'mask' not in inputs:
            return

        image_paths = inputs['images']
        img_embeds, patch_tokens = self.encode_image_from_tensor(image_paths)  # img_embeds: (B,1,hidden)
        B = img_embeds.shape[0]

        loss_pixel = torch.zeros((), device=self.device)
        feats_text_tensor = self.encode_text_soft(NORMAL_TEXT, ABNORMAL_TEXT)

        gt = inputs['mask'].to(self.device)
        gt[gt > 0], gt[gt <= 0] = 1, 0

        bin_sum = None
        num_layers = len(patch_tokens)

        for layer in range(num_layers):
            patch = patch_tokens[layer]
            patch = patch / (patch.norm(dim=-1, keepdim=True) + 1e-6)

            logits = 100.0 * (patch @ feats_text_tensor.transpose(-2, -1))
            Bc, L, C = logits.shape
            H = int(np.sqrt(L))

            logits = logits.permute(0, 2, 1).contiguous().view(Bc, C, H, H)
            logits = F.interpolate(logits, size=224, mode="bilinear", align_corners=True)

            normal_score = logits[:, 0, :, :]
            abnormal_score = torch.logsumexp(logits[:, 1:, :, :], dim=1)

            logits_bin = torch.stack([normal_score, abnormal_score], dim=1)
            prob_bin = torch.softmax(logits_bin, dim=1)

            f_loss = self.loss_focal(prob_bin, gt)
            d_loss = self.loss_dice(prob_bin[:, 1, :, :], gt)
            loss_pixel = loss_pixel + f_loss + d_loss

            bin_sum = prob_bin if bin_sum is None else (bin_sum + prob_bin)

        anomaly_map_bin = bin_sum / float(num_layers)
        anomaly_map_prompts = self.prompt_learner(anomaly_map_bin)  # (B, 18, hidden)

        visual_embeds_full = torch.cat([img_embeds, anomaly_map_prompts], dim=1)  # (B, 19, hidden)
        assert visual_embeds_full.shape[1] == self.N_VISUAL_TOKENS

        output_texts = inputs['texts']  # list[B] of [{"from": "human"/"gpt", "value": str}, ...]

        gen_loss_sum = torch.zeros((), device=self.device)
        acc_num, acc_den = 0, 0
        for b in range(B):
            conv = output_texts[b]
            human_turn = next(t['value'] for t in conv if t['from'] == 'human')
            gpt_turn = next(t['value'] for t in conv if t['from'] == 'gpt')

            input_ids, labels, visual_pos = self._build_training_sample(human_turn, gpt_turn)
            input_ids = input_ids[:self.max_tgt_len]
            labels = labels[:self.max_tgt_len]

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            labels_t = torch.tensor(labels, dtype=torch.long, device=self.device).unsqueeze(0)

            out = self.llama_model(
                input_ids=input_ids_t,
                visual_embeds=visual_embeds_full[b:b + 1],
                visual_position=visual_pos,
                labels=labels_t,
            )
            gen_loss_sum = gen_loss_sum + out.loss

            chosen_tokens = torch.argmax(out.logits, dim=-1)[:, :-1]
            target = labels_t[:, 1:]
            valid_mask = (target != -100)
            acc_num += int(((chosen_tokens == target) & valid_mask).sum().item())
            acc_den += int(valid_mask.sum().item())

        gen_loss = gen_loss_sum / B
        gen_acc = acc_num / max(acc_den, 1)

        total_loss = gen_loss + loss_pixel / float(num_layers)
        return total_loss, gen_loss, loss_pixel, gen_acc

    # inference 

    def extract_multimodal_feature(self, inputs, web_demo=False):
        image_embeds, patch_tokens = self.encode_image(inputs['image_paths'])
        feats_text_tensor = self.encode_text_soft(NORMAL_TEXT, ABNORMAL_TEXT)

        anomaly_maps_bin = self._patch_tokens_to_anomaly_maps(patch_tokens, feats_text_tensor)
        anomaly_map_ret = torch.mean(torch.stack(anomaly_maps_bin, dim=0), dim=0)  # (B,2,224,224)

        return image_embeds, anomaly_map_ret

    def generate(self, inputs, web_demo=False):
        '''
            inputs = {
                'image_paths': required,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature,
            }
        '''
        image_embeds, anomaly_map = self.extract_multimodal_feature(inputs, web_demo)
        anomaly_map_prompts = self.prompt_learner(anomaly_map)  # (1, 18, hidden)
        visual_embeds = torch.cat([image_embeds, anomaly_map_prompts], dim=1)  # (1, 19, hidden)
        assert visual_embeds.shape[1] == self.N_VISUAL_TOKENS

        tok = self.llama_tokenizer
        context_ids, visual_position = self._build_context_ids(inputs['prompt'])
        input_ids = context_ids + [tok.gmask_token_id, tok.bos_token_id]
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        outputs = self.llama_model.generate(
            input_ids=input_ids_t,
            visual_embeds=visual_embeds,
            visual_position=visual_position,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        output_text = tok.decode(outputs[0][input_ids_t.shape[1]:], skip_special_tokens=True)
        return output_text, anomaly_map
