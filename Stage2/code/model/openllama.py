from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .modeling_llama import LlamaForCausalLM
from .AnomalyGPT_models import LinearLayer, PromptLearner
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.loss import FocalLoss, BinaryDiceLoss
import kornia as K

import torch
from torch.nn.utils import rnn

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import LlamaTokenizer

import torchvision.utils as vutils
import matplotlib.pyplot as plt
from types import SimpleNamespace

from .learnable_context import SoftPrompt_IB
from .ImageBind.models.multimodal_preprocessors import SimpleTokenizer
from .ImageBind.models import lora as IBLoRA
from .ImageBind.models.imagebind_model import load_module
from pathlib import Path

# Defect type classes
NORMAL_TEXT     = ["normal wafer map"]
ABNORMAL_TEXT   = ["center defect type wafer map", "donut defect type wafer map","loc defect type wafer map"
                    , "edge loc defect type wafer map", "edge ring defect type wafer map", "scratch defect type wafer map"]

prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
prompt_abnormal = ['center defect', 'donut defect', 'Loc type defect', 'Edge Loc type defect' ,'Edge ring type defect' , 'scratch type defect']


prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = ['a photo of a {}.', 'a photo of the {}.']

objs = ['wafer map']

prompt_sentences = {}

for obj in objs:
    prompt_sentence_obj = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = data.load_and_transform_text(prompted_sentence, torch.cuda.current_device())
        prompt_sentence_obj.append(prompted_sentence)
    prompt_sentences[obj] = prompt_sentence_obj

# fixed text embeddings for each class
def encode_text_with_prompt_ensemble(model, obj, device):

    global prompt_sentences
    normal_sentences = []
    abnormal_sentences = []
    for idx in range(len(obj)):
        sentence = prompt_sentences[obj[idx].replace('_', ' ')]
        normal_sentences.append(sentence[0])
        abnormal_sentences.append(sentence[1])
        # print(f"normal sentences: {normal_sentences}")
        # print(f"abnormal sentences: {abnormal_sentences}")

    normal_sentences = torch.cat(normal_sentences).to(device)
    abnormal_sentences = torch.cat(abnormal_sentences).to(device)

    class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT][0]
    class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT][0]
    # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    class_embeddings_normal = class_embeddings_normal.reshape((len(obj), len(prompt_templates) * len(prompt_normal), 1024))
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.reshape((len(obj), len(prompt_templates) * len(prompt_abnormal), 1024))
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)

    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)

    # print(f"text_features shape: {text_features.shape}")

    return text_features

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()



PROMPT_START = '### Human: <Img>'
class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        max_tgt_len = args['max_tgt_len']
        stage = args['stage']

        self.device = torch.cuda.current_device()

        print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')

        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)
        
        # soft prompt
        self.soft_prompt = SoftPrompt_IB(num_classes=7, embed_dim=1024, n_ctx=8).to(self.device)

        imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)

        # print(f"size: {self.visual_hidden_size}")

        # load stage1 fine-tuned imagebind model
        lora_ckpt_dir = args["imagebind_lora_ckpt_dir"]       
        lora_postfix  = args.get("imagebind_lora_postfix", "_mixed_last") 
        lora_rank     = args.get("imagebind_lora_rank", 4)
        print(f"Loading ImageBind-LoRA adapter from {lora_ckpt_dir} ...")
        print(f"LoRA rank: {lora_rank}, postfix: {lora_postfix}")

        # apply LoRA wrappers on modality trunks
        self.visual_encoder.modality_trunks.update(
            IBLoRA.apply_lora_modality_trunks(
                self.visual_encoder.modality_trunks,
                rank=lora_rank,
                modality_names=[ModalityType.VISION, ModalityType.TEXT],
            )
        )

        # print("[Debug] visual_encoder.out_layers =", self.visual_encoder.out_layers)
        # print("[Debug] vision trunk =", type(self.visual_encoder.modality_trunks[ModalityType.VISION]))
        # breakpoint()

        # load lora weights
        IBLoRA.load_lora_modality_trunks(
            self.visual_encoder.modality_trunks,
            checkpoint_dir=lora_ckpt_dir,
            postfix=lora_postfix,
        )

        # 3) load postprocessors & heads
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

        self.iter = 0

        self.image_decoder = LinearLayer(1280, 1024, 3)


        self.prompt_learner = PromptLearner(2, 4096)


        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print ('Visual encoder initialized.')

        print (f'Initializing language decoder from {vicuna_ckpt_path} ...')
        
        # load the cofiguration of peft
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )

        self.max_tgt_len = max_tgt_len


    def rot90_img(self,x,k):
        # k is 0,1,2,3
        degreesarr = [0., 90., 180., 270., 360]
        degrees = torch.tensor(degreesarr[k]).to(self.llama_model.dtype).to(self.device)
        x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
        return x

    def encode_video(self, video_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION][0] # bsz x 1024
        inputs_llama = self.llama_proj(video_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO][0] # bsz x 1024
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_thermal(self, thermal_paths):
        inputs = {ModalityType.THERMAL: data.load_and_transform_thermal_data(thermal_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['thermal'][0] # bsz x 1024
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            # original
            del patch_features[1]
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024
        # patch_tokens = None

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens
    
    def encode_image_for_web_demo(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data_for_web_demo(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens
    
    def encode_image_for_one_shot(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    def encode_image_for_one_shot_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    def encode_image_for_one_shot_with_aug(self, image_paths):
        image_tensors = data.load_and_transform_vision_data(image_paths, self.device).to(self.llama_model.dtype)
        B,C,H,W = image_tensors.shape
        # print(B,C,H,W)

        rotated_images = torch.zeros((4, B, C, H, W)).to(self.llama_model.dtype).to(self.device)


        for j, degree in enumerate([0, 1, 2, 3]):
            rotated_img = self.rot90_img(image_tensors, degree)
            # 存储旋转后的图像
            rotated_images[j] = rotated_img

        # test
        # vutils.save_image(rotated_img, 'normal_aug.png', normalize=True)
        
        image_tensors = rotated_images.transpose(0,1).reshape(B * 4, C, H, W)

        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :].reshape(B,4,256,1280).reshape(B, 4 * 256, 1280)

        return patch_features
    
    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1024
            del patch_features[1]

        patch_tokens = self.image_decoder(patch_features)
        # patch_tokens = None

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens, image_embeds
    
    def encode_image_from_tensor_no_patch(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask, anomaly_embedding = None):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        p_middle = '</Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim


        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim

        

        if anomaly_embedding != None:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 
        else:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 



    # learnable context prompt for each class
    def encode_text_soft(self, normal_texts, abnormal_texts):
        """
        Return:
            text_features: (1, 7, 1024)
            index 0: normal
            index 1~6: [center, donut, loc, edge_loc, edge_ring, scratch]
        """

        def encode_one(text_list, class_idx):
            # text_list: List[str]
            token_ids = data.load_and_transform_text(text_list, self.device)

            prep = self.visual_encoder.modality_preprocessors[ModalityType.TEXT]
            prep_out = prep(text=token_ids)

            tokens = prep_out["trunk"]["tokens"]              # token embeddings
            attn_mask = prep_out["trunk"].get("attn_mask", None)
            head_inputs = prep_out["head"]

            if tokens.ndim == 4:
                tokens = tokens.mean(dim=1)

            B = tokens.shape[0]

            # Get class-specific soft prompt
            prompt = self.soft_prompt.get_prompt(class_idx, B)  # (B, n_ctx, 1024)
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
            feat = post(feat)  # (B, 1024)

            # L2 normalize
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            return feat

        # normal context
        normal_all = encode_one(normal_texts, class_idx=0)                # (Nn,1024)
        normal_feat = normal_all.mean(dim=0, keepdim=True)                # (1,1024)
        normal_feat = normal_feat / (normal_feat.norm(dim=-1, keepdim=True) + 1e-6)

        # abnormal context
        ab_feats = []
        for i, t in enumerate(abnormal_texts):
            ab_i = encode_one([t], class_idx=i + 1)                       # (1,1024)
            ab_feats.append(ab_i)

        ab_feats = torch.cat(ab_feats, dim=0).unsqueeze(0)                # (1,6,1024)

        # stack: (1,7,1024)
        text_features = torch.cat([normal_feat.unsqueeze(1), ab_feats], dim=1)
        return text_features


    def forward(self, inputs):

        if 'mask' in inputs:

            image_paths = inputs['images']
            img_embeds, _, patch_tokens, img_embeds_raw = self.encode_image_from_tensor(image_paths)
   
            class_name = ['wafer map']

            loss_pixel = torch.zeros((), device=self.device)

            # fixed text features with prompt ensemble
            # feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, class_name, self.device)
  
            # text features with learnable soft prompt
            feats_text_tensor = self.encode_text_soft(NORMAL_TEXT, ABNORMAL_TEXT)
            
            # ground truth mask
            gt = inputs['mask'].to(self.device)
            gt[gt > 0], gt[gt <= 0] = 1, 0

            bin_sum = None
            num_layers = len(patch_tokens)

            # generate anomaly map and calculate pixel-wise loss
            for layer in range(num_layers):
                patch = patch_tokens[layer]
                patch = patch / (patch.norm(dim=-1, keepdim=True) + 1e-6)

                # (B, L, 1024) @ (1,1024,7) -> (B, L, 7)
                logits = 100.0 * (patch @ feats_text_tensor.transpose(-2, -1))
                B, L, C = logits.shape  
                H = int(np.sqrt(L))

                # (B, L, 7) -> (B, 7, H, H) -> (B, 7, 224, 224)
                logits = logits.permute(0, 2, 1).contiguous().view(B, C, H, H)
                logits = F.interpolate(logits, size=224, mode="bilinear", align_corners=True)

                # binary logits
                normal_score = logits[:, 0, :, :]  # (B,224,224)
                abnormal_score = torch.logsumexp(logits[:, 1:, :, :], dim=1)  # (B,224,224)

                logits_bin = torch.stack([normal_score, abnormal_score], dim=1)  # (B,2,224,224)
                prob_bin = torch.softmax(logits_bin, dim=1)                      # (B,2,224,224)

                # pixel-wise loss 
                f_loss = self.loss_focal(prob_bin, gt)
                d_loss = self.loss_dice(prob_bin[:, 1, :, :], gt)
                loss_pixel = loss_pixel + f_loss + d_loss

                bin_sum = prob_bin if bin_sum is None else (bin_sum + prob_bin)

            # Average binary maps over layers
            anomaly_map_bin = bin_sum / float(num_layers)  # (B,2,224,224)

            # Feed into prompt learner
            anomaly_map_prompts = self.prompt_learner(anomaly_map_bin)

            output_texts = inputs['texts']
            input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts, self.max_tgt_len)
            inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask, anomaly_map_prompts)

            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            gen_loss = outputs.loss

            # loss_l2 = torch.norm(anomaly_map_prompts / 2 , p=2)
            # loss_l2 = nn.MSELoss()(img_embeds_origin, img_embeds)
            # calculate the token accuarcy
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
            labels = targets[:, 2:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
            valid_mask = (labels != -100).reshape(-1)
            valid_tokens = gen_acc & valid_mask    # [B*S]
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()


            # total_loss = gen_loss + loss_pixel
            total_loss = gen_loss + loss_pixel / float(num_layers) 
            
            return total_loss, gen_loss, loss_pixel, gen_acc
        
    # inference
    def extract_multimodal_feature(self, inputs, web_demo):
        features = []
        if inputs['image_paths']:
            c_name = 'wafer map'
            if not web_demo:
                image_embeds, _, patch_tokens = self.encode_image(inputs['image_paths'])


                class_name = ['wafer map']
                # fixed text features with prompt ensemble
                # feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, class_name, self.device)

                # text features with learnable soft prompt
                feats_text_tensor = self.encode_text_soft(NORMAL_TEXT, ABNORMAL_TEXT)
            else:
                image_embeds, _, patch_tokens = self.encode_image_for_web_demo(inputs['image_paths'])
               
                # fixed text features with prompt ensemble
                # feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, class_name, self.device)

                # text features with learnable soft prompt
                feats_text_tensor = self.encode_text_soft(NORMAL_TEXT, ABNORMAL_TEXT)

            anomaly_maps_bin = []
            for layer in range(len(patch_tokens)):
                pt = patch_tokens[layer]
                pt = pt / (pt.norm(dim=-1, keepdim=True) + 1e-6)

                logits = 100.0 * (pt @ feats_text_tensor.transpose(-2, -1))  # (B,L,C)
                B, L, C = logits.shape
                H = int(np.sqrt(L))

                logits = logits.permute(0, 2, 1).contiguous().view(B, C, H, H)
                logits = F.interpolate(logits, size=224, mode="bilinear", align_corners=True)

                # normal and abnormal logits
                normal_score = logits[:, 0, :, :]                             # (B,224,224)
                abnormal_score = torch.logsumexp(logits[:, 1:, :, :], dim=1)  # (B,224,224)

                logits_bin = torch.stack([normal_score, abnormal_score], dim=1)  # (B,2,224,224)
                prob_bin = torch.softmax(logits_bin, dim=1)                      # (B,2,224,224)

                anomaly_maps_bin.append(prob_bin)

            # average over layers -> (B,2,224,224)
            anomaly_map_ret = torch.mean(torch.stack(anomaly_maps_bin, dim=0), dim=0)
                
            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)
        if inputs['thermal_paths']:
            thermal_embeds, _ = self.encode_thermal(inputs['thermal_paths'])
            features.append(thermal_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds, anomaly_map_ret

    def prepare_generation_embedding(self, inputs, web_demo):
        prompt = inputs['prompt']
        feature_embeds, anomaly_map = self.extract_multimodal_feature(inputs, web_demo)
        inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        
        p_middle = '</Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        # self.prompt_learner.eval()
        anomaly_map_prompts = self.prompt_learner(anomaly_map)

        text = prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, anomaly_map_prompts, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim

        # inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, p_after_embeds], dim=1)

        return inputs_embeds, anomaly_map

    def generate(self, inputs, web_demo=False):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''
        # self.prompt_learner.eval()
        # self.llama_model.eval()
        # self.llama_proj.eval()
        # self.image_decoder.eval()
        # self.llama_tokenizer.eval()
        input_embeds, pixel_output = self.prepare_generation_embedding(inputs, web_demo)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True, 
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text, pixel_output


