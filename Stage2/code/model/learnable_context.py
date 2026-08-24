import torch
import torch.nn as nn
import torch.nn.functional as F
    
class SoftPrompt_IB(nn.Module):
    def __init__(self, num_classes, embed_dim=1024, n_ctx=4):
        super().__init__()
        self.n_ctx = n_ctx
        self.num_classes = num_classes

        # shape = (C, n_ctx, D)
        self.ctx = nn.Parameter(torch.randn(num_classes, n_ctx, embed_dim) * 0.02)

    def get_prompt(self, class_idx, batch_size):
        """
        class_idx: int
        return: (B, n_ctx, D)
        """
        prompt = self.ctx[class_idx].unsqueeze(0).expand(batch_size, -1, -1)
        return prompt

