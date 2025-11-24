import torch
import torch.nn.functional as F
from torch import nn

from .mae_vit import mae_vit_base_patch16, mae_vit_large_patch16

from .bert_backbone import BertModel
from .utils import Mlp, DropPath, LayerScale
import numpy as np
import os


class CrossAttention(nn.Module):
    """Simple cross-attention block that projects q/k/v separately for each modality."""
    def __init__(self, q_dim, kv_dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert q_dim % num_heads == 0, 'q_dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.scale = (q_dim // num_heads) ** -0.5

        self.q = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, q_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(q_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_input, kv_input, kv_mask=None):
        B, Nq, Cq = q_input.shape
        Nk = kv_input.shape[1]

        q = self.q(q_input).reshape(B, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv_input).reshape(B, Nk, 2, self.num_heads, Cq // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if kv_mask is not None:
            attn = attn.masked_fill(kv_mask[:, None, None, :], float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, Cq)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DeltaBlock(nn.Module):
    """Bidirectional modality exchange: vision<->language with cross-attn + FFN for each."""
    def __init__(self, vis_dim, txt_dim, num_heads, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, mlp_ratio=4.0):
        super().__init__()
        self.vis_norm_q = nn.LayerNorm(vis_dim)
        self.vis_norm_kv = nn.LayerNorm(txt_dim)
        self.txt_norm_q = nn.LayerNorm(txt_dim)
        self.txt_norm_kv = nn.LayerNorm(vis_dim)

        self.vis_cross = CrossAttention(vis_dim, txt_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.txt_cross = CrossAttention(txt_dim, vis_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        self.vis_mlp_norm = nn.LayerNorm(vis_dim)
        self.txt_mlp_norm = nn.LayerNorm(txt_dim)
        self.vis_mlp = Mlp(in_features=vis_dim, hidden_features=int(vis_dim * mlp_ratio))
        self.txt_mlp = Mlp(in_features=txt_dim, hidden_features=int(txt_dim * mlp_ratio))

        self.vis_ls1 = LayerScale(vis_dim)
        self.txt_ls1 = LayerScale(txt_dim)
        self.vis_ls2 = LayerScale(vis_dim)
        self.txt_ls2 = LayerScale(txt_dim)

        # Gating parameters for residual modulation
        self.vis_attn_gate = nn.Parameter(torch.ones(1, 1, vis_dim))
        self.txt_attn_gate = nn.Parameter(torch.ones(1, 1, txt_dim))
        self.vis_ffn_gate = nn.Parameter(torch.ones(1, 1, vis_dim))
        self.txt_ffn_gate = nn.Parameter(torch.ones(1, 1, txt_dim))

        self.vis_drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.txt_drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.vis_drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.txt_drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, vis_feat, txt_feat, vis_mask=None, txt_mask=None):
        vis_attn_out = self.vis_drop_path1(self.vis_ls1(self.vis_cross(self.vis_norm_q(vis_feat),
                                                                       self.vis_norm_kv(txt_feat),
                                                                       txt_mask)))
        txt_attn_out = self.txt_drop_path1(self.txt_ls1(self.txt_cross(self.txt_norm_q(txt_feat),
                                                                       self.txt_norm_kv(vis_feat),
                                                                       vis_mask)))
        vis_feat = vis_feat + torch.tanh(self.vis_attn_gate) * vis_attn_out
        txt_feat = txt_feat + torch.tanh(self.txt_attn_gate) * txt_attn_out

        vis_ffn_out = self.vis_drop_path2(self.vis_ls2(self.vis_mlp(self.vis_mlp_norm(vis_feat))))
        txt_ffn_out = self.txt_drop_path2(self.txt_ls2(self.txt_mlp(self.txt_mlp_norm(txt_feat))))
        vis_feat = vis_feat + torch.tanh(self.vis_ffn_gate) * vis_ffn_out
        txt_feat = txt_feat + torch.tanh(self.txt_ffn_gate) * txt_ffn_out
        return vis_feat, txt_feat


class ModalityUnifiedFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        """ Initializes the model."""
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cont_loss_layer = cfg.MODEL.BACKBONE.CONT_LOSS_LAYER
        self.txt_token_mode = cfg.MODEL.BACKBONE.TXT_TOKEN_MODE
        
        if 'base' in cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            self.vit = mae_vit_base_patch16(img_size=(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE), 
                                            learnable_pos=cfg.MODEL.LEARNABLE_POSITION,
                                            drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
            self.vit.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_PATH, map_location='cpu')['model'], strict=False)

            # Text feature encoder (BERT)
            self.bert = BertModel.from_pretrained(cfg.MODEL.BACKBONE.LANGUAGE.TYPE)
            
        elif 'large' in cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            self.vit = mae_vit_large_patch16(img_size=(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE), 
                                            learnable_pos=cfg.MODEL.LEARNABLE_POSITION,
                                            drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
            self.vit.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_PATH, map_location='cpu')['model'], strict=False)

            # Text feature encoder (BERT)
            self.bert = BertModel.from_pretrained(cfg.MODEL.BACKBONE.LANGUAGE.TYPE)
            
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

        # Load ViT weights from OSTrack and freeze ViT/BERT transformer blocks
        self._load_vit_pretrain("pretrain/OSTrack_ep0300.pth.tar")
        for p in self.vit.blocks.parameters():
            p.requires_grad_(False)
        # Freeze ViT embedding & positional params
        for p in self.vit.patch_embed.parameters():
            p.requires_grad_(False)
        for p in [self.vit.pos_embed_z, self.vit.pos_embed_x]:
            p.requires_grad_(False)
        for p in self.bert.encoder.layer.parameters():
            p.requires_grad_(False)
        # Freeze BERT embeddings
        for p in self.bert.embeddings.parameters():
            p.requires_grad_(False)

        # Build delta blocks for every layer (vision + text).
        if len(self.vit.blocks) != len(self.bert.encoder.layer):
            raise ValueError(f"ViT layers ({len(self.vit.blocks)}) and BERT layers ({len(self.bert.encoder.layer)}) must match.")
        num_layers = len(self.vit.blocks)
        attn_drop = getattr(self.vit.blocks[0].attn.attn_drop, 'p', 0.0) if len(self.vit.blocks) > 0 else 0.0
        proj_drop = getattr(self.vit.blocks[0].attn.proj_drop, 'p', 0.0) if len(self.vit.blocks) > 0 else 0.0
        self.delta_blocks = nn.ModuleList([
            DeltaBlock(
                vis_dim=self.vit.blocks[0].attn.qkv.in_features,
                txt_dim=self.bert.encoder.layer[0].attention.self.query.in_features,
                num_heads=self.vit.blocks[0].attn.num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=getattr(self.vit.blocks[i].drop_path1, 'drop_prob', 0.0)
            ) for i in range(num_layers)
        ])

    def cat_mask(self, text, flag):
        x_mask = torch.ones([flag.shape[0], self.vit.num_patches_x]).to(flag.device)
        z_mask = torch.ones([flag.shape[0], self.vit.num_patches_z]).to(flag.device)*(flag!=1) # =1 mask
        c_mask = torch.ones([flag.shape[0], 1]).to(flag.device)*(flag!=1) # =1 mask
        t_mask = text.mask*(flag!=0) # =0 mask
        mask = ~torch.cat([c_mask, z_mask, x_mask, t_mask], dim=1).bool()
        visual_mask = ~torch.cat([c_mask, z_mask, x_mask], dim=1).bool()
        return mask, visual_mask

    def build_text_mask(self, bert_mask):
        # bert_mask: (B, 1, 1, L) with negative values for padded tokens
        if bert_mask is None:
            return None
        return bert_mask.squeeze(1).squeeze(1) < 0

    def forward(self, template, search, text, flag): # one more token
        img_feat = self.vit.patchify(template, search)
        txt_feat, bert_mask = self.bert.embedding(text.tensors, token_type_ids=None, attention_mask=text.mask)
        _, visual_mask = self.cat_mask(text, flag)
        text_mask = self.build_text_mask(bert_mask)
        logits_list = []
        for i in range(len(self.vit.blocks)):
            img_feat = self.vit.blocks[i](img_feat, visual_mask, flag=flag)
            txt_feat = self.bert.encoder.layer[i](txt_feat, bert_mask)
            img_feat, txt_feat = self.delta_blocks[i](img_feat, txt_feat, visual_mask, text_mask)
            if i in self.cont_loss_layer:
                logits = self.contractive_learning(img_feat, txt_feat, text, flag)
                logits_list.append(logits)
        vis_token, z, x = img_feat.split([1, self.vit.num_patches_z, self.vit.num_patches_x], dim=1)
        b, s, c = x.shape
        out_dict = {
            "search": x,
            "template": z,
            "text": txt_feat,
            "vis_token": vis_token,
            "txt_token": self.generate_txt_token(txt_feat, text),
            "flag": flag.reshape(-1),
            "logits": torch.stack(logits_list, dim=1).reshape(b, -1, int(s**0.5), int(s**0.5))
        }
        return out_dict
    
    def generate_txt_token(self, txt_feat, text):
        if self.txt_token_mode == 'mean':
            return (txt_feat*text.mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / text.mask.unsqueeze(-1).sum(dim=1, keepdim=True)
        elif self.txt_token_mode == 'cls':
            return txt_feat[:, :1]
    
    def contractive_learning(self, img_feat, txt_feat, text, flag):
        vis_token, z, x = img_feat.split([1, self.vit.num_patches_z, self.vit.num_patches_x], dim=1)
        txt_token = self.generate_txt_token(txt_feat, text)
        vis_logits = self.logit_scale.exp()*(F.normalize(x, dim=-1) @ F.normalize(vis_token, dim=-1).transpose(-2,-1))
        txt_logits = self.logit_scale.exp()*(F.normalize(x, dim=-1) @ F.normalize(txt_token, dim=-1).transpose(-2,-1))
        logits_group = torch.stack([vis_logits, txt_logits, (vis_logits+txt_logits)/2], dim=1)
        bid = torch.arange(flag.shape[0])
        logits = logits_group[bid, flag.reshape(-1)]
        return logits

    def _load_vit_pretrain(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('net', ckpt.get('model', ckpt.get('state_dict', ckpt)))
        if any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        # Strip backbone prefix if present and keep ViT-relevant keys
        vit_state = {}
        for k, v in state.items():
            key = k
            for prefix in ['backbone.', 'module.backbone.']:
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if key.startswith('encoder.'):
                key = key[len('encoder.'):]
            # only keep vit params
            if any(key.startswith(t) for t in ['patch_embed', 'pos_embed', 'cls_token', 'modal_embed', 'blocks', 'norm']):
                vit_state[key] = v
        missing_keys, unexpected_keys = self.vit.load_state_dict(vit_state, strict=False)
        # fallback: keep original load if nothing matched
        if len(vit_state) == 0:
            self.vit.load_state_dict(state, strict=False)

    @staticmethod
    def _freeze_params(module):
        for p in module.parameters():
            p.requires_grad_(False)
        
        

def modality_unified_feature_extractor(cfg):
    model = ModalityUnifiedFeatureExtractor(cfg)
    return model
