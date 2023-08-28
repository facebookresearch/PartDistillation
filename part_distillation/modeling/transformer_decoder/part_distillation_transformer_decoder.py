# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


@TRANSFORMER_DECODER_REGISTRY.register()
class PartDistillationTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification,
        # *,
        # num_classes: int,
        # hidden_dim: int,
        # num_queries: int,
        # nheads: int,
        # dim_feedforward: int,
        # dec_layers: int,
        # pre_norm: bool,
        # mask_dim: int,
        *args,
        num_object_classes: int,
        num_part_classes: int,
        **kwargs,
    ):
        super().__init__(in_channels, mask_classification, *args, **kwargs)
        # assert mask_classification, "Only support mask classification model"
        # self.mask_classification = mask_classification

        # # positional encoding
        # N_steps = hidden_dim // 2
        # self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # # define Transformer decoder here
        # self.num_heads = nheads
        # self.num_layers = dec_layers
        # self.transformer_self_attention_layers = nn.ModuleList()
        # self.transformer_cross_attention_layers = nn.ModuleList()
        # self.transformer_ffn_layers = nn.ModuleList()

        # for _ in range(self.num_layers):
        #     self.transformer_self_attention_layers.append(
        #         SelfAttentionLayer(
        #             d_model=hidden_dim,
        #             nhead=nheads,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )

        #     self.transformer_cross_attention_layers.append(
        #         CrossAttentionLayer(
        #             d_model=hidden_dim,
        #             nhead=nheads,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )

        #     self.transformer_ffn_layers.append(
        #         FFNLayer(
        #             d_model=hidden_dim,
        #             dim_feedforward=dim_feedforward,
        #             dropout=0.0,
        #             normalize_before=pre_norm,
        #         )
        #     )

        # self.decoder_norm = nn.LayerNorm(hidden_dim)

        # self.num_queries = num_queries
        # # learnable query features
        # self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # # learnable query p.e.
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # # level embedding (we always use 3 scales)
        # self.num_feature_levels = 3
        # self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        # self.input_proj = nn.ModuleList()
        # for _ in range(self.num_feature_levels):
        #     if in_channels != hidden_dim or enforce_input_project:
        #         self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
        #         weight_init.c2_xavier_fill(self.input_proj[-1])
        #     else:
        #         self.input_proj.append(nn.Sequential())

        # output FFNs
        # if self.mask_classification:
        #     self.class_embed = nn.Linear(self.hidden_dim, num_part_classes * num_object_classes + 1).double()
        self.class_embed = nn.Linear(self.hidden_dim, num_part_classes * num_object_classes + 1).double()
        # self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        # self.query_feature_normalize = query_feature_normalize
        self.num_part_classes = num_part_classes


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["num_object_classes"] = cfg.PART_DISTILLATION.NUM_OBJECT_CLASSES
        ret["num_part_classes"] = cfg.PART_DISTILLATION.NUM_PART_CLASSES
        ret["query_feature_normalize"] = cfg.MODEL.MASK_FORMER.QUERY_FEATURE_NORMALIZE
        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # NOTE: We abuse [mask] argument, but it works for now
        targets = mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, _ = self.forward_prediction_heads(output, mask_features, targets,
                                                        attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, decoder_output = self.forward_prediction_heads(output, mask_features, targets,
                                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)


        out = {
            'query_feats': output.permute(1, 0, 2),
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'decoder_output': decoder_output,
        }

        return out

    def apply_gradient_mask(self, outputs, targets):
        # outputs: BxQxC
        new_outputs = []
        for i, target_per_image in enumerate(targets):
            start_idx = target_per_image["gt_object_class"] * self.num_part_classes
            end_idx = (target_per_image["gt_object_class"] + 1) * self.num_part_classes

            new_outputs.append(outputs[i][:, start_idx:end_idx])

        new_outputs = torch.stack(new_outputs, dim=0)
        new_outputs = torch.cat([new_outputs, outputs[:, :, -1:]], dim=-1)

        # NOTE: Ugly trick to make pytorch optimizer happy ...
        new_outputs = new_outputs + (outputs.sum() * 0)

        return new_outputs


    def forward_prediction_heads(self, output, mask_features, targets, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1) # BxQxC

        outputs_class = self.class_embed(decoder_output.double())
        outputs_class = self.apply_gradient_mask(outputs_class, targets)
        mask_embed = self.mask_embed(decoder_output)

        if self.query_feature_normalize:
            mask_embed = F.normalize(mask_embed, p=2, dim=-1)

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, decoder_output
