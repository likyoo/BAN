from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.structures import PixelData
from mmengine.model.weight_init import caffe2_xavier_init

from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, SampleList, add_prefix
from mmseg.models.utils import nlc_to_nchw, LayerNorm2d, resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from opencd.registry import MODELS

from .ban_utils import BridgeLayer, MixFFN


class BAN_SCD_MLPDecoder(BaseDecodeHead):
    """
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        # projection head
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))
                
    def base_forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        
        return out

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        out = self.base_forward(inputs)

        out = self.discriminator(out)
        out = self.cls_seg(out)

        return out


class BAN_BCD_MLPDecoder(BaseDecodeHead):
    """
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i] * 2,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        # projection head
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))
                
    def forward(self, inputs1, inputs2):
        outs = []
        for idx in range(len(inputs1)):
            x = torch.cat([inputs1[idx], inputs2[idx]], dim=1)
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs1[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.discriminator(out)
        out = self.cls_seg(out)
        
        return out


class BitemporalAdapterNetwork(nn.Module):
    """The encoder of Bi-temporal Adapter Branch.

    Args:
        clip_channels (int): Number of channels of visual features.
            Default: 768.
        fusion_index (List[int]): The layer number of the encode
            transformer to fuse with the CLIP feature.
            Default: [0, 1, 2].
        side_enc_cfg (ConfigType): Configs for the encode layers.
    """

    def __init__(
            self,
            clip_channels: int = 768,
            fusion_index: list = [0, 1, 2],
            side_enc_cfg: ConfigType = ...,
    ):
        super().__init__()

        self.side_encoder = MODELS.build(side_enc_cfg)
        side_enc_channels = [num * self.side_encoder.embed_dims
                                for num in self.side_encoder.num_heads]

        conv_clips = []
        clip_attns = []
        for i in fusion_index:
            conv_clips.append(
                nn.Sequential(
                    LayerNorm2d(clip_channels),
                    ConvModule(
                        clip_channels,
                        side_enc_channels[i],
                        kernel_size=1,
                        norm_cfg=None,
                        act_cfg=None)))
            clip_attns.append(
                BridgeLayer(
                    num_heads=self.side_encoder.num_heads[i],
                    embed_dims=side_enc_channels[i],
                    kdim=None,
                    vdim=None))
        self.clip_attns = nn.ModuleList(clip_attns)
        self.conv_clips = nn.ModuleList(conv_clips)
        self.fusion_index = fusion_index

    def init_weights(self):
        self.side_encoder.init_weights()
        for i in range(len(self.conv_clips)):
            caffe2_xavier_init(self.conv_clips[i][1].conv)
        for i in range(len(self.clip_attns)):
            self.clip_attns[i].init_weights()

    def fuse_clip(self, fused_index: int, x: torch.Tensor,
                clip_feature: torch.Tensor):
        """Fuse CLIP feature and visual tokens."""
        clip_fea = self.conv_clips[fused_index](clip_feature.contiguous())
        fused_clip = self.clip_attns[fused_index](x, clip_fea)
        
        return fused_clip

    def encode_feature(self, x: torch.Tensor,
                       clip_features: List[torch.Tensor]) -> List[List]:
        """Encode images by a lightweight vision transformer."""
        
        outs = []
        fused_index = 0
        cls_token = False
        if isinstance(clip_features[0], list):
            cls_token = True

        for index, layer in enumerate(self.side_encoder.layers, start=0):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)

            if index in self.fusion_index:
                if cls_token:
                    x = self.fuse_clip(fused_index, x,
                                    clip_features[fused_index][0])
                else:
                    x = self.fuse_clip(fused_index, x,
                                    clip_features[fused_index])
                fused_index += 1
            outs.append(x)
        return outs

    def forward(
        self, image: torch.Tensor, clip_features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward function."""
        features = self.encode_feature(image, clip_features)
        return features


@MODELS.register_module()
class SCD_BitemporalAdapterHead(BaseDecodeHead):
    """Bi-Temporal Adapter Network (BAN) for Remote Sensing
    Image Change Detection.

    Args:
        ban_cfg (ConfigType): Configs for BitemporalAdapterNetwork
        ban_bcd_dec_cfg (ConfigType): Configs for Bi-TAB's BCD decoder
        ban_scd_dec_cfg (ConfigType): Configs for Bi-TAB's SCD decoder
    """

    def __init__(self,
                 ban_cfg: ConfigType,
                 ban_bcd_dec_cfg: ConfigType,
                 ban_scd_dec_cfg: ConfigType,
                 **kwargs):
        super().__init__(
            in_channels=ban_cfg.side_enc_cfg.in_channels,
            channels=ban_cfg.side_enc_cfg.embed_dims,
            num_classes=ban_bcd_dec_cfg.num_classes,
            **kwargs)

        del self.conv_seg

        self.side_adapter_network = BitemporalAdapterNetwork(**ban_cfg)
        self.binary_cd_head = BAN_BCD_MLPDecoder(**ban_bcd_dec_cfg)
        self.semantic_cd_head = BAN_SCD_MLPDecoder(**ban_scd_dec_cfg)

    def init_weights(self):
        self.side_adapter_network.init_weights()

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[List]:
        """Forward function.

        Args:
            inputs (Tuple[Tensor]): A pair including images,
            list of multi-level visual features from image encoder.

        Returns:
            output (List[Tensor]): Mask predicted by BAN.
        """
        img_from, img_to, fm_feat_from, fm_feat_to = inputs

        mask_props_from = self.side_adapter_network(
            img_from, fm_feat_from)
        
        mask_props_to = self.side_adapter_network(
            img_to, fm_feat_to)
        
        out = self.binary_cd_head(mask_props_from, mask_props_to)
        out1 = self.semantic_cd_head(mask_props_from)
        out2 = self.semantic_cd_head(mask_props_to)

        out_dict = dict(
            seg_logits=out,
            seg_logits_from=out1, 
            seg_logits_to=out2
        )

        return out_dict

    def predict(self, inputs: Tuple[torch.Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> torch.Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): Images, visual features from image encoder
            and class embedding from text encoder.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        mask_props = self.forward(inputs)

        return self.predict_by_feat(mask_props,
                                    batch_img_metas)

    def loss(self, x: Tuple[torch.Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # forward
        seg_logits = self.forward(x)

        # loss
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        return losses

    def predict_by_feat(self, seg_logits: torch.Tensor,
                        batch_img_metas: List[dict]) -> torch.Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        assert ['seg_logits', 'seg_logits_from', 'seg_logits_to'] \
            == list(seg_logits.keys()), "`seg_logits`, `seg_logits_from` \
            and `seg_logits_to` should be contained."

        self.align_corners = {
            'seg_logits': self.binary_cd_head.align_corners,
            'seg_logits_from': self.semantic_cd_head.align_corners,
            'seg_logits_to': self.semantic_cd_head.align_corners}

        for seg_name, seg_logit in seg_logits.items():
            seg_logits[seg_name] = resize(
                input=seg_logit,
                size=batch_img_metas[0]['img_shape'],
                mode='bilinear',
                align_corners=self.align_corners[seg_name])
        return seg_logits
    
    def get_sub_batch_data_samples(self, batch_data_samples: SampleList, 
                                   sub_metainfo_name: str,
                                   sub_data_name: str) -> list:
        sub_batch_sample_list = []
        for i in range(len(batch_data_samples)):
            data_sample = SegDataSample()

            gt_sem_seg_data = dict(
                data=batch_data_samples[i].get(sub_data_name).data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

            img_meta = {}
            seg_map_path = batch_data_samples[i].metainfo.get(sub_metainfo_name)
            for key in batch_data_samples[i].metainfo.keys():
                if not 'seg_map_path' in key:
                    img_meta[key] = batch_data_samples[i].metainfo.get(key)
            img_meta['seg_map_path'] = seg_map_path
            data_sample.set_metainfo(img_meta)

            sub_batch_sample_list.append(data_sample)
        return sub_batch_sample_list
    
    def loss_by_feat(self, seg_logits: dict,
                     batch_data_samples: SampleList, **kwargs) -> dict:
        """Compute segmentation loss."""
        assert ['seg_logits', 'seg_logits_from', 'seg_logits_to'] \
            == list(seg_logits.keys()), "`seg_logits`, `seg_logits_from` \
            and `seg_logits_to` should be contained."

        losses = dict()
        binary_cd_loss_decode = self.binary_cd_head.loss_by_feat(
            seg_logits['seg_logits'],
            self.get_sub_batch_data_samples(batch_data_samples,
                                            sub_metainfo_name='seg_map_path',
                                            sub_data_name='gt_sem_seg'))
        losses.update(add_prefix(binary_cd_loss_decode, 'binary_cd'))

        if getattr(self, 'semantic_cd_head'):
            semantic_cd_loss_decode_from = self.semantic_cd_head.loss_by_feat(
                seg_logits['seg_logits_from'],
                self.get_sub_batch_data_samples(batch_data_samples,
                                                sub_metainfo_name='seg_map_path_from',
                                                sub_data_name='gt_sem_seg_from'))
            losses.update(add_prefix(semantic_cd_loss_decode_from, 'semantic_cd_from'))

            semantic_cd_loss_decode_to = self.semantic_cd_head.loss_by_feat(
                seg_logits['seg_logits_to'],
                self.get_sub_batch_data_samples(batch_data_samples,
                                                sub_metainfo_name='seg_map_path_to',
                                                sub_data_name='gt_sem_seg_to'))
            losses.update(add_prefix(semantic_cd_loss_decode_to, 'semantic_cd_to'))

        return losses