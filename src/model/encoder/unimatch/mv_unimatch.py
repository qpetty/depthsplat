import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Patch xformers globally to use PyTorch's attention as fallback (for macOS compatibility)
# This needs to happen before DINOv2 is loaded
try:
    import xformers.ops as xops
    if hasattr(xops, 'memory_efficient_attention'):
        original_mea = xops.memory_efficient_attention
        
        def patched_memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None):
            try:
                return original_mea(query, key, value, attn_bias=attn_bias, p=p, scale=scale)
            except (NotImplementedError, RuntimeError) as e:
                # Fall back to PyTorch's attention if xformers fails
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['memory_efficient_attention', 'xformers', 'not supported', 'not implemented', 'device=cpu', 'dtype=torch.float32']):
                    # Convert to format expected by scaled_dot_product_attention
                    # xformers format: (batch, seq_len, num_heads, head_dim)
                    # PyTorch format: (batch, num_heads, seq_len, head_dim)
                    if query.dim() == 4:
                        q = query.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
                        k = key.transpose(1, 2)
                        v = value.transpose(1, 2)
                    else:
                        q, k, v = query, key, value
                    
                    # Handle attn_bias if provided
                    attn_mask = None
                    if attn_bias is not None:
                        if hasattr(attn_bias, 'materialize'):
                            # xformers LowerTriangularMask or similar
                            try:
                                attn_mask = attn_bias.materialize(
                                    (query.shape[0], query.shape[1], key.shape[1]), 
                                    device=query.device, 
                                    dtype=query.dtype
                                )
                            except:
                                attn_mask = None
                        elif isinstance(attn_bias, torch.Tensor):
                            attn_mask = attn_bias
                    
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p, scale=scale)
                    
                    # Convert back to xformers format if needed
                    if query.dim() == 4:
                        out = out.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]
                    return out
                else:
                    raise
        
        xops.memory_efficient_attention = patched_memory_efficient_attention
except ImportError:
    # xformers not available, nothing to patch
    pass

from .backbone import CNNEncoder
from .vit_fpn import ViTFeaturePyramid
from .mv_transformer import (
    MultiViewFeatureTransformer,
    batch_features_camera_parameters,
)
from .matching import warp_with_pose_depth_candidates
from .utils import mv_feature_add_position
from .dpt_head import DPTHead
from .ldm_unet.unet import UNetModel, AttentionBlock
from einops import rearrange


def _patch_dinov2_attention(model):
    """
    Patch DINOv2 attention layers to use PyTorch's scaled_dot_product_attention
    instead of xformers. This allows the model to work on macOS where xformers
    is not available or doesn't support the current device/dtype.
    """
    import torch.nn.functional as F
    
    def patched_attention_forward(self, qkv, attn_bias=None):
        """
        Patched attention forward that uses PyTorch's scaled_dot_product_attention
        instead of xformers.memory_efficient_attention.
        """
        # DINOv2 attention expects qkv in a specific format
        # The original uses xformers.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        # We need to extract q, k, v from the attention module's internal structure
        
        # Try to get q, k, v from the module's internal state
        # The attention module in DINOv2 processes qkv internally
        # We need to patch at the block level, not the attention level
        
        # For now, we'll patch the block's attention forward method
        # The actual patching happens in the block's forward method
        pass
    
    # Patch all attention blocks in the model
    for name, module in model.named_modules():
        # DINOv2 uses attention layers in blocks
        # The attention is typically in a 'attn' attribute of blocks
        if hasattr(module, 'attn') and hasattr(module.attn, 'forward'):
            original_forward = module.attn.forward
            
            def make_patched_forward(orig_fwd, attn_module):
                def patched_forward(x, attn_bias=None):
                    # Try to use PyTorch's attention if xformers fails
                    try:
                        return orig_fwd(x, attn_bias=attn_bias)
                    except (NotImplementedError, RuntimeError) as e:
                        # If xformers fails (e.g., on CPU or unsupported dtype),
                        # fall back to PyTorch's scaled_dot_product_attention
                        if 'memory_efficient_attention' in str(e) or 'xformers' in str(e).lower():
                            # Extract q, k, v from the attention module
                            # DINOv2 attention structure: it has qkv projection
                            if hasattr(attn_module, 'qkv'):
                                qkv_proj = attn_module.qkv(x)
                                # Reshape and split qkv
                                # Format depends on DINOv2's internal structure
                                # This is a simplified version - may need adjustment
                                B, N, C = qkv_proj.shape
                                head_dim = C // (3 * attn_module.num_heads)
                                qkv = qkv_proj.reshape(B, N, 3, attn_module.num_heads, head_dim)
                                q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
                                q, k, v = q[0], k[1], v[2]  # Extract q, k, v
                                
                                # Use PyTorch's scaled_dot_product_attention
                                out = F.scaled_dot_product_attention(q, k, v, attn_bias=attn_bias)
                                
                                # Reshape back
                                out = out.reshape(B, N, C)
                                
                                # Apply output projection if exists
                                if hasattr(attn_module, 'proj'):
                                    out = attn_module.proj(out)
                                
                                return out
                            else:
                                # Fallback: re-raise the original error
                                raise
                        else:
                            raise
                
                return patched_forward
            
            module.attn.forward = make_patched_forward(original_forward, module.attn)
    
    # Also patch the xformers import at the module level if possible
    # This is a more aggressive approach - monkey-patch xformers.ops.memory_efficient_attention
    try:
        import xformers.ops as xops
        original_mea = xops.memory_efficient_attention
        
        def patched_memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None):
            try:
                return original_mea(query, key, value, attn_bias=attn_bias, p=p, scale=scale)
            except (NotImplementedError, RuntimeError) as e:
                # Fall back to PyTorch's attention
                if 'memory_efficient_attention' in str(e) or 'xformers' in str(e).lower() or 'not supported' in str(e).lower():
                    # Convert to format expected by scaled_dot_product_attention
                    # xformers format: (batch, seq_len, num_heads, head_dim)
                    # PyTorch format: (batch, num_heads, seq_len, head_dim)
                    q = query.transpose(1, 2) if query.dim() == 4 else query
                    k = key.transpose(1, 2) if key.dim() == 4 else key
                    v = value.transpose(1, 2) if value.dim() == 4 else value
                    
                    # Handle attn_bias if provided (simplified - may need more work for complex biases)
                    attn_mask = None
                    if attn_bias is not None:
                        # Convert attn_bias to attn_mask format if needed
                        # This is a simplified conversion
                        if hasattr(attn_bias, 'materialize'):
                            # xformers LowerTriangularMask or similar
                            attn_mask = attn_bias.materialize((query.shape[0], query.shape[1], key.shape[1]), 
                                                             device=query.device, dtype=query.dtype)
                        elif isinstance(attn_bias, torch.Tensor):
                            attn_mask = attn_bias
                    
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p, scale=scale)
                    
                    # Convert back to xformers format
                    out = out.transpose(1, 2) if out.dim() == 4 else out
                    return out
                else:
                    raise
        
        xops.memory_efficient_attention = patched_memory_efficient_attention
    except ImportError:
        # xformers not available, nothing to patch
        pass


class MultiViewUniMatch(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        grid_sample_disable_cudnn=False,
        **kwargs,
    ):
        super(MultiViewUniMatch, self).__init__()

        # CNN
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        # monocular backbones: final
        self.vit_type = vit_type

        # cost volume
        self.num_depth_candidates = num_depth_candidates

        # upsampler
        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        # CNN
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        if self.num_scales > 1:
            # generate multi-scale features
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128, scale_factors=[2**i for i in range(self.num_scales)]
            )

        # monodepth
        encoder = vit_type  # can also be 'vitb' or 'vitl'
        self.pretrained = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{:}14".format(encoder)
        )

        del self.pretrained.mask_token  # unused
        
        # Patch DINOv2 attention to use PyTorch's scaled_dot_product_attention
        # instead of xformers (for macOS compatibility)
        _patch_dinov2_attention(self.pretrained)

        if self.num_scales > 1:
            # generate multi-scale features
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=vit_feature_channel,
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        # UNet regressor
        self.regressor = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head = nn.ModuleList()

        for i in range(self.num_scales):
            curr_depth_candidates = num_depth_candidates // (4**i)
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2**i)

            mono_feature_channels = vit_feature_channel // (2**i)

            # concat(cost volume, cnn feature, mv feature, mono feature)
            in_channels = (
                curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

            # unet channels
            channels = unet_channels // (2**i)

            # unet channel mult & unet_attn_resolutions
            if i > 0:
                unet_channel_mult = unet_channel_mult + [1]
                unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

            # unet
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=unet_num_res_blocks,
                    attention_resolutions=unet_attn_resolutions,
                    channel_mult=unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                )
            )

            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

            self.regressor.append(nn.Sequential(*modules))

            # regressor residual
            self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))

            # depth head
            self.depth_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels * 2, 3, 1, 1, padding_mode="replicate"
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        channels * 2,
                        curr_depth_candidates,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                )
            )

        # upsampler
        # concat(lowres_depth, cnn feature, mv feature, mono feature)
        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.upsampler = DPTHead(
            **model_configs[vit_type],
            downsample_factor=upsample_factor,
            num_scales=num_scales,
        )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        # images: [B, V, C, H, W]
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        # list of [BV, C, H, W], resolution from high to low
        features = self.backbone(concat)
        # reverse: resolution from low to high
        features = features[::-1]

        return features

    def forward(
        self,
        images,
        attn_splits_list=None,
        intrinsics=None,
        min_depth=1.0 / 0.5,  # inverse depth range
        max_depth=1.0 / 100,
        num_depth_candidates=128,
        extrinsics=None,
        nn_matrix=None,
        **kwargs,
    ):

        results_dict = {}
        depth_preds = []
        match_probs = []

        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.regressor, num_views=v)

        # NOTE: in this codebase, intrinsics are normalized by image width and height
        # in unimatch's codebase: https://github.com/autonomousvision/unimatch, no normalization
        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0] *= ori_w
        intrinsics[:, :, 1] *= ori_h

        # max_depth, min_depth: [B, V] -> [BV]
        max_depth = max_depth.view(-1)
        min_depth = min_depth.view(-1)

        # list of features, resolution low to high
        # list of [BV, C, H, W]
        cnn_feature_start = time.perf_counter()
        cnn_feature_start_wall = time.time()
        print(f"      [MultiViewUniMatch] CNN feature extraction start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cnn_feature_start_wall))}")
        
        features_list_cnn = self.extract_feature(images)
        
        cnn_feature_end = time.perf_counter()
        cnn_feature_end_wall = time.time()
        cnn_feature_elapsed = cnn_feature_end - cnn_feature_start
        print(f"      [MultiViewUniMatch] CNN feature extraction end: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cnn_feature_end_wall))} (elapsed: {cnn_feature_elapsed:.3f}s)")
        features_list_cnn_all_scales = features_list_cnn
        features_list_cnn = features_list_cnn[: self.num_scales]
        results_dict.update({"features_cnn_all_scales": features_list_cnn_all_scales})
        results_dict.update({"features_cnn": features_list_cnn})

        # mv transformer features
        # add position to features
        attn_splits = attn_splits_list[0]

        # [BV, C, H, W]
        features_cnn_pos = mv_feature_add_position(
            features_list_cnn[0], attn_splits, self.feature_channels
        )

        # list of [B, C, H, W]
        features_list = list(
            torch.unbind(
                rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
            )
        )
        
        mv_transformer_start = time.perf_counter()
        mv_transformer_start_wall = time.time()
        print(f"      [MultiViewUniMatch] Multi-view transformer start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mv_transformer_start_wall))}")
        
        features_list_mv = self.transformer(
            features_list,
            attn_num_splits=attn_splits,
            nn_matrix=nn_matrix,
        )
        
        mv_transformer_end = time.perf_counter()
        mv_transformer_end_wall = time.time()
        mv_transformer_elapsed = mv_transformer_end - mv_transformer_start
        print(f"      [MultiViewUniMatch] Multi-view transformer end: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mv_transformer_end_wall))} (elapsed: {mv_transformer_elapsed:.3f}s)")

        features_mv = rearrange(
            torch.stack(features_list_mv, dim=1), "b v c h w -> (b v) c h w"
        )  # [BV, C, H, W]

        if self.num_scales > 1:
            # multi-scale mv features: resolution from low to high
            # list of [BV, C, H, W]
            features_list_mv = self.mv_pyramid(features_mv)
        else:
            features_list_mv = [features_mv]

        results_dict.update({"features_mv": features_list_mv})

        # mono feature
        ori_h, ori_w = images.shape[-2:]
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        concat = F.interpolate(
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        )

        # get intermediate features
        intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }

        mono_feature_start = time.perf_counter()
        mono_feature_start_wall = time.time()
        print(f"      [MultiViewUniMatch] Monocular feature extraction (DINOv2) start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mono_feature_start_wall))}")
        
        mono_intermediate_features = list(
            self.pretrained.get_intermediate_layers(
                concat, intermediate_layer_idx[self.vit_type], return_class_token=False
            )
        )

        for i in range(len(mono_intermediate_features)):
            curr_features = (
                mono_intermediate_features[i]
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # resize to 1/8 resolution
            curr_features = F.interpolate(
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            mono_intermediate_features[i] = curr_features
        
        mono_feature_end = time.perf_counter()
        mono_feature_end_wall = time.time()
        mono_feature_elapsed = mono_feature_end - mono_feature_start
        print(f"      [MultiViewUniMatch] Monocular feature extraction (DINOv2) end: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mono_feature_end_wall))} (elapsed: {mono_feature_elapsed:.3f}s)")

        results_dict.update({"features_mono_intermediate": mono_intermediate_features})

        # last mono feature
        mono_features = mono_intermediate_features[-1]

        if self.lowest_feature_resolution == 4:
            mono_features = F.interpolate(
                mono_features, scale_factor=2, mode="bilinear", align_corners=True
            )

        if self.num_scales > 1:
            # multi-scale mono features, resolution from low to high
            # list of [BV, C, H, W]
            features_list_mono = self.mono_pyramid(mono_features)
        else:
            features_list_mono = [mono_features]

        results_dict.update({"features_mono": features_list_mono})

        depth = None

        multiscale_loop_start = time.perf_counter()
        multiscale_loop_start_wall = time.time()
        print(f"      [MultiViewUniMatch] Multi-scale depth prediction loop start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(multiscale_loop_start_wall))} (num_scales={self.num_scales})")

        for scale_idx in range(self.num_scales):
            scale_start = time.perf_counter()
            print(f"        [MultiViewUniMatch] Processing scale {scale_idx+1}/{self.num_scales}...")
            downsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )

            # scale intrinsics
            intrinsics_curr = intrinsics.clone()  # [B, V, 3, 3]
            intrinsics_curr[:, :, :2] = intrinsics_curr[:, :, :2] / downsample_factor

            # build cost volume
            features_mv = features_list_mv[scale_idx]  # [BV, C, H, W]

            # list of [B, C, H, W]
            features_mv_curr = list(
                torch.unbind(
                    rearrange(features_mv, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
                )
            )

            intrinsics_curr = list(
                torch.unbind(intrinsics_curr, dim=1)
            )  # list of [B, 3, 3]
            extrinsics_curr = list(torch.unbind(extrinsics, dim=1))  # list of [B, 4, 4]

            # ref: [BV, C, H, W], [BV, 3, 3], [BV, 4, 4]
            # tgt: [BV, V-1, C, H, W], [BV, V-1, 3, 3], [BV, V-1, 4, 4]
            (
                ref_features,
                ref_intrinsics,
                ref_extrinsics,
                tgt_features,
                tgt_intrinsics,
                tgt_extrinsics,
            ) = batch_features_camera_parameters(
                features_mv_curr,
                intrinsics_curr,
                extrinsics_curr,
                nn_matrix=nn_matrix,
            )

            b_new, _, c, h, w = tgt_features.size()

            # relative pose
            # extrinsics: c2w
            # NOTE: Avoid generic matrix inverse (torch.inverse / torch.linalg.inv)
            # when exporting (ONNX / torch.export / CoreML), since they can decompose
            # to ops without converter support. Use an analytical rigid-transform
            # inverse instead, assuming standard 4x4 [[R, t], [0, 1]] structure.
            R = tgt_extrinsics[..., :3, :3]
            t = tgt_extrinsics[..., :3, 3]
            R_inv = R.transpose(-1, -2)
            t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)
            tgt_extrinsics_inv = torch.zeros_like(tgt_extrinsics)
            tgt_extrinsics_inv[..., :3, :3] = R_inv
            tgt_extrinsics_inv[..., :3, 3] = t_inv
            tgt_extrinsics_inv[..., 3, 3] = 1.0

            pose_curr = torch.matmul(
                tgt_extrinsics_inv, ref_extrinsics.unsqueeze(1)
            )  # [BV, V-1, 4, 4]

            if scale_idx > 0:
                # 2x upsample depth
                assert depth is not None
                depth = F.interpolate(
                    depth, scale_factor=2, mode="bilinear", align_corners=True
                ).detach()

            num_depth_candidates = self.num_depth_candidates // (4**scale_idx)

            # generate depth candidates
            if scale_idx == 0:
                # min_depth, max_depth: [BV]
                depth_interval = (max_depth - min_depth) / (
                    self.num_depth_candidates - 1
                )  # [BV]

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]

                depth_candidates = min_depth.view(-1, 1, 1, 1) + linear_space * (
                    max_depth - min_depth
                ).view(
                    -1, 1, 1, 1
                )  # [BV, D, 1, 1]
            else:
                # half interval each scale
                depth_interval = (
                    (max_depth - min_depth)
                    / (self.num_depth_candidates - 1)
                    / (2**scale_idx)
                )  # [BV]
                # [BV, 1, 1, 1]
                depth_interval = depth_interval.view(-1, 1, 1, 1)

                # [BV, 1, H, W]
                depth_range_min = torch.maximum(
                    depth - depth_interval * (num_depth_candidates // 2),
                    min_depth.view(-1, 1, 1, 1)
                )
                depth_range_max = torch.minimum(
                    depth + depth_interval * (num_depth_candidates // 2 - 1),
                    max_depth.view(-1, 1, 1, 1)
                )

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]
                depth_candidates = depth_range_min + linear_space * (
                    depth_range_max - depth_range_min
                )  # [BV, D, H, W]

            if scale_idx == 0:
                # [BV*(V-1), D, H, W]
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, h, w)
                    .view(-1, num_depth_candidates, h, w)
                )
            else:
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, 1, 1)
                    .view(-1, num_depth_candidates, h, w)
                )

            intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(
                -1, 3, 3
            )  # [BV, 3, 3]
            intrinsics_input = intrinsics_input.unsqueeze(1).repeat(
                1, tgt_features.size(1), 1, 1
            )  # [BV, V-1, 3, 3]

            cost_volume_start = time.perf_counter()
            
            warped_tgt_features_flat = warp_with_pose_depth_candidates(
                rearrange(tgt_features, "b v ... -> (b v) ..."),
                rearrange(intrinsics_input, "b v ... -> (b v) ..."),
                rearrange(pose_curr, "b v ... -> (b v) ..."),
                1.0 / depth_candidates_curr,  # convert inverse depth to depth
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            )  # [BV*(V-1), C, D, H, W]

            # ref: [BV, C, H, W]
            # warped: [BV*(V-1), C, D, H, W]
            # AVOID 6D TENSORS: Don't rearrange to [BV, V-1, C, D, H, W]!
            # Instead, chunk the batch dimension and process each view separately
            num_views = tgt_features.size(1)
            cost_volumes = []
            for v_idx in range(num_views):
                # Extract features for this target view: [BV, C, D, H, W] (5D)
                warped_v = warped_tgt_features_flat[v_idx * b_new : (v_idx + 1) * b_new]
                # ref: [BV, C, H, W] -> [BV, C, 1, H, W] (5D)
                # warped_v: [BV, C, D, H, W] (5D)
                # multiply and sum over C: [BV, D, H, W] (4D)
                cost_v = (ref_features.unsqueeze(2) * warped_v).sum(1) / (c**0.5)
                cost_volumes.append(cost_v)
            # Stack and mean over views: [V-1, BV, D, H, W] -> [BV, D, H, W]
            cost_volume = torch.stack(cost_volumes, dim=0).mean(0)
            
            cost_volume_end = time.perf_counter()
            cost_volume_elapsed = cost_volume_end - cost_volume_start
            print(f"          [MultiViewUniMatch] Scale {scale_idx+1} cost volume building: {cost_volume_elapsed:.3f}s")

            # regressor
            features_cnn = features_list_cnn[scale_idx]  # [BV, C, H, W]

            features_mono = features_list_mono[scale_idx]  # [BV, C, H, W]

            concat = torch.cat(
                (cost_volume, features_cnn, features_mv, features_mono), dim=1
            )

            regressor_start = time.perf_counter()
            
            out = self.regressor[scale_idx](concat) + self.regressor_residual[
                scale_idx
            ](concat)

            # depth pred
            match_prob = F.softmax(
                self.depth_head[scale_idx](out), dim=1
            )  # [BV, D, H, W]
            
            regressor_end = time.perf_counter()
            regressor_elapsed = regressor_end - regressor_start
            print(f"          [MultiViewUniMatch] Scale {scale_idx+1} regressor + depth head: {regressor_elapsed:.3f}s")
            
            match_probs.append(match_prob)

            if scale_idx == 0:
                # [BV, D, H, W]
                depth_candidates = depth_candidates.repeat(1, 1, h, w)
            depth = (match_prob * depth_candidates).sum(
                dim=1, keepdim=True
            )  # [BV, 1, H, W]

            # upsample to the original resolution for supervison at training time only
            if self.training and scale_idx < self.num_scales - 1:
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=downsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth_preds.append(depth_bilinear)

            # final output, learned upsampler
            if scale_idx == self.num_scales - 1:
                upsampler_start = time.perf_counter()
                
                residual_depth = self.upsampler(
                    mono_intermediate_features,
                    # resolution high to low
                    cnn_features=features_list_cnn_all_scales[::-1],
                    mv_features=(
                        features_mv if self.num_scales == 1 else features_list_mv[::-1]
                    ),
                    depth=depth,
                )

                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=self.upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth = torch.minimum(
                    torch.maximum(depth_bilinear + residual_depth, min_depth.view(-1, 1, 1, 1)),
                    max_depth.view(-1, 1, 1, 1)
                )
                
                upsampler_end = time.perf_counter()
                upsampler_elapsed = upsampler_end - upsampler_start
                print(f"          [MultiViewUniMatch] Final upsampler: {upsampler_elapsed:.3f}s")

                depth_preds.append(depth)
            
            scale_end = time.perf_counter()
            scale_elapsed = scale_end - scale_start
            print(f"        [MultiViewUniMatch] Scale {scale_idx+1} total: {scale_elapsed:.3f}s")

        multiscale_loop_end = time.perf_counter()
        multiscale_loop_end_wall = time.time()
        multiscale_loop_elapsed = multiscale_loop_end - multiscale_loop_start
        print(f"      [MultiViewUniMatch] Multi-scale depth prediction loop end: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(multiscale_loop_end_wall))} (elapsed: {multiscale_loop_elapsed:.3f}s)")

        # convert inverse depth to depth
        depth_convert_start = time.perf_counter()
        
        for i in range(len(depth_preds)):
            depth_pred = 1.0 / depth_preds[i].squeeze(1)  # [BV, H, W]
            depth_preds[i] = rearrange(
                depth_pred, "(b v) ... -> b v ...", b=b, v=v
            )  # [B, V, H, W]
        
        depth_convert_end = time.perf_counter()
        depth_convert_elapsed = depth_convert_end - depth_convert_start
        print(f"      [MultiViewUniMatch] Depth conversion (inverse to depth): {depth_convert_elapsed:.3f}s")

        results_dict.update({"depth_preds": depth_preds})
        results_dict.update({"match_probs": match_probs})

        # Print MultiViewUniMatch component timing summary
        print(f"      [MultiViewUniMatch] Component timing summary:")
        print(f"        CNN feature extraction: {cnn_feature_elapsed:.3f}s")
        print(f"        Multi-view transformer: {mv_transformer_elapsed:.3f}s")
        print(f"        Monocular feature extraction (DINOv2): {mono_feature_elapsed:.3f}s")
        print(f"        Multi-scale depth prediction loop: {multiscale_loop_elapsed:.3f}s")
        print(f"        Depth conversion: {depth_convert_elapsed:.3f}s")
        total_mv_unimatch_time = cnn_feature_elapsed + mv_transformer_elapsed + mono_feature_elapsed + multiscale_loop_elapsed + depth_convert_elapsed
        print(f"        Total MultiViewUniMatch time: {total_mv_unimatch_time:.3f}s")

        return results_dict


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
        isinstance(module, nn.ModuleList)
        or isinstance(module, nn.Sequential)
        or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)
