import torch
import torch.nn.functional as F
from .position import PositionEmbeddingSine


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid(
        [
            torch.linspace(w_min, w_max, len_w, device=device),
            torch.linspace(h_min, h_max, len_h, device=device),
        ],
    )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2.0, (h - 1) / 2.0]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255.0 - mean) / std
    img1 = (img1 / 255.0 - mean) / std

    return img0, img1


def split_feature(
    feature,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        # Avoid 6D: merge batch with first split dimension incrementally
        # Step 1: [B, H, W, C] -> [B, K, H/K, W, C] (5D)
        feature = feature.view(b, num_splits, h_new, w, c)
        # Step 2: Merge B*K into batch -> [B*K, H/K, W, C] (4D)
        feature = feature.reshape(b * num_splits, h_new, w, c)
        # Step 3: Split width -> [B*K, H/K, K, W/K, C] (5D)
        feature = feature.view(b * num_splits, h_new, num_splits, w_new, c)
        # Step 4: Permute to group splits -> [B*K, K, H/K, W/K, C] (5D)
        feature = feature.permute(0, 2, 1, 3, 4)
        # Step 5: Merge into final batch -> [B*K*K, H/K, W/K, C]
        feature = feature.reshape(b_new, h_new, w_new, c)
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        # Avoid 6D: merge batch with first split dimension incrementally
        # Step 1: [B, C, H, W] -> [B, C, K, H/K, W] (5D)
        feature = feature.view(b, c, num_splits, h_new, w)
        # Step 2: Permute C and K -> [B, K, C, H/K, W] (5D)
        feature = feature.permute(0, 2, 1, 3, 4)
        # Step 3: Merge B*K into batch -> [B*K, C, H/K, W] (4D)
        feature = feature.reshape(b * num_splits, c, h_new, w)
        # Step 4: Split width -> [B*K, C, H/K, K, W/K] (5D)
        feature = feature.view(b * num_splits, c, h_new, num_splits, w_new)
        # Step 5: Permute to group splits -> [B*K, K, C, H/K, W/K] (5D)
        feature = feature.permute(0, 3, 1, 2, 4)
        # Step 6: Merge into final batch -> [B*K*K, C, H/K, W/K]
        feature = feature.reshape(b_new, c, h_new, w_new)

    return feature


def merge_splits(
    splits,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        # Avoid 6D: merge splits incrementally
        # Step 1: [B*K*K, H/K, W/K, C] -> [B*K, K, H/K, W/K, C] (5D)
        splits = splits.view(new_b * num_splits, num_splits, h, w, c)
        # Step 2: Permute to group width windows -> [B*K, H/K, K, W/K, C] (5D)
        splits = splits.permute(0, 2, 1, 3, 4)
        # Step 3: Merge width windows -> [B*K, H/K, W, C] (4D)
        splits = splits.reshape(new_b * num_splits, h, num_splits * w, c)
        # Step 4: Reshape to separate batch and height window -> [B, K, H/K, W, C] (5D)
        splits = splits.view(new_b, num_splits, h, num_splits * w, c)
        # Step 5: Permute to group height windows -> [B, K*H/K, W, C] but do it via permute
        #         Actually: [B, K, H/K, W, C] -> [B, H/K, K, W, C] ... wait, let me reconsider
        # Correction: [B, K, H/K, W, C] permute -> [B, H/K, K, W, C]
        splits = splits.permute(0, 2, 1, 3, 4)
        # Step 6: Merge height windows -> [B, H, W, C]
        merge = splits.reshape(new_b, num_splits * h, num_splits * w, c)
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        # Avoid 6D: merge splits incrementally (reverse of split_feature)
        # Step 1: [B*K*K, C, H/K, W/K] -> [B*K, K, C, H/K, W/K] (5D)
        splits = splits.view(new_b * num_splits, num_splits, c, h, w)
        # Step 2: Permute to prepare for width merge -> [B*K, C, H/K, K, W/K] (5D)
        splits = splits.permute(0, 2, 3, 1, 4)
        # Step 3: Merge width windows -> [B*K, C, H/K, W] (4D)
        splits = splits.reshape(new_b * num_splits, c, h, num_splits * w)
        # Step 4: Reshape to separate batch and height window -> [B, K, C, H/K, W] (5D)
        splits = splits.view(new_b, num_splits, c, h, num_splits * w)
        # Step 5: Permute to prepare for height merge -> [B, C, K, H/K, W] (5D)
        splits = splits.permute(0, 2, 1, 3, 4)
        # Step 6: Merge height windows -> [B, C, H, W]
        merge = splits.reshape(new_b, c, num_splits * h, num_splits * w)

    return merge


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1


def mv_feature_add_position(features, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    assert features.dim() == 4  # [B*V, C, H, W]

    if attn_splits > 1:  # add position in splited window
        features_splits = split_feature(features, num_splits=attn_splits)
        position = pos_enc(features_splits)
        features_splits = features_splits + position
        features = merge_splits(features_splits, num_splits=attn_splits)
    else:
        position = pos_enc(features)
        features = features + position

    return features
