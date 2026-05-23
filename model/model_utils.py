import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
from functools import partial


model_sizes = { # small -> xl same as mamba https://arxiv.org/pdf/2312.00752
    "4xs": {
        "num_transformer_blocks": 2,
        "multiheaded_attention_heads": 2,
        "embedding_dim": 128,
    },
    "3xs": {
        "num_transformer_blocks": 4,
        "multiheaded_attention_heads": 4,
        "embedding_dim": 256,
    },
    "xxs": {
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "2xs": { # same as xxs
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "xs": {
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "small": {
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 12,
        "embedding_dim": 768,
    },
    "medium": {
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1024,
    },
    "large": {
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1536,
    },
    "xl": {
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 32,
        "embedding_dim": 2048,
    },
}


def get_cv_transforms(dataset_name, image_dim, custom_image_normalization=True, vae_normalization=False):

    normal_lookup = { #NOTE is std, mean
        "ucf101": ([1.04731617, 1.04372056, 1.02795228], [-0.40689788, -0.36098219, -0.25687788]),
        "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
        "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
        "ImageNet": ([1, 1, 1], [0, 0, 0])
    }
    normal_lookup["something"] = normal_lookup["smth"]
    normal_lookup["ssv2"] = normal_lookup["smth"]
    normal_lookup["ImageNet1k"] = normal_lookup["ImageNet"]
    normal_lookup["imagenet1k"] = normal_lookup["ImageNet"]

    if custom_image_normalization:
        transform = transforms.Compose([
            transforms.Resize((image_dim[0], image_dim[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if dataset_name in normal_lookup:
            std, mean = normal_lookup[dataset_name]
            transform.transforms.append(transforms.Normalize(mean=mean, std=std))
        elif dataset_name in ["aggregate"]: # these are combined datasets
            pass
        else:
            raise ValueError(f"{dataset_name} not in normal lookup")

    else:
        if vae_normalization:
            transform = transforms.Compose([
                transforms.Resize((image_dim[0], image_dim[1])),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else: # imagenet standardization
            transform = transforms.Compose([
                transforms.Resize((image_dim[0], image_dim[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    return transform, normal_lookup


def denormalize(tensor, dataset_name, device, custom_normalization):
    tensor = tensor.clone().detach()

    # Define default normalization values
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    default_mean = torch.tensor(default_mean, device=device).view(1, 1, 3, 1, 1)
    default_std = torch.tensor(default_std, device=device).view(1, 1, 3, 1, 1)
    # Dataset-specific normalization lookup
    if custom_normalization:
        normal_lookup = {
            "ucf101": ([1.04731617, 1.04372056, 1.02795228], [-0.40689788, -0.36098219, -0.25687788]),
            "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
            "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet": ([1, 1, 1], [0, 0, 0]),
            "something": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet1k": ([1, 1, 1], [0, 0, 0])
        }
        dataset_std, dataset_mean = normal_lookup.get(dataset_name, ([1, 1, 1], [0, 0, 0]))

        # Convert means and stds to tensors and reshape for broadcast compatibility
        dataset_mean = torch.tensor(dataset_mean, device=device).view(1, 1, 3, 1, 1)
        dataset_std = torch.tensor(dataset_std, device=device).view(1, 1, 3, 1, 1)

        # Perform denormalization
        # First reverse the dataset-specific normalization
        tensor = tensor * dataset_std + dataset_mean
    # Then reverse the default normalization
    return tensor * default_std + default_mean


def load_trained_pl_model(ckpt_path, new_hparams, for_inference = False):
    from base_model_trainer import ModelTrainer
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model = ModelTrainer(new_hparams)
    model.load_state_dict(checkpoint['state_dict'])
    if for_inference:
        model.cuda().eval()
        model.model.eval()
    return model.model


def create_image_encoder(backbone_type, backbone_size, pretrained=True, use_rope=False, use_ape=False, use_masking=False, patch_size=14):
    vit_backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "huge": "vith14",
        "giant": "vitg14",
    }

    if backbone_type == 'dinov2':
        if pretrained:
            backbone_name = vit_backbone_archs[backbone_size]
            backbone = torch.hub.load('facebookresearch/dinov2', model=f"dinov2_{backbone_name}")
        else:
            from model.cv.dinov2.vision_transformer import vit_xsmall, vit_small, vit_base, vit_large, vit_huge, vit_giant2
            vit_constructors = {
                "xsmall": vit_xsmall,
                "small": vit_small,
                "base": vit_base,
                "large": vit_large,
                "huge": vit_huge,
                "giant": vit_giant2,
            }
            backbone = vit_constructors[backbone_size](patch_size=patch_size, use_rope=use_rope, use_ape=use_ape)

        if not (use_masking):
            del backbone._parameters['mask_token'] # this is done as this param was unused and was causing pl ddp unused param issues
        backbone = backbone.to("cpu")  # ensure on cpu to prevent any device mismatch issues, lightning will handle moving to gpu
        return backbone

    elif backbone_type == "vae":
        from diffusers import AutoencoderKL

        if pretrained:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        else:
            raise NotImplementedError("Loading VAE without weights is currently not supported.")
        vae = vae.to("cpu")  # ensure on cpu to prevent any device mismatch issues, lightning will handle moving to gpu
        return vae

    elif backbone_type == "mae":
        from transformers import ViTMAEModel

        if pretrained:
            mae = ViTMAEModel.from_pretrained(f'facebook/vit-mae-{backbone_size}')
        else:
            from transformers import ViTMAEConfig
            config = ViTMAEConfig.from_pretrained(f'facebook/vit-mae-{backbone_size}')
            mae = ViTMAEModel(config)
        mae = mae.to("cpu")  # ensure on cpu to prevent any device mismatch issues, lightning will handle moving to gpu
        return mae

    else:
        raise NotImplementedError(f"Unspported backbone type: {backbone_type}")


def load_tdv_encoder_from_checkpoint(ckpt, backbone_type, backbone_size, key="frame_encoder"):
	checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)

	encoder_sd = {k.replace(f"model.{key}.", ""): v
				  for k, v in checkpoint["state_dict"].items()
				  if k.startswith(f"model.{key}.")}

	encoder = create_image_encoder(backbone_type, backbone_size, pretrained=False)
	encoder.load_state_dict(encoder_sd, strict=True)

	encoder.eval().to("cuda")
	for p in encoder.parameters():
		p.requires_grad_(False)

	return encoder


def encode_images(images, encoder_name, encoder, condition=None, masks=None, mask_ratio=None, return_masks=False, return_all_layers=False):
    if encoder_name == 'dinov2':
        if condition is not None:
            encoder_out = encoder(images, condition=condition, is_training=True, masks=masks, mask_ratio=mask_ratio, return_all_layers=return_all_layers)
        else:
            encoder_out = encoder(images, is_training=True, masks=masks, mask_ratio=mask_ratio, return_all_layers=return_all_layers)

        cls = encoder_out['x_norm_clstoken'].unsqueeze(1)  	            # (B, 1, D)
        patches = encoder_out['x_norm_patchtokens']		 	            # (B, num_patches, D)
        masks = encoder_out['masks']

        if return_masks:
            return torch.cat((cls, patches), dim=1), masks                  # (B, num_patches, D)
        elif return_all_layers:
            all_layers = encoder_out['all_layers']
            return torch.cat((cls, patches), dim=1), all_layers             # list of (B, num_patches, D) for each layer
        else:
            return torch.cat((cls, patches), dim=1)                     	# (B, num_patches, D)

    elif encoder_name == 'dinov1':
        return encoder(images)

    elif encoder_name == 'mae':
        batch, c, h, w = images.shape
        num_patches = (h//16) * (w//16)
        tokens = encoder(images, noise=torch.zeros(images.size(0),num_patches), return_dict=True).last_hidden_state
        return tokens                        										    # (B, num_patches_ D)

    raise ValueError(f"{encoder_name} not supported for now to get all tokens")


def create_motion_encoder(difference_encoder_type, depth=None, xattn_condition_dim=None, use_rope=False, use_ape=False, use_spatial_conditioning=False, spatial_condn_gating=False, ignore_prefix_tokens_in_condition=False, use_masking=False):
    '''
    Returns a difference encoder and the embedding dimension.

    ARGS:
    depth -> (only applicable for dinoxattn models) the number of transformer blocks to be used

    NOTES:
    For resnets -> the embedding dimension is the embedding before the final linear layer (in_features to the final projection layer)
    For vision transformers -> the embedding dimension is the embedding before the final projection head (in_features to the final projection head)

    For both, we replace the final classification layer with an identity layer to retain the embeddings
    '''
    # -- resnets
    if difference_encoder_type == "resnet18":
        difference_encoder = models.resnet18(pretrained=False)
        output_embedding_dim = difference_encoder.fc.in_features
    elif difference_encoder_type == "resnet50":
        difference_encoder = models.resnet50(pretrained=False)
        output_embedding_dim = difference_encoder.fc.in_features
    elif difference_encoder_type == "resnet101":
        difference_encoder = models.resnet101(pretrained=False)
        output_embedding_dim = difference_encoder.fc.in_features
    elif difference_encoder_type == "resnet152":
        difference_encoder = models.resnet152(pretrained=False)
        output_embedding_dim = difference_encoder.fc.in_features

    # -- vision transformers
    elif difference_encoder_type in ["vit_base16", "vit_b16"]:
        difference_encoder = models.vit_b_16(pretrained=False)
        output_embedding_dim = difference_encoder.heads.head.in_features
    elif difference_encoder_type in ["vit_large16", "vit_l16"]:
        difference_encoder = models.vit_l_16(pretrained=False)
        output_embedding_dim = difference_encoder.heads.head.in_features
    elif difference_encoder_type in ["vit_huge14", "vit_h14"]:
        difference_encoder = models.vit_h_14(pretrained=False)
        output_embedding_dim = difference_encoder.heads.head.in_features

    # -- DinoViT with cross attention
    elif difference_encoder_type.startswith("dinoViT_xattn"):
        assert xattn_condition_dim is not None

        from model.cv.dinov2_with_cross_attention.vision_transformer import vit_xattn_base, vit_xattn_giant2, vit_xattn_large, vit_xattn_small, vit_xattn_xsmall
        dino_vit_xattn_map = {
            "dinoViT_xattn_xsmall14": partial(vit_xattn_xsmall, patch_size=14),
            "dinoViT_xattn_small14": partial(vit_xattn_small, patch_size=14),
            "dinoViT_xattn_base14": partial(vit_xattn_base, patch_size=14),
            "dinoViT_xattn_base14-1024": partial(vit_xattn_base, patch_size=14, embed_dim=1024, num_heads=16),
            "dinoViT_xattn_base14-1280": partial(vit_xattn_base, patch_size=14, embed_dim=1280, num_heads=16),
            "dinoViT_xattn_base16": partial(vit_xattn_base, patch_size=16),
            "dinoViT_xattn_large14": partial(vit_xattn_large, patch_size=14),
            "dinoViT_xattn_giant14": partial(vit_xattn_giant2, patch_size=14),
        }
        difference_encoder = dino_vit_xattn_map[difference_encoder_type](
            xattn_condition_dim=xattn_condition_dim,
            num_register_tokens=0,
            depth=depth,
            use_spatial_conditioning=use_spatial_conditioning,
            use_gating_for_spatial_condn=spatial_condn_gating,
            ignore_prefix_tokens_in_condition=ignore_prefix_tokens_in_condition,
            use_rope=use_rope,
            use_ape=use_ape,
        )
        output_embedding_dim = difference_encoder.embed_dim
        if not use_masking:
            del difference_encoder._parameters['mask_token'] # this is done as this param was unused and was causing pl ddp unused param issues
    else:
        raise ValueError(f"Invalid difference encoder type: {difference_encoder_type}")

    # -- remove final linear layer
    if difference_encoder_type.startswith("resnet"):
        difference_encoder.fc = nn.Identity()
    elif difference_encoder_type.startswith("vit"):
        difference_encoder.heads.head = nn.Identity()

    return difference_encoder, output_embedding_dim


def has_layer_norm(model):
    return any(isinstance(module, nn.LayerNorm) for _, module in model.named_modules())

def init_wandb_watch(wandb_logger, model_trainer, wandb_watch_log_freq):
    if not has_layer_norm(model_trainer.model):
        wandb_logger.watch(model_trainer.model, log="all", log_freq = wandb_watch_log_freq)

    else: # all of complex below code is to get around the issue where wandb watch with layer norm has 'AttributeError: 'NoneType' object has no attribute 'data'' when logging gradients...
        non_layernorm_container = nn.Module()
        layernorm_container = nn.Module()

        non_ln_modules = {}
        ln_modules = {}

        for name, module in model_trainer.model.named_modules():
            if name == "": # skips top level model
                continue
            safe_name = name.replace(".", "_") # model cant contain '.' in name

            if isinstance(module, nn.LayerNorm):
                ln_modules[safe_name] = module
            else:
                # Only add modules that don't contain LayerNorm as submodules
                has_ln_child = any(isinstance(child, nn.LayerNorm)
                                for child in module.modules())
                if not has_ln_child:
                    non_ln_modules[safe_name] = module

        for name, module in non_ln_modules.items():
            non_layernorm_container.add_module(name, module)

        for name, module in ln_modules.items():
            layernorm_container.add_module(name, module)

        wandb_logger.watch(non_layernorm_container, log="all", log_freq=wandb_watch_log_freq)
        wandb_logger.watch(layernorm_container, log="parameters", log_freq=wandb_watch_log_freq)

def calculate_var_covar(previous_frame_encodings):
    with torch.no_grad():
        flattened = previous_frame_encodings.reshape(-1, previous_frame_encodings.size(-1))
        flattened_centered = flattened - flattened.mean(dim=0, keepdim=True)

        variance = flattened.var(dim=0, unbiased=False).mean()

        cov_matrix = (flattened_centered.T @ flattened_centered) / (flattened_centered.shape[0] - 1)

        def off_diagonal(mat):
            return mat.flatten()[1:].view(mat.size(0) - 1, mat.size(1) + 1)[:, :-1]

        off_diag = off_diagonal(cov_matrix)
        off_diag_covariance = off_diag.abs().mean()

    return variance, off_diag_covariance
