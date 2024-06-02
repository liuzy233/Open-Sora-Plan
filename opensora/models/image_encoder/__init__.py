import torch
from torch import nn
from transformers import T5EncoderModel, CLIPModel, CLIPProcessor, CLIPVisionModelWithProjection, AutoProcessor, CLIPImageProcessor
from opensora.models.image_encoder.condition import FrozenOpenCLIPImageEmbedderV2
from opensora.models.image_encoder.resampler import Resampler

from opensora.utils.utils import get_precision

class CLIPImageWrapper(nn.Module):
    def __init__(self, args):
        super(CLIPImageWrapper, self).__init__()
        self.model_name = args.image_encoder_name
        dtype = get_precision(args)
        model_kwargs = {'cache_dir': args.cache_dir, 'low_cpu_mem_usage': True, 'torch_dtype': dtype}
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(self.model_name, **model_kwargs).eval()
        self.image_processor = AutoProcessor.from_pretrained(self.model_name, **model_kwargs)
        self.image_proj_model = Resampler(
            dim=1152,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=self.image_enc.config.hidden_size,
            output_dim=1152,
            ff_mult=4,
            video_length=1,
        )

        self.video_proj_model = Resampler(
            dim=1152,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=self.image_enc.config.hidden_size,
            output_dim=1152,
            ff_mult=4,
            video_length=17,
        )

    def forward(self, image, video_input=False): 
        inputs = self.image_processor(images=image, return_tensors="pt").pixel_values.to(self.image_enc.device, dtype=self.image_enc.dtype)
        image_encoder_embs = self.image_enc(inputs, output_hidden_states=True).hidden_states[-1] # [1,257,1280]
        if video_input:
            image_prompt_embeds = self.video_proj_model(image_encoder_embs) # [1,256,1152]
        else:
            image_prompt_embeds = self.image_proj_model(image_encoder_embs) # [4,16,1152]
        return image_prompt_embeds.detach()

# class CLIPImageWrapper(nn.Module):
#     def __init__(self, args):
#         super(CLIPImageWrapper, self).__init__()
#         self.model_name = args.text_encoder_name
#         dtype = get_precision(args)
#         model_kwargs = {'cache_dir': args.cache_dir, 'low_cpu_mem_usage': True, 'torch_dtype': dtype}
#         self.embedder = FrozenOpenCLIPImageEmbedderV2.from_pretrained(self.model_name, **model_kwargs).eval()
#         self.image_proj_model = Resampler(
#             dim=1152,
#             depth=4,
#             dim_head=64,
#             heads=12,
#             num_queries=16,
#             embedding_dim=self.image_enc.config.hidden_size,
#             output_dim=1152,
#             ff_mult=4,
#             video_length=16,
#         )

#     def forward(self, image, attention_mask): 
#         inputs = self.image_processor(images=image, return_tensors="pt")
#         image_encoder_embs = self.image_enc.get_text_features(input_ids=inputs, attention_mask=attention_mask).hidden_states[-1]
#         image_prompt_embeds = self.image_proj_model(image_encoder_embs) # [1,257,1280]
#         return image_prompt_embeds.detach()



image_encoder = {
    'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': CLIPImageWrapper
}


def get_image_enc(args):
    """deprecation"""
    image_enc = image_encoder.get(args.image_encoder_name, None)
    assert image_enc is not None
    return image_enc(args)

def get_image_warpper(image_encoder_name):
    """deprecation"""
    image_enc = image_encoder.get(image_encoder_name, None)
    assert image_enc is not None
    return image_enc
