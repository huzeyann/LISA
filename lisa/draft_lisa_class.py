# %%

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torch import nn, Tensor
from typing import Tuple
from einops import rearrange


class LISA(nn.Module):

    def __init__(self, prompt_str='where is the object?'):
        super().__init__()
        try :
            import lisa
        except ImportError:
            raise ImportError("Please install lisa from \n `pip install git+https://github.com/huzeyann/LISA.git`")
            
        import bleach
        from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

        from lisa.model.LISA import LISAForCausalLM
        from lisa.model.llava import conversation as conversation_lib
        from lisa.model.llava.mm_utils import tokenizer_image_token
        from lisa.model.segment_anything.utils.transforms import ResizeLongestSide
        from lisa.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

        version = "xinlai/LISA-7B-v1"
        model_max_length = 512
        tokenizer = AutoTokenizer.from_pretrained(
            version,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

        vision_tower = "openai/clip-vit-large-patch14"
        model = LISAForCausalLM.from_pretrained(
            version, low_cpu_mem_usage=True, vision_tower=vision_tower, seg_token_idx=seg_token_idx,
            torch_dtype=torch.bfloat16,
        )

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16)

        model = model.to(dtype=torch.bfloat16)

        def expand_hw(tensor):
            hw = np.sqrt(tensor.shape[-2]).astype(int)
            return rearrange(tensor, "b (h w) c -> b h w c", h=hw)

        def new_forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
        ) -> Tuple[Tensor, Tensor]:
            self.input_keys = expand_hw(keys.clone())
            # print("forward", queries.shape, keys.shape, query_pe.shape, key_pe.shape)
            # Self attention block
            if self.skip_first_layer_pe:
                queries = self.self_attn(q=queries, k=queries, v=queries)
            else:
                q = queries + query_pe
                attn_out = self.self_attn(q=q, k=q, v=queries)
                queries = queries + attn_out
            queries = self.norm1(queries)

            # Cross attention block, tokens attending to image embedding
            q = queries + query_pe
            k = keys + key_pe
            attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
            queries = queries + attn_out
            queries = self.norm2(queries)

            # MLP block
            mlp_out = self.mlp(queries)
            queries = queries + mlp_out
            queries = self.norm3(queries)

            # Cross attention block, image embedding attending to tokens
            q = queries + query_pe
            k = keys + key_pe
            attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
            
            self.attn_output = expand_hw(attn_out.clone())
            
            keys = keys + attn_out
            keys = self.norm4(keys)
            
            self.block_output = expand_hw(keys.clone())
            # print("forward, block_output", queries.shape, keys.shape)


            return queries, keys

        setattr(model.model.visual_model.mask_decoder.transformer.layers[0].__class__, "forward", new_forward)

        import math
        def new_final_forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            # Input projections
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

            # Separate into heads
            q = self._separate_heads(q, self.num_heads)
            k = self._separate_heads(k, self.num_heads)
            v = self._separate_heads(v, self.num_heads)

            # Attention
            _, _, _, c_per_head = q.shape
            attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
            attn = attn / math.sqrt(c_per_head)
            attn = torch.softmax(attn, dim=-1)

            # Get output
            out = attn @ v
            out = self._recombine_heads(out)
            out = self.out_proj(out)
            
            self.attn_output = out.clone()
            # print("final_forward", q.shape, k.shape, v.shape, out.shape)

            return out

        setattr(model.model.visual_model.mask_decoder.transformer.final_attn_token_to_image.__class__, "forward", new_final_forward)
        
        self.model = model
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.prompt_str = prompt_str
        
    def forward(self, images, input_str=None):
        
        from transformers import CLIPImageProcessor

        from lisa.model.llava import conversation as conversation_lib
        from lisa.model.llava.mm_utils import tokenizer_image_token
        from lisa.model.segment_anything.utils.transforms import ResizeLongestSide
        from lisa.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

        input_str = input_str if input_str is not None else self.prompt_str

        # Model Inference
        conv = conversation_lib.conv_templates['llava_v1'].copy()
        conv.messages = []

        prompt = input_str
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        use_mm_start_end = True
        if use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()


        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # resize to 224
        image_clip = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        image_clip = image_clip.bfloat16()

        image = images.bfloat16()
        
        resize_list = [(1024, 1024)]
        original_size_list = [(1024, 1024)]
        
        output_ids, pred_masks = self.model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=self.tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        text_output = text_output.split("ASSISTANT: ")[-1]
        
        out_dict = {}
        for i_layer, layer in enumerate(self.model.model.visual_model.mask_decoder.transformer.layers):
            out_dict[f"dec_{i_layer}_input"] = [layer.input_keys]
            out_dict[f"dec_{i_layer}_attn"] = [layer.attn_output]
            out_dict[f"dec_{i_layer}_block"] = [layer.block_output]
        return out_dict
# %%
model = LISA().cuda()
input_image = torch.randn(1, 3, 1024, 1024).cuda()
out_dict = model(input_image)
# %%
for k, v in out_dict.items():
    print(k, v.shape, v.dtype)
# %%
