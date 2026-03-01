#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from ..modeling_llama_prune import LlamaForCausalLM  # 修改

from ..modeling_llama_prune import LlamaModel_prune
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from llava.constants import  IMAGE_TOKEN_INDEX
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel_prune):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 标记 generate已经设置vision_exists
        self.vision_exists_set_by_generate = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if not self.vision_exists_set_by_generate:
            # prefill和 decoder 进入forward函数的时候，images is None，不会进入这里的if，在 generate函数设置了视觉存在标志
            # 训练的时候 images is not None，进入设置 视觉存在标志；下一个 数据case 可能没有image，
            # 特殊情况: 数据case本身就没有image
            vision_token_positions = (input_ids == IMAGE_TOKEN_INDEX).float().argmax(dim=1).long()
            # 验证batch-size的所有输入的vis position是否一致
            if torch.all(vision_token_positions == vision_token_positions[0]):
                # 如果一致，赋值为int值
                self.model.vision_token_pos = vision_token_positions[0].item()
            else:
                # 如果不一致，记录下来并选择最多相同的pos
                unique_positions, counts = torch.unique(vision_token_positions, return_counts=True)
                most_common_idx = torch.argmax(counts)
                most_common_pos = unique_positions[most_common_idx].item()
                print(f"Vision token positions are inconsistent: {vision_token_positions.tolist()}")
                self.model.vision_token_pos = most_common_pos
            # 是否有图片的标志 保存到 self.model.vision_exists
            if images is None:
                self.model.vision_exists = False
            else:
                self.model.vision_exists = bool(torch.any(images != 0).item()) # finetune数据中 纯文本输入，img为全0张量
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        # 设置 prefill阶段视觉存在标志
        vision_token_positions = (inputs == IMAGE_TOKEN_INDEX).float().argmax(dim=1).long()
        # 验证batch-size的所有输入的vis position是否一致
        if torch.all(vision_token_positions == vision_token_positions[0]):
            # 如果一致，赋值为int值
            self.model.vision_token_pos = vision_token_positions[0].item()
        else:
            # 如果不一致，记录下来并选择最多相同的pos
            unique_positions, counts = torch.unique(vision_token_positions, return_counts=True)
            most_common_idx = torch.argmax(counts)
            most_common_pos = unique_positions[most_common_idx].item()
            print(f"Vision token positions are inconsistent: {vision_token_positions.tolist()}")
            self.model.vision_token_pos = most_common_pos
        # 是否有图片的标志 保存到 self.model.vision_exists
        if images is None:
            self.model.vision_exists = False
        else:
            self.model.vision_exists = bool(torch.any(images != 0).item()) # finetune数据中 纯文本输入，img为全0张量
        self.vision_exists_set_by_generate = True # 这个标志 进入 forward函数，就不用设置vision_exists
        
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # 新增
        self.model.input_token_length =inputs_embeds.shape[1]
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
