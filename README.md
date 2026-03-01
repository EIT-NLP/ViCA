# ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention

This repository contains the code for the paper : [ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention](https://arxiv.org/abs/2602.07574). 

## Install
1.  Set up LLavA  https://github.com/haotian-liu/LLaVA 
```Shell
cd LLaVA
conda create -n llava-vica python=3.10 -y
conda activate llava-vica
pip install --upgrade pip  
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation   
pip install transformers==4.36.2
```   


2. Copy our updated `modeling_llama.py` and `llava_llama.py` to llava library
当然，下面是更加简洁的版本：

---

### 2. Copy updated files to llava library

* `modeling_llama_mask.py` and `llava_llama_mask.py` implement theoretical pruning:
  * Mask the corresponding attention weight in the attention block of each transformer layer.
  * In FFN, extract text tokens from hidden states, feed them through FFN, and concatenate with visual tokens.
```bash
cp ../modeling_llama_mask.py ./llava/models/modeling_llama_prune.py
cp ../llava_llama_mask.py ./llava/model/language_model/llava_llama.py
```

* `modeling_llama_accel.py` and `llava_llama_accel.py` implement practical acceleration and are compatible with flash-attention:
  * Using only text tokens as the hidden state, while the visual tokens remain frozen. In a few layers, the visual tokens are used as KV-pairs in the attention block..

```bash
cp ../modeling_llama_accel.py ./llava/models/modeling_llama_prune.py
cp ../llava_llama_accel.py ./llava/model/language_model/llava_llama.py
```


## Inference
1. Download the checkpoints of pruned LLaVA

   [LLaVA-1.5-3B] (https://huggingface.co/Mickey25/lwj-3b)
   
   [LLaVA-1.5-7B] (https://huggingface.co/Mickey25/liuwj-7b)
   
   [LLaVA-1.5-13B] (https://huggingface.co/Mickey25/liuwj-13b-new)

2. efficiency evaluation
TODO
```Shell

```


## Train
### Training Data
For our experiments, we primarily use the **LLaVA-1.5** training dataset, which can be prepared following the [official guidelines](https://github.com/haotian-liu/LLaVA#train). 

### Models Used

### Training Recipe

Our training approach consists of two stages: **pretraining** and **fine-tuning**. The training process is configured via the following shell script:


- `T2V_LAYERS`: Controls which transformer layers in the LLM apply text-vision cross-attention. 
  Only the specified layers perform cross-modal interaction between text and visual tokens; 
  all remaining layers function as standard self-attention layers.



## Contact

If you have any questions about this project, or would like to discuss related topics, feel free to reach out via email:

- **Wenjie Liu**: [wenjay_leo@outlook.com](mailto:wenjay_leo@outlook.com)  


