# ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention

This repository contains the code for the paper **"ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention"**. 

## Install
1.  Set up LLavA  https://github.com/haotian-liu/LLaVA 
```Shell
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation   
pip install transformers==4.36.2
```   


2. Copy our updated `modeling_llama_prune.py` and `llava_llama.py` to llava library

TODO

```Shell

```


## Inference
1. Download the checkpoints of pruned LLaVA

   [LLaVA-1.5-3B] (https://huggingface.co/Mickey25/lwj-3b)
   
   [LLaVA-1.5-7B] (https://huggingface.co/Mickey25/liuwj-7b)
   
   [LLaVA-1.5-13B] (https://huggingface.co/Mickey25/liuwj-13b-new)



## Contact

If you have any questions about this project, or would like to discuss related topics, feel free to reach out via email:

- **Wenjie Liu**: [wenjay_leo@outlook.com](mailto:wenjay_leo@outlook.com)  


