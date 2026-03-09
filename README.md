<h1 align="center">
<span>ViCA: Efficient Multimodal LLMs</span><br>
<span>with Vision-Only Cross-Attention</span>
</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.07574-b31b1b)](https://arxiv.org/abs/2602.07574)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-ViCA-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/HarrisonWu/ViCA)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![Last Commit](https://img.shields.io/github/last-commit/EIT-NLP/ViCA)](https://github.com/EIT-NLP/ViCA)


</div>

> <strong> ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention </strong>
>
> <a href="https://github.com/FakeWoke" rel="nofollow">Wenjie Liu</a><sup>\*,1</sup>, 
<a href="https://harrisonwu42.github.io/" rel="nofollow">Hao Wu</a><sup>\*,1</sup>, 
Xin Qiu<sup>1</sup>, 
<a href="https://scholar.google.com/citations?user=FwXKs_YAAAAJ" rel="nofollow">Yingqi Fan</a><sup>1</sup>, 
Yihan Zhang<sup>1</sup>, 
<a href="https://anhaozhao-llmer.github.io/" rel="nofollow">Anhao Zhao</a><sup>1</sup>, 
<a href="https://yunpuma.github.io/" rel="nofollow">Yunpu Ma</a><sup>2</sup>, 
<a href="https://chin-gyou.github.io/" rel="nofollow">Xiaoyu Shen</a><sup>†,1</sup> 
>
> <sup>1</sup>Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Institute of Digital Twin, Eastern Institute of Technology, Ningbo
>
> <sup>2</sup>LMU Munich
>
> <sup>\*</sup> Equal Contribution, <sup>†</sup> Corresponding Author (xyshen@eitech.edu.cn)


<p align="center">
  <img src="assets/vica.png" width="400">
</p>

If you find this work useful for your research and applications, please consider citing:

```bibtex
@misc{liu2026vicaefficientmultimodalllms,
    title={ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention}, 
    author={Wenjie Liu and Hao Wu and Xin Qiu and Yingqi Fan and Yihan Zhang and Anhao Zhao and Yunpu Ma and Xiaoyu Shen},
    year={2026},
    eprint={2602.07574},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2602.07574}, 
}
```

<!-- 🔥 📚 👀 🌟 ✨ ✒️ 🎯 📄 🙏 ✉️ 🤗 🌐 🚀 🔔 💡 🔧 ⭐️ -->


## 🔥News <a id="news"></a>

- **[TODO]** Code, checkpoints, and documentation are being prepared and will be released soon.
- **[2026.02.07]** The preprint is now published! 

## 💡 Highlights <a id="highlights"></a>
- Insights on MLLM Redundancy: Demonstrates that projected visual embeddings are already well-aligned with language space, and effective vision-language interaction occurs in only a small subset of Transformer layers, revealing substantial redundancy in dense visual processing.
- ViCA Architecture: Introduces Vision-only Cross-Attention (ViCA), a minimal MLLM design where visual tokens bypass all self-attention and feed-forward layers, interacting with text solely via sparse cross-attention at key layers for efficient multimodal fusion.
- Performance-Efficiency Trade-off: Maintains approximately 98% of baseline accuracy across three MLLM backbone models and nine multimodal benchmarks, while reducing visual-side computation to about 4% of the original, significantly outperforming 26 existing pruning methods in performance-efficiency trade-offs.
- Hardware-Friendly Acceleration: Achieves >3.5× speedup in single-batch inference and >10× speedup in multi-batch inference, compatible with FlashAttention.
- Orthogonal to Token Pruning: Compatible with token pruning methods for further gains, e.g., combining with PDrop in training-free inference reduces visual computation to 2% with over 96% performance retention.



## 📚 Contents <a id="contents"></a>

- [News](#news): Latest updates, news, and announcements.
- [Highlights](#highlights): Core insights and key features highlighted in this work.
- [Preparation](#preparation): Environment setup and required dependencies.
- [Usage](#usage): Instructions on how to run and use the code.
- [License](#license): License information for this repository.
- [Acknowledgments](#acknowledgments): Credits to projects and contributors that inspired or supported this work.
- [Contact](#contact): Contact information for questions, feedback, or collaboration.
- [Related Projects](#projects): Research projects from our group ([EIT-NLP](https://idt.eitech.edu.cn/nlp/)) related to MLLM compression.


## ✒️ Preparation <a id="preparation"></a>

### Installation

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



**2. Copy updated files to llava library**

* `modeling_llama_mask.py` and `llava_llama_mask.py` implement theoretical pruning in `eager-attention`:
  * Mask the corresponding attention weight in the attention block of each transformer layer.
  * In FFN, extract text tokens from hidden states, feed them through FFN, and concatenate with visual tokens.
```bash
cp ../modeling_llama_mask.py ./llava/models/modeling_llama_prune.py
cp ../llava_llama_mask.py ./llava/model/language_model/llava_llama.py
```

* `modeling_llama_accel.py` and `llava_llama_accel.py` implement practical acceleration in `eager-attention` and `flash-attention`:
  * Using only text tokens as the hidden state, while the visual tokens remain frozen. In a few layers, the visual tokens are used as KV-pairs in the attention block..

```bash
cp ../modeling_llama_accel.py ./llava/models/modeling_llama_prune.py
cp ../llava_llama_accel.py ./llava/model/language_model/llava_llama.py
```



## 🎯 Usage <a id="usage"></a>

### Inference
1. Download the checkpoints from our [Model Zoo](docs/MODEL_ZOO.md).
2. We evaluate our model on the following **9 widely-used multimodal benchmarks** to provide a comprehensive assessment across perception, reasoning, hallucination, and specialized capabilities:
- MME
- GQA
- MMBench
- MMBench_CN
- POPE
- SEED-I (image subset of SEED-Bench)
- SQA-I (image subset of ScienceQA)
- TextVQA
- VQAv2
```bash
T2V_LAYERS="[0,1,7,8,9,10,11,14]" bash scripts/v1_5/eval/mme.sh
……
T2V_LAYERS="[0,1,7,8,9,10,11,14]" bash scripts/v1_5/eval/textvqa.sh
```


### Train

####  Training Data
For our experiments, we primarily use the **LLaVA-1.5** training dataset, which can be prepared following the [official guidelines](https://github.com/haotian-liu/LLaVA#train). 

#### Models Used

We provide support for three model scales for LLaVA-1.5:
- [MobileLLaMA-2.7B](https://huggingface.co/mtgv/MobileLLaMA-2.7B-Base)
- [Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [Vicuna-13B-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)
- [CLIP-ViT-Large-Patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)



#### Training Recipe

Our training approach consists of two stages: **pretraining** and **fine-tuning**. The training process is configured via the following shell script:
```bash
T2V_LAYERS="[0,1,7,8,9,10,11,14]" bash scripts/v1_5/pretrain.sh
T2V_LAYERS="[0,1,7,8,9,10,11,14]" bash scripts/v1_5/finetune.sh
```

- `T2V_LAYERS`: Controls which transformer layers in the LLM apply text-vision cross-attention. 
  Only the specified layers perform cross-attention between text and visual tokens; 
  all remaining layers process only text tokens.

We train model variants in the implementation versions of the theoretical pruning of ` modeling.lama_mask. py ` and ` llava_1lama_mask. py `

Preserved text-to-vision cross-attention layers in LLaVA-1.5 models:
- LLaVA-1.5-3B: \{0, 1, 14, 15, 18, 19, 21, 22, 23\}
- LLaVA-1.5-7B: \{0, 1, 7, 8, 9, 10, 11, 14\}
- LLaVA-1.5-13B: \{0, 6, 8, 9, 10, 13, 14, 16\}


## 📄 License <a id="license"></a>

This project is released under the [Apache 2.0 license](https://opensource.org/licenses/Apache-2.0).


## 🙏 Acknowledgments <a id="acknowledgments"></a>

- Thanks for the [LLaVA](https://github.com/haotian-liu/LLaVA), [FastV](https://github.com/pkunlp-icler/FastV), and [PyramidDrop](https://github.com/Cooperx521/PyramidDrop) library, which helps us to quickly implement our ideas.

## ✉️ Contact <a id="contact"></a>

For questions, suggestions, or collaboration opportunities, please feel free to reach out:

- **Wenjie Liu**: wenjay_leo@outlook.com
- **Hao Wu**: haowu.ai.research@gmail.com
- **Xiaoyu Shen**: xyshen@eitech.edu.cn

## 🌐 Related Projects (ours) <a id="projects"></a>
- Survey
  - [From Data to Model: A Survey of the Compression Lifecycle in MLLMs](https://github.com/EIT-NLP/Awesome-MLLM-Compression)
- Vision Encoder
  - [UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](https://github.com/EIT-NLP/UTPTrack)
- MLLM
  - [VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs](https://github.com/EIT-NLP/VisiPruner)
  - [HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](https://github.com/EIT-NLP/HiDrop)
