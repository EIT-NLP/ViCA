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
<a href="https://scholar.google.com/citations?hl=zh-CN&user=Ix9RD18AAAAJ" rel="nofollow">Hao Wu</a><sup>\*,1</sup>, 
Xin Qiu<sup>1</sup>, 
<a href="https://scholar.google.com/citations?hl=zh-CN&user=FwXKs_YAAAAJ" rel="nofollow">Yingqi Fan</a><sup>1</sup>, 
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

If you find ViCA useful for your research and applications, please consider citing:

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
- Empirical Insights on MLLM Redundancy: Demonstrates that projected visual embeddings are already well-aligned with language space, and effective vision-language interaction occurs in only a small subset of Transformer layers, revealing substantial redundancy in dense visual processing.
- ViCA Architecture: Introduces Vision-only Cross-Attention (ViCA), a minimal MLLM design where visual tokens bypass all self-attention and feed-forward layers, interacting with text solely via sparse cross-attention at key layers for efficient multimodal fusion.
- Superior Performance-Efficiency Trade-off: Maintains approximately 98% of baseline accuracy across three MLLM backbone models and nine multimodal benchmarks, while reducing visual-side computation to about 4% of the original, significantly outperforming 26 existing pruning methods in performance-efficiency trade-offs.
- Hardware-Friendly Acceleration: Achieves >3.5× speedup in single-batch inference and >10× speedup in multi-batch inference, compatible with FlashAttention.
- Orthogonal to Token Pruning: Compatible with token pruning methods for further gains, e.g., combining with PDrop in training-free inference reduces visual computation to 2% with over 96% performance retention.

## result

\begin{table*}[t]
\centering
\caption{
Performance comparison of different pruning approaches on LLaVA-1.5-7B across nine benchmarks.  
The table reports per-benchmark accuracy and the average performance relative to the original model. Methods are grouped by pruning strategy: no background color indicates training-free methods, red denotes training-based pruning, and blue highlights our approach. ViCA refers to the model retrained under the proposed minimal efficient architecture using the standard two-stage pretraining and fine-tuning pipeline. ViCA+PDrop$^\dagger$ further applies PyramidDrop at inference time. Op and Tok denote operation-level and token-level pruning methods, respectively. Vision-related computation and token counts are measured following the FLOPs formulation in Appendix~\ref{sec:Calculation-Equation}.
}
\label{tab:finally-result}
\resizebox{0.97\textwidth}{!}{
    \begin{tabular}{c|cc|cc|ccccccccc|c}
    \toprule
    \multirow{2}{*}{\textbf{Method}} &
    \multicolumn{2}{c|}{\textbf{Sparsity}} &
    \multicolumn{2}{c|}{\textbf{Vision-side}} &
    \multirow{2}{*}{\textbf{MME\textsuperscript{P}}} &
    \multirow{2}{*}{\textbf{MMB}} &
    \multirow{2}{*}{\textbf{MMB\textsuperscript{CN}}} &
    \multirow{2}{*}{\textbf{GQA}} &
    \multirow{2}{*}{\textbf{VQA\textsuperscript{v2}}} &
    \multirow{2}{*}{\textbf{SQA\textsuperscript{I}}} &
    \multirow{2}{*}{\textbf{VQA\textsuperscript{T}}} &
    \multirow{2}{*}{\textbf{POPE}} &
    \multirow{2}{*}{\textbf{SEED\textsuperscript{I}}} &
    \multirow{2}{*}{\textbf{Rel. Avg.}} \\
    & \textbf{Op.} & \textbf{Tok.} & \textbf{Token} & \textbf{TFLOPs (Rel.)} & & & & & & & & & \\
    \midrule
    LLaVA-1.5-7B & - & - & 576 & 7.65 (100.0\%) & 1506.5 & 64.7 & 58.1 & 61.9 & 78.5 & 69.5 & 58.2 & 86.8 & 66.2 & 100.0\% \\
    \midrule
    ToMe & & \cmark & 64 & 0.83 (10.9\%) & - & 43.7 & 38.9 & 48.6 & 57.1 & 50.0 & 45.3 & 52.5 & - & 70.9\% \\
    PDrop & & \cmark & 64 & 0.83 (10.9\%) & 1309.2 & 58.8 & 50.5 & 47.5 & 69.2 & 69.0 & 50.6 & 55.9 & - & 85.0\% \\
    HiRED & & \cmark & 64 & 0.83 (10.9\%) & - & 60.2 & 51.3 & 54.6 & 69.7 & 68.2 & 44.2 & 73.6 & - & 88.2\% \\
    HiPrune & & \cmark & 64 & 0.83 (10.9\%) & - & 59.5 & 53.4 & 53.6 & 69.2 & 68.9 & 54.9 & 73.0 & - & 90.9\% \\
    FlowCut & & \cmark & 64 & 0.83 (10.9\%) & - & 60.8 & 55.4 & 55.6 & 72.8 & 69.1 & 55.6 & 80.2 & - & 94.2\% \\
    HoloV & & \cmark & 64 & 0.83 (10.9\%) & - & 63.3 & 55.1 & 55.3 & 72.8 & 69.5 & 55.4 & 80.3 & - & 94.6\% \\
    VISA & & \cmark & 64 & 0.83 (10.9\%) & 1420.6 & 62.1 & 57.3 & 56.2 & 74.1 & 67.9 & 55.6 & 77.6 & - & 94.6\% \\
    D$^2$Pruner & & \cmark & 64 & 0.83 (10.9\%) & - & 61.9 & 55.6 & 57.9 & 74.6 & 70.0 & 56.1 & 82.4 & - & 96.0\% \\
    VScan & & \cmark & 64 & 0.83 (10.9\%) & - & 62.1 & 55.7 & 58.3 & 75.4 & 69.1 & 55.6 & 85.0 & - & 96.4\% \\
    FiCoCo-L & & \cmark & 58 & 0.75 (9.9\%) & - & 61.5 & 53.3 & 53.2 & 69.7 & 69.5 & 55.7 & 82.1 & - & 93.1\% \\
    FastV & & \cmark & 32 & 0.42 (5.4\%) & 884.6 & 37.8 & 33.2 & 41.5 & 43.4 & 42.6 & 42.5 & 32.5 & - & 58.6\% \\
    SparseVLM & & \cmark & 32 & 0.42 (5.4\%) & 1046.7 & 51.4 & 40.6 & 48.3 & 58.6 & 57.3 & 46.1 & 67.9 & - & 76.4\% \\
    VisPruner & & \cmark & 32 & 0.42 (5.4\%) & 1271.0 & 58.4 & 52.7 & 52.2 & 67.7 & 69.2 & 53.9 & 72.7 & 54.3 & 88.2\% \\
    DivPrune & & \cmark & 32 & 0.42 (5.4\%) & 1284.9 & 57.6 & 49.1 & 54.9 & 71.2 & 68.6 & 52.9 & 81.5 & 58.7 & 90.0\% \\
    DOP\textsubscript{V} & \cmark & \cmark & 32 & 0.42 (5.4\%) & 1306.5 & 59.4 & 53.7 & 54.8 & 71.0 & 69.1 & 54.5 & 79.6 & 56.7 & 91.2\% \\
    CDPruner & & \cmark & 32 & 0.42 (5.4\%) & 1373.0 & 59.6 & 49.6 & 57.0 & 73.6 & 69.5 & 53.2 & 87.9 & - & 93.2\% \\
    DOP\textsubscript{CD} & \cmark & \cmark & 32 & 0.42 (5.4\%) & 1397.5 & 60.1 & 52.2 & 58.1 & 74.7 & 69.3 & 54.2 & 87.9 & 82.2 & 94.7\% \\
    \midrule
    \rowcolor{red!8}
    PDrop & & \cmark & 270 & 3.54 (46.3\%) & 1490.1 & 63.9 & 56.7 & 61.7 & 78.7 & 70.1 & 57.7 & 86.9 & 65.8 & 99.4\% \\
    \rowcolor{red!8}
    Dynamic-LLaVA & & \cmark & 115 & 1.50 (19.6\%) & 1479.8 & 65.4 & - & 61.4 & 78.0 & 69.1 & 57.0 & 85.0 & 64.6 & 98.8\% \\
    \rowcolor{red!8}
    YOPO & \cmark &  & 70 & 0.92 (12.0\%) & 1423.5 & 64.6 & 57.0 & 60.7 & 77.4 & 68.0 & 55.2 & 86.6 & 64.6 & 97.7\% \\
    \rowcolor{red!8}
    TwigVLM & & \cmark & 64 & 0.83 (10.9\%) & 1404.0 & 60.4 & 53.8 & 61.2 & 75.6 & 70.0 & 55.8 & 82.7 & 56.9 & 94.7\% \\
    \rowcolor{red!8}
    VisionZip & & \cmark & 64 & 0.83 (10.9\%) & - & 61.5 & - & 57.0 & 74.2 & 68.8 & 56.0 & 80.9 & - & 95.0\% \\
    \rowcolor{red!8}
    DART & & \cmark & 64 & 0.83 (10.9\%) & - & 64.7 & 56.7 & 57.1 & 74.6 & 71.1 & 54.7 & 79.3 & - & 96.1\% \\
    \rowcolor{red!8}
    LLaVA-PruMerge & & \cmark & 32 & 0.42 (5.4\%) & 1350.3 & 60.9 & 50.0 & 57.2 & 72.0 & 68.5 & 56.0 & 76.3 & 50.7 & 90.4\% \\
    \rowcolor{red!8}
    TRIM & & \cmark & 29 & 0.38 (4.9\%) & 1415.4 & 63.3 & 46.6 & 58.4 & 71.5 & 67.9 & 49.1 & 84.8 & 61.8 & 92.3\% \\
    \rowcolor{red!8}
    TokenPacker & & \cmark & 16 & 0.21 (2.7\%) & 1378.8 & 62.7 & - & 58.9 & 74.4 & 68.1 & 52.5 & 83.7 & - & 94.7\% \\
    \rowcolor{red!8}
    Delta-LLaVA & & \cmark & 16 & 0.21 (2.7\%) & 1375.9 & 62.9 & - & 59.5 & 73.1 & 69.7 & 53.6 & 84.7 & - & 95.4\% \\
    \midrule
    \rowcolor{lightblue}
    ViCA (ours) & \cmark &  & 24 & 0.31 (4.1\%) & 1464.5 & 64.0 & 57.7 & 60.4 & 76.6 & 68.5 & 55.5 & 86.7 & 63.2 & 97.8\% \\
    \rowcolor{lightblue}
    ViCA+PDrop$^\dagger$ (ours) & \cmark & \cmark & 12 & 0.16 (2.0\%) & 1449.7 & 63.3 & 56.9 & 59.0 & 75.8 & 69.3 & 54.4 & 85.2 & 60.8 & 96.3\% \\
    \bottomrule
    \end{tabular}
}
\end{table*}



## 🔧 TODO <a id="todo"></a>
- [x] xxxxxxxxx
- [ ] xxxxxxxxx
- [ ] xxxxxxxxx
- [ ] xxxxxxxxx

## 📚 Contents <a id="contents"></a>

- [News](#news)
- [Highlights](#highlights)
- [TODO](#todo)
- [Preparation](#preparation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Related Projects](#projects)


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


## 

## 🎯 Usage <a id="usage"></a>

### Inference
1. Download the checkpoints from our [Model Zoo](docs/MODEL_ZOO.md).
2. efficiency evaluation.
```bash
T2V_LAYERS="[0,1,7,8,9,10,11,14]" bash scripts/v1_5/eval/mme.sh
```


### Train

####  Training Data
For our experiments, we primarily use the **LLaVA-1.5** training dataset, which can be prepared following the [official guidelines](https://github.com/haotian-liu/LLaVA#train). 

#### Models Used

#### Training Recipe

Our training approach consists of two stages: **pretraining** and **fine-tuning**. The training process is configured via the following shell script:
```bash
T2V_LAYERS="[0,1,7,8,9,10,11,14]" bash scripts/v1_5/pretrain.sh
T2V_LAYERS="[0,1,7,8,9,10,11,14]" bash scripts/v1_5/finetune.sh
```

- `T2V_LAYERS`: Controls which transformer layers in the LLM apply text-vision cross-attention. 
  Only the specified layers perform cross-attention between text and visual tokens; 
  all remaining layers process only text tokens.

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

## 🌐 Related Projects <a id="projects"></a>
- Survey
  - [From Data to Model: A Survey of the Compression Lifecycle in MLLMs](https://github.com/EIT-NLP/Awesome-MLLM-Compression)
- Vision Encoder
  - [UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](https://github.com/EIT-NLP/UTPTrack)
- MLLM
  - [VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs](https://github.com/EIT-NLP/VisiPruner)
  - [HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](https://github.com/EIT-NLP/HiDrop)
