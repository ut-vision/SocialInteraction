# Multi-speaker Attention Alignment for Multimodal Social Interaction

**Liangyang Ouyang**, Yifei Huang, Mingfang Zhang, Caixin Kang, Ryosuke Furuta, and Yoichi Sato

<p align="center">The University of Tokyo</p>

Our paper is accepted by **CVPR 2026**!

[[arXiv]](https://arxiv.org/abs/2511.17952) [[Paper]](paper/CVPR2026_MultiSpeakerAlignment.pdf) [[Supplementary]](paper/CVPR2026_MultiSpeakerAlignment_supp.pdf)

<p align="center">
  <img src="paper/fig_first.png" alt="Teaser" width="100%">
</p>

## TODO

- ✅ Full paper release
- Zero-shot implementation release
- LoRA fine-tuned implementation release

## Abstract

Understanding social interaction in video requires reasoning over a dynamic interplay of verbal and non-verbal cues: who is speaking, to whom, and with what gaze or gestures. While Multimodal Large Language Models (MLLMs) are natural candidates, simply adding visual inputs yields surprisingly inconsistent gains on social tasks. Our quantitative analysis of cross-modal attention inside state-of-the-art MLLMs reveals a core failure mode: in multi-speaker scenes, visual and textual tokens lack speaker-consistent alignment, exhibiting substantially weaker cross-modal attention than in object-centric images. To address this, we propose a multimodal multi-speaker attention alignment method that can be integrated into existing MLLMs. First, we introduce dynamic cross-modal head selection to identify attention heads most responsible for grounding. Then, an adaptive social-aware attention bias, computed from existing attention patterns and speaker locations, is injected into the attention mechanism. This bias reinforces alignment between a speaker's visual representation and their utterances without introducing trainable parameters or architectural changes. We integrate our method into three distinct MLLMs (LLaVA-NeXT-Video, Qwen2.5-VL, and InternVL3) and evaluate on three benchmarks (TVQA+, MMSI, OnlineMMSI). Across four social tasks, results demonstrate that our approach improves the ability of MLLMs and achieves state-of-the-art results. Attention visualizations confirm our method successfully focuses the model on speaker-relevant regions, enabling more robust multi-party social reasoning.

## Citation

If this work is helpful in your research, please cite:

```bibtex
@article{ouyang2025multi,
  title={Multi-speaker Attention Alignment for Multimodal Social Interaction},
  author={Ouyang, Liangyang and Huang, Yifei and Zhang, Mingfang and Kang, Caixin and Furuta, Ryosuke and Sato, Yoichi},
  journal={arXiv preprint arXiv:2511.17952},
  year={2025}
}
```
