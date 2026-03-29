# EvHDR-GS: Event-guided HDR Video Reconstruction with 3D Gaussian Splatting

Official PyTorch implementation and project website for **EvHDR-GS** (AAAI 2025).

**Links**

- [Project page](https://zehaoc.github.io/EvHDR-GS/)
- [Paper (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/32237)
- [This repository](https://github.com/Zehaoc/EvHDR-GS)

## Abstract

High Dynamic Range (HDR) video reconstruction seeks to accurately restore the extensive dynamic range present in real-world scenes. Existing methods often operate on only a few consecutive frames, which can yield inconsistent brightness across the video, and supervised approaches may suffer from data bias under domain shift. EvHDR-GS builds **3D Gaussian Splatting (3DGS)** with event guidance for temporally consistent HDR reconstruction, using **HDR 3D Gaussians** and a learnable HDR-to-LDR mapping driven by event streams and LDR frames.

## Repository layout

| Path | Description |
|------|-------------|
| `index.html`, `static/` | GitHub Pages project site |
| `code/` | Training and rendering code (`train.py`, `scene/`, `utils/`, …); see `code/README.md` |
| `image/` | Figures used on the project page |

## Contact

If you have any questions, please contact us by email.

## Citation

If you find EvHDR-GS useful, please cite:

```bibtex
@inproceedings{chen2025evhdr,
  title={EvHDR-GS: Event-guided HDR Video Reconstruction with 3D Gaussian Splatting},
  author={Chen, Zehao and Lu, Zhan and Ma, De and Tang, Huajin and Jiang, Xudong and Zheng, Qian and Pan, Gang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2367--2375},
  year={2025}
}
```

## License

The project website is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).

[![CC BY-SA 4.0](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
