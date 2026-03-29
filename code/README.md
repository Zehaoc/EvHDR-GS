# EvHDR-GS training code

## CRF / HDR-to-LDR and optimization stability

The current **camera response function (CRF)** and HDR-to-LDR path is implemented mainly in `scene/crf_learner.py` (`CRFLearner`). In practice, this formulation can make **joint optimization unstable**.

For a more stable design, we recommend studying **GaussHDR** (CVPR 2025), which proposes **two tonemapper variants** and discusses how they behave in HDR Gaussian splatting:

- Liu et al., *GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D*, CVPR 2025.  
  PDF: [https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_GaussHDR_High_Dynamic_Range_Gaussian_Splatting_via_Learning_Unified_3D_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_GaussHDR_High_Dynamic_Range_Gaussian_Splatting_via_Learning_Unified_3D_CVPR_2025_paper.pdf)

**Note:** This repository will be **updated** to align the tonemapping / HDR pipeline with that direction; the present `code/` snapshot is kept for reference until the new version is released.

