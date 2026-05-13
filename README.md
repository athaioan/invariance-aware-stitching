
# [ICML 2026] Grounding Functional Similarity by Invariance-Aware Model Stitching

**Authors**: [Ioannis Athanasiadis](https://scholar.google.com/citations?user=RCAtJgUAAAAJ), [Anmar Karmush](https://scholar.google.com/citations?user=alaN6Z8AAAAJ&hl=sv), [Michael Felsberg](https://scholar.google.com/citations?user=lkWfR08AAAAJ)


## Abstract

In deep learning, functional similarity evaluation quantifies the extent to which independently trained models learn similar input–output relationships. In model stitching, functional similarity is framed as representation forward compatibility, i.e., whether the representations of two models can be aligned to solve a given task. Recent studies, however, highlight a critical limitation: models relying on different information cues can still produce compatible representations, making them appear misleadingly similar \cite{smithfunctional}. We attribute this failure to standard model stitching being inherently blind to the invariance properties of the stitched models. To address this limitation, we introduce the forward-backward compatibility requirement under which we formulate the invariance-aware model stitching. Through analyzing key stitching configurations, we study the interplay between forward and backward compatibility, showing that invariance-aware model stitching provides a more principled approach to functional similarity evaluation while revealing functional discrepancies previously obscured.


**TL;DR:** Invariance-aware model stitching make for a more meaningful notion of functional similarity compared to standard model stitching.

[📄 View Paper on OpenReview](https://openreview.net/forum?id=yUEFOa32CG)
