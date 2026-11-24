# Roadmap

## v0.1 — paper release with a leak-free re-evaluation (this version)

- Paper PDF and BibTeX in `paper/`, project page in `docs/`.
- PyTorch implementation of the full method: preprocessing (Sec. 3.2), baseline CNN (3.3), per-block Grad-CAM relevance
  (3.4, Eq. 4), percentile pruning and retraining (3.5), SHAP and LIME validation (3.6).
- The paper's protocol (random 80/10/10 split by image) and a leak-free protocol (group-aware split from the duplicate
  audit), each over three seeds, with every number reported on the full and on the unseen test sets.
- Tests and CI.

## v0.2 — robustness of the refinement rule

- Scale-normalised relevance: raw Grad-CAM magnitudes grow with depth, so Eq. 4 always ranks the earliest blocks
  lowest. Compare against relevance normalised per layer (for example by the activation norm) and against random
  block removal of the same size.
- Sweep of the percentile p and of repeated refinement rounds.
- Quantitative explanation quality: deletion and insertion curves for Grad-CAM, SHAP and LIME, and agreement between
  the three methods, on the leak-free test set.

## v0.3 — external validation

- Evaluation on brain MRI collections that are disjoint from Msoud by construction (for example BraTS-derived 2D
  slices), with duplicate checks against the training set before any number is reported.
- Patient-level splits where patient identifiers exist (figshare / Cheng et al.).

## Sharing plan

1. Link the repository and the project page from the arXiv listing, the ACM DL entry and the personal site.
2. Add the leak-free table to Papers with Code under the BIC 2026 entry.
3. Cross-reference the Kaggle duplicate audit (dataset and notebook) and this repository from each other.
4. One announcement per release on LinkedIn and X, each with the leak-free table, not more often.
