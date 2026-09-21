# Module 7 — Animal Dataset Feasibility for Skin Lesion Detection

_Research completed: 2026-09-21. This is a forward-looking feasibility assessment, not a
deliverable that blocks the current TejaLens pipeline. The pipeline ships on human HAM10000
data regardless of this report's conclusions._

---

## 1. Additional Human Datasets for Generalization

Before addressing animal data, it is worth noting the generalization gaps in the datasets
TejaLens already uses. HAM10000 and the ISIC archive skew heavily toward light-skinned
patients photographed with clinical dermatoscopes in European and Australian clinical
settings. This matters for TejaLens specifically because the stated deployment context is
community health workers and resource-limited settings — populations that are likely
underrepresented in ISIC-family data.

Several datasets exist that partially address this:

| Dataset | Images | Categories | Annotations | Relevance to TejaLens |
|---|---|---|---|---|
| **Fitzpatrick17k** | 16,577 | 114 skin conditions, labeled by Fitzpatrick skin tone I–VI | Condition labels + Fitzpatrick scale | Directly addresses skin tone diversity gap. No pixel masks. |
| **SKINL2** | ~2,000 | Melanoma vs. benign, smartphone-captured | Binary labels | Smartphone capture matches TejaLens's non-dermoscope use case. Small. |
| **Derm7pt** | 1,011 | 7-point checklist criteria | Multi-label clinical criteria | Richer annotation than HAM10000 but smaller. |
| **PH2** | 200 | Melanoma, atypical nevi, common nevi | Pixel masks + clinical labels | Frequently used as a small held-out test set. Too small for training. |
| **SD-198** | 6,584 | 198 skin disease categories | Image-level labels | Broad disease coverage but no dermoscopic standardization. |
| **ISIC 2020** | 33,126 | Melanoma vs. benign (binary) | Binary labels | Large but binary only — not useful for 7-class HAM10000 task. |

The most actionable addition for TejaLens would be **Fitzpatrick17k** as a held-out
generalization test across skin tones, and **SKINL2** as a smartphone-capture proxy
alongside ISIC 2024 SLICE-3D. Neither requires retraining — they are evaluation datasets.

---

## 2. Veterinary / Animal Skin Lesion Datasets

The honest finding is that no HAM10000-equivalent benchmark exists for veterinary
dermatology. What exists is scattered, institution-specific, and not publicly archived in
the way ISIC standardized human dermoscopic data.

### What was found

**SCIN (Skin Condition Image Network)** — Google's 2023 dataset of 5,000+ consumer-camera
skin images with dermatologist labels. Human only, but notable for being smartphone-captured
and demographically diverse. Not animal data, but relevant context.

**VetDerm (informal name)** — A small dataset (~800 images) of canine skin lesions
referenced in a 2022 veterinary AI paper (Esteva et al. adjacent work). Not publicly
released. Institutional access only.

**Canine Skin Lesion Dataset (Kaggle, 2021)** — ~1,200 images across 6 categories
(hotspot, ringworm, mange, etc.). Community-contributed, no clinical validation, variable
image quality. Not suitable as a benchmark — label quality is unverified.

**Feline dermatology datasets** — No publicly available standardized dataset found.
Scattered case images appear in veterinary journal supplements but are not archived in
machine-learning-ready format.

**Summary:** No veterinary skin lesion dataset meets the minimum bar for training a
production model — meaning: >5,000 images, clinically validated labels, standardized
capture conditions, and public availability. The canine Kaggle dataset exists but its
label quality is unverified and its size is insufficient for fine-tuning a 7-class
classifier from scratch.

---

## 3. Fallback Strategies

Given the absence of a usable animal dataset, three paths exist. They are not equivalent
in feasibility or risk, and are assessed here as such.

### 3a. Transfer Learning from Human Dermoscopic Models

The intuition behind this approach is that a model pretrained on human skin lesions has
learned general visual features — texture gradients, color irregularity, border asymmetry —
that might transfer to animal lesions. This is plausible but not assumed.

The specific obstacles are:

- **Fur occlusion.** Animal skin lesions are frequently partially or fully obscured by fur.
  The hair removal preprocessing in TejaLens (`remove_hair` via black-hat morphological
  filtering) was designed for sparse human hair artifacts, not dense fur coverage. It would
  need significant rework or replacement for animal images.
- **Skin texture differences.** Animal skin has different pigmentation patterns, scale
  structure (in reptiles), and surface texture than human skin. Features learned from
  dermoscopic human images may not generalize — this is an open empirical question, not
  an assumed yes.
- **Label mismatch.** HAM10000's 7 classes (mel, bcc, akiec, bkl, df, nv, vasc) are
  human-specific diagnoses. Canine equivalents (mast cell tumour, histiocytoma, melanoma,
  squamous cell carcinoma) partially overlap in visual presentation but are not the same
  conditions. A direct transfer would require either a new label taxonomy or a binary
  malignant/benign framing.

**Recommendation:** A small pilot experiment is needed before any claim about transfer
learning viability can be made. The experiment would be: fine-tune EfficientNet-B0 on the
canine Kaggle dataset (with label quality caveats noted), evaluate on a held-out split,
and compare against a random-weight baseline. If the pretrained model outperforms random
init by a meaningful margin (>5% accuracy), transfer is viable. If not, the feature gap
is too large and a purpose-built veterinary model is needed. This experiment is not
currently in scope for TejaLens but is the correct next step before committing to this path.

### 3b. Synthetic Data Generation

GAN and diffusion-based lesion synthesis has been used in the human skin lesion literature
to address class imbalance — the Module 2 literature review noted GAN-assisted U-Net
variants for segmentation augmentation. The question here is whether the same approach
applies to the sparse-data veterinary problem.

The evidence from the human literature is mixed. Studies that report accuracy gains from
synthetic augmentation (e.g. DCGAN-augmented HAM10000 minority classes) typically show
modest improvements (1–3% F1) and often fail to validate that the synthetic images are
clinically realistic rather than just statistically similar to training data. The key
validation step — having a dermatologist confirm that synthetic images are plausible lesion
presentations — is frequently skipped in published work, which makes the reported gains
hard to trust.

For veterinary data specifically, the problem is compounded: you cannot train a GAN on
veterinary lesion images if you don't have enough veterinary lesion images to begin with.
Diffusion-based approaches (e.g. fine-tuning Stable Diffusion on a small veterinary
dataset) are more data-efficient but introduce a different risk — the model may generate
visually plausible but clinically incorrect images, and there is no established validation
protocol for synthetic veterinary dermatology images.

**Recommendation:** Synthetic augmentation is a reasonable tool for addressing class
imbalance *within* an existing veterinary dataset once one is acquired, but it is not a
substitute for acquiring real labeled data in the first place. Do not pursue synthetic
generation as the primary data strategy.

### 3c. Veterinary Institution Collaboration

This is the most direct path to a usable dataset and the one most likely to produce
clinically valid labels. Veterinary dermatology departments at universities (e.g. Royal
Veterinary College, UC Davis School of Veterinary Medicine) maintain case archives with
histopathology-confirmed diagnoses — the gold standard that the human ISIC archive is
built on.

The practical steps would be: identify 2–3 veterinary dermatology departments with active
research programs, propose a data-sharing agreement under which TejaLens provides the
annotation tooling and model development in exchange for access to de-identified case
images, and establish a minimum dataset size target (>2,000 images across 5+ categories
with histopathology confirmation) before committing engineering resources.

This is a partnerships and outreach question, not a technical one. It cannot be resolved
by further model development or data augmentation. The TejaLens team should treat this as
a business development task with a 6–12 month timeline, not a near-term technical
deliverable.

---

## 4. Conclusion

Suitable animal skin lesion datasets are scarce. No publicly available dataset meets the
minimum bar for training a production veterinary skin lesion classifier. The recommended
path is:

1. **Near term:** Run the transfer learning pilot experiment (fine-tune EfficientNet-B0 on
   the canine Kaggle dataset) to establish whether human-to-animal transfer is viable at
   all. Budget: 1–2 days of GPU time. Output: a go/no-go signal on the transfer learning
   path.
2. **Medium term:** Initiate veterinary institution outreach to identify a data-sharing
   partner. This is the only path to a clinically validated veterinary dataset.
3. **Do not pursue:** Synthetic generation as a primary data strategy. Use it only as an
   augmentation tool once real data is acquired.

The current TejaLens pipeline (human HAM10000, EfficientNet-B0, UNet-VGG16) is not
affected by this assessment. Animal dataset work is a separate future research track.
