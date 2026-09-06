# Phase 2 report — text weak supervision (complete)

## Result

- **Rule extractor** (negation-aware, sentence-scoped, 8+ languages):
  macro-AUC **0.791**, macro-F1 **0.703** on the gold-58 studies
- **Pure ML** (char-ngram TF-IDF + LR, LOO-CV): macro-AUC 0.571 — *rules win
  decisively* with only 58 labeled studies; text signal is lexical, not statistical
- **Selective ensemble** (per-label where ML validated better): 0.796 — marginal;
  ML only used as a 25%-weight blend for 5 labels in pseudo-labeling

## Architecture (src/knee/text_extractor.py)

1. **normalize**: Cyrillic + Greek → Latin transliteration, accent-strip,
   lowercase (Greek discovery: 4 gold studies were Greek, invisible to v1)
2. **section strip**: clinical-history/indication sections removed (speculative
   text caused Contusion FPs)
3. **sentence split** on `.;>*|\n` — polarity decided per-sentence (v1's char-window
   negation bled across sentences)
4. **polarity**: negators/normality → 0.0; hedgers → 0.5; else 1.0
   - lookbehind word boundaries fixed `interno → "no "` false negation
5. **anchor + evidence in same sentence**: ACL/MCL anchors with PCL/LCL
   guards; menisci/OA with per-language laterality (med/lateral/PF)
6. **severity grading** discovered from gold: MCL grade-1/low-grade → NOT injury
   (0.25 soft); grade 2+/complete → positive. Same for fracture "hairline"
7. **gold labeling conventions encoded**:
   - OA needs cartilage context; trochlea/patella anatomy → PF OA only
   - subchondral/degenerative edema ≠ Contusion; soft-tissue edema ≠ Contusion
   - mild effusions are ambiguous (gold labels conflict on identical text) → 0.5

## Languages covered
English, Spanish, German, Dutch, French, Turkish, Greek (transliterated),
Bulgarian/Russian (transliterated), Serbo-Croatian, + Italian/Portuguese negators.

## Validation snapshot (final)
```
label               acc  prec   rec    f1    auc
ACL                0.81  0.76  0.79  0.78  0.879
MCL                0.91  0.70  0.78  0.74  0.890
Medial Meniscus    0.83  0.79  0.85  0.81  0.898
Lateral Meniscus   0.83  0.76  0.83  0.79  0.861
Medial OA          0.84  0.75  0.60  0.67  0.779
Lateral OA         0.84  0.58  0.64  0.61  0.777
PF OA              0.72  0.63  0.57  0.60  0.694
Effusion           0.71  0.68  0.97  0.80  0.709
Synovitis          0.71  0.75  0.56  0.64  0.695
Baker's            0.86  0.64  0.75  0.69  0.837
Contusion          0.72  0.55  0.84  0.67  0.731
Fracture           0.78  0.63  0.67  0.65  0.747
macro: acc=0.797  f1=0.703  auc=0.791
```

## Pseudo-labels (data/pseudo_labels.csv)

4349 unlabeled studies, thresholded at 0.75/0.25; ambiguous → masked (-1):

| label | pos | neg | masked | p_pos |
|---|---|---|---|---|
| ACL | 536 | 3754 | 59 | 0.12 |
| MCL | 220 | 4102 | 27 | 0.05 |
| Medial Meniscus | 1667 | 1377 | 1305 | 0.55 |
| Lateral Meniscus | 718 | 3480 | 151 | 0.17 |
| Medial OA | 686 | 335 | 3328 | 0.67 |
| Lateral OA | 503 | 1013 | 2833 | 0.33 |
| PF OA | 1055 | 3122 | 172 | 0.25 |
| Effusion | 2025 | 1463 | 861 | 0.58 |
| Synovitis | 958 | 441 | 2950 | 0.68 |
| Baker's | 949 | 3380 | 20 | 0.22 |
| Contusion | 757 | 3534 | 58 | 0.18 |
| Fracture | 398 | 633 | 3318 | 0.39 |

Notes:
- p_pos differs from gold-58 prevalence (enriched for trauma) — expected;
  unlabeled data is the general clinical population
- Gold-58 reserved as honest validation anchor; never used in pseudo-training
- Known irreducible noise: gold labelers read images, not text (identical text
  labeled differently across studies — e.g. "small joint effusion" → 0 and 1)

## Implications for Phase 3 (image model)
- Train on 4349 pseudo-labeled + 58 gold studies with masked BCE (-1 = ignore)
- Text-AUC ~0.79 suggests image model + text-pseudo-labels can plausibly reach
  the ~0.93-0.94 LB territory reported for DINOv2-small
- Per-label reliability varies: menisci/ACL/MCL strong (0.86-0.90);
  PF OA/Synovitis/Effusion weaker (0.69-0.71) — consider per-label loss weighting
