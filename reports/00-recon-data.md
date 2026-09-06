# Phase 0 recon — CSV profile

## STRATEGIC IMPLICATIONS (read first)

1. **Only 58/4407 studies (1.3%) carry labels — and they're fully labeled (all 12).**
   This is a weak-supervision competition, not a standard supervised one.
   Text pseudo-labeling is the core of the problem, not an optional boost.
2. **All 4407 studies have reports.** Reports explicitly state findings
   (e.g. "Rotura de menisco interno ... Artrosis femorotibial medial. Derrame."
   = Medial Meniscus=1, Medial OA=1, Effusion=1). The text task is closer to
   multilingual information extraction than fuzzy classification.
   -> Phase 2 (text) is the highest-leverage phase; validate against the 58
   gold studies (LOO/5-fold CV on text models).
3. **Languages (keyword-detected, confirmed by raw samples):** English 39.3%,
   Turkish 15.5%, Spanish 15.5%, German 12.0%, Cyrillic (BG/RU) 5.0%,
   Dutch 3.4%, French 1.9%, other/unknown 7.4%. At least 7 languages —
   includes Turkish and Cyrillic which keyword-rule design must cover.
   58 gold studies are too few for pure ML text classification across
   languages — combine multilingual keyword extraction with char-ngram models.
4. **Fluid_Sensitive ≡ Fat_Suppression in train (perfect diagonal crosstab)**
   but organizers warn they may differ on hidden test. Keep both flags in the
   pipeline; select series on Fluid_Sensitive but don't merge the concepts.
5. **Series selection: 100% of studies fill 3 priority slots** (sag-fluid 94.2%,
   cor-fluid 96.4%, ax-fluid 100%, fallbacks cover the rest). 3-series plan is safe.
   Series/study: mean 5.53, min 3, max 14.
6. **Image model will train on ~4407 pseudo-labeled + 58 gold studies.**
   Gold 58 are precious: reserve them as the honest validation anchor
   (text-extractor quality AND final image-model sanity check).

train.csv: 4407 studies, columns: ['StudyInstanceUID', 'Report', 'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
train_series.csv: 24371 rows

## Label availability
- studies with >=1 label: 58 / 4407 (1.3%)
- studies with all 12 labels: 58
- studies with 0 labels: 4349
- labels-per-study distribution:
     0 labels: 4349 studies
    12 labels: 58 studies

## Label prevalence (among labeled entries for that label)
label              n_labeled   n_pos  prevalence
ACL                       58      24       0.414
MCL                       58       9       0.155
Medial Meniscus           58      26       0.448
Lateral Meniscus          58      23       0.397
Medial OA                 58      15       0.259
Lateral OA                58      11       0.190
PF OA                     58      21       0.362
Effusion                  58      35       0.603
Synovitis                 58      27       0.466
Baker's                   58      12       0.207
Contusion                 58      19       0.328
Fracture                  58      18       0.310

## Label correlation (Pearson, labeled pairs only, |r|>0.3 shown)

## Reports
- studies with non-empty report: 4407 (100.0%)
- report length chars: mean=1098, p50=977, p95=2453
- detected languages (keyword-based, keyword counts):
    english: 1734 studies (39.3%)
    turkish: 685 studies (15.5%)
    spanish: 682 studies (15.5%)
    german: 528 studies (12.0%)
    other/unknown: 324 studies (7.4%)
    cyrillic (bulgarian/russian): 220 studies (5.0%)
    dutch: 152 studies (3.4%)
    french: 82 studies (1.9%)

## Series structure
- series per study: mean=5.53, p50=5, min=3, max=14
- studies in train_series.csv: 4407
- studies in train.csv missing from series table: 0
- plane counts: {'Sagittal': 9864, 'Coronal': 8609, 'Axial': 5898}
- fluid-sensitive counts: {1: 14010, 0: 10361}
- Fluid_Sensitive x Fat_Suppression crosstab:
Fat_Suppression      0      1
Fluid_Sensitive              
0                10361      0
1                    0  14010
- bucket sizes (plane, fluid):
    Sagittal   fluid=0: 5197
    Axial      fluid=1: 4719
    Sagittal   fluid=1: 4667
    Coronal    fluid=1: 4624
    Coronal    fluid=0: 3985
    Axial      fluid=0: 1179

## Priority-bucket coverage per study
    Sagittal   fluid=1: 4150 studies (94.2%)
    Coronal    fluid=1: 4248 studies (96.4%)
    Axial      fluid=1: 4407 studies (100.0%)
    Sagittal   fluid=0: 4266 studies (96.8%)
    Coronal    fluid=0: 3406 studies (77.3%)
    Axial      fluid=0: 857 studies (19.4%)
    -> with 3-series selection: 4407 studies fill all 3 slots (100.0%); distribution of filled slots: {1: 0, 2: 0, 3: 4407}

## Example test studies
- test_series.csv rows: 15, studies: 3
- planes available per test study: {'1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191': ['Axial', 'Sagittal', 'Sagittal', 'Coronal', 'Axial'], '1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776': ['Sagittal', 'Axial', 'Sagittal', 'Sagittal', 'Coronal'], '1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402': ['Sagittal', 'Sagittal', 'Axial', 'Coronal', 'Coronal']}

