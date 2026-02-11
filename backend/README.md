# Dataset Pipeline – Steps

## 1️⃣ Download Images

**Script**
`backend/scripts/download_kaggle_images.py`

**Command**

```bash
python download_kaggle_images.py
```

**Reads**

```
data/datasets/kaggle_prefiltered.csv
```

**Creates**

```
data/images/kaggle_raw/
```

**Purpose**
Downloads images incrementally without duplicates.

---

## 2️⃣ Visual Filtering (Scoring)

**Script**
`backend/scripts/filter_interiors.py`

**Command**

```bash
python filter_interiors.py
```

**Reads**

```
data/images/kaggle_raw/
```

**Creates / Updates**

```
data/datasets/interior_filter_pro.csv
```

**Purpose**
Analyzes images and generates an `indoor_score` based on visual features.

---

## 3️⃣ Preview Results

**Script**
`backend/scripts/preview_filter_results.py`

**Command**

```bash
python preview_filter_results.py
```

**Reads**

```
data/datasets/interior_filter_pro.csv
```

**Creates**

```
data/images/filter_preview/
├── interiors/
└── rejected/
```

**Purpose**
Copies sample images to visually validate filtering results.

---

## 4️⃣ Semantic Filtering (YOLO)

**Script**
`backend/scripts/yolo_semantic_filter.py`

**Command**

```bash
python yolo_semantic_filter.py
```

**Reads**

```
data/datasets/interior_filter_pro.csv
```

**Creates**

```
data/datasets/interior_semantic.csv
```

**Purpose**
Runs a pretrained YOLO model to detect interior-related objects (bed, couch, chair, table, tv, etc.) and adds a semantic validation layer on top of the visual filter.

---

## 5️⃣ Create Final Dataset (Visual + YOLO Fusion)

**Script**
`backend/scripts/create_final_dataset.py`

**Command**

```bash
python create_final_dataset.py
```

**Reads**

```
data/datasets/interior_semantic.csv
```

**Creates**

```
data/datasets/interior_final_candidates.csv
```

**Purpose**
Combines visual filtering and YOLO semantic validation to generate a clean dataset of confirmed interior images.
Adds an initial automatic `quality_bucket` label (good / medium / bad) and supports optional human labeling.

---

## ⚠️ Notes

* Scripts never modify original images.
* Preview folders contain copies and can be deleted anytime.
* `interior_filter_pro.csv` stores only metadata and scores, not images.
* YOLO adds semantic information but does not move or delete files.
* The final dataset step is the entry point for human-in-the-loop quality labeling.
