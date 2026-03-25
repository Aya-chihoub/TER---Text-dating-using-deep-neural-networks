# Architecture du CNN 1D — Guide d'implémentation

Ce document décrit l'architecture complète du réseau et comment chaque module
s'articule. Tu peux t'en servir pour coder l'entraînement (`train.py`).

---

## Vue d'ensemble

```
Texte brut
   │
   ▼
Tokenisation (split sur les espaces)
   │
   ▼
Extraction de 14 features par mot  ←  src/data/features.py (déjà codé)
   │
   ▼
Fenêtre glissante : 1000 mots × 14 features  ←  src/data/dataset.py (déjà codé)
   │
   ▼
CNN 1D multi-branches  ←  src/models/cnn.py (déjà codé)
   │
   ▼
Prédiction : 1 âge parmi 41 classes (30–70 ans)
```

---

## 1. Les données d'entrée

Chaque sample est un tenseur de shape **(batch, 1000, 14)** :

| Dimension | Signification |
|-----------|---------------|
| batch     | Nombre de textes dans le mini-batch (par défaut 32) |
| 1000      | Nombre de mots (fenêtre glissante, stride = 500) |
| 14        | Nombre de features par mot |

Les 14 features (calculées par `FormFeatureExtractor`) :

| #  | Feature             | Groupe        |
|----|---------------------|---------------|
| 1  | freq_in_text        | Fréquence     |
| 2  | log_freq            | Fréquence     |
| 3  | freq_rank           | Fréquence     |
| 4  | global_freq         | Fréquence     |
| 5  | word_length         | Longueur      |
| 6  | syllable_count      | Longueur      |
| 7  | vowel_ratio         | Composition   |
| 8  | accent_type         | Composition   |
| 9  | punctuation_type    | Lexical       |
| 10 | pos_tag             | Lexical       |
| 11 | pos_in_sentence     | Positionnel   |
| 12 | sentence_length     | Positionnel   |
| 13 | is_sentence_boundary| Positionnel   |
| 14 | adjacent_to_period  | Contexte      |

---

## 2. Architecture du CNN

### Principe : 3 branches parallèles × 3 couches empilées

Le réseau a **3 branches qui tournent en parallèle**, chacune avec un kernel
size différent. Chaque branche empile **3 couches de Conv1d + MaxPool**.

```
                        Input (batch, 1000, 14)
                                 │
                         Transpose → (batch, 14, 1000)
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
          Branche k=3      Branche k=7      Branche k=13
                │                │                │
        ┌───────┴───────┐ ┌─────┴──────┐  ┌──────┴──────┐
        │ Conv1d(k=3)   │ │ Conv1d(k=7)│  │Conv1d(k=13) │
        │ BatchNorm     │ │ BatchNorm  │  │ BatchNorm   │
        │ ReLU          │ │ ReLU       │  │ ReLU        │
        │ MaxPool(2)    │ │ MaxPool(2) │  │ MaxPool(2)  │
        ├───────────────┤ ├────────────┤  ├─────────────┤
        │ Conv1d(k=3)   │ │ Conv1d(k=7)│  │Conv1d(k=13) │
        │ BatchNorm     │ │ BatchNorm  │  │ BatchNorm   │
        │ ReLU          │ │ ReLU       │  │ ReLU        │
        │ MaxPool(2)    │ │ MaxPool(2) │  │ MaxPool(2)  │
        ├───────────────┤ ├────────────┤  ├─────────────┤
        │ Conv1d(k=3)   │ │ Conv1d(k=7)│  │Conv1d(k=13) │
        │ BatchNorm     │ │ BatchNorm  │  │ BatchNorm   │
        │ ReLU          │ │ ReLU       │  │ ReLU        │
        │ MaxPool(2)    │ │ MaxPool(2) │  │ MaxPool(2)  │
        └───────┬───────┘ └─────┬──────┘  └──────┬──────┘
                │                │                │
         GlobalAvgPool    GlobalAvgPool    GlobalAvgPool
          → (batch,64)    → (batch,64)     → (batch,64)
                │                │                │
                └────────────────┼────────────────┘
                                 │
                    Concaténation → (batch, 192)
                                 │
                         Dropout (p=0.3)
                                 │
                     Linéaire (192 → 41)
                                 │
                        Prédiction (âge)
```

### Détail des dimensions dans chaque branche

La séquence rétrécit à chaque couche (Conv1d enlève `k-1` positions,
MaxPool divise par 2) :

**Branche k=3 :**
```
Entrée :        (batch, 14, 1000)
Couche 1 Conv : (batch, 64, 998)   → MaxPool → (batch, 64, 499)
Couche 2 Conv : (batch, 64, 497)   → MaxPool → (batch, 64, 248)
Couche 3 Conv : (batch, 64, 246)   → MaxPool → (batch, 64, 123)
GlobalAvgPool : (batch, 64)
```

**Branche k=7 :**
```
Entrée :        (batch, 14, 1000)
Couche 1 Conv : (batch, 64, 994)   → MaxPool → (batch, 64, 497)
Couche 2 Conv : (batch, 64, 491)   → MaxPool → (batch, 64, 245)
Couche 3 Conv : (batch, 64, 239)   → MaxPool → (batch, 64, 119)
GlobalAvgPool : (batch, 64)
```

**Branche k=13 :**
```
Entrée :        (batch, 14, 1000)
Couche 1 Conv : (batch, 64, 988)   → MaxPool → (batch, 64, 494)
Couche 2 Conv : (batch, 64, 482)   → MaxPool → (batch, 64, 241)
Couche 3 Conv : (batch, 64, 229)   → MaxPool → (batch, 64, 114)
GlobalAvgPool : (batch, 64)
```

### Pourquoi ces kernel sizes ?

| Kernel | Ce qu'il capture |
|--------|------------------|
| k=3    | Motifs locaux : bigrammes/trigrammes (ex : « ne...pas ») |
| k=7    | Motifs syntagmatiques : tournures de phrase |
| k=13   | Structures de proposition/phrase complète |

### Pourquoi MaxPool(2) ?

À chaque couche, MaxPool garde **1 valeur sur 2** (la plus forte).
Après 3 couches, la séquence est réduite à environ 1/8 de sa taille.
Cela force le réseau à ne garder que les motifs les plus importants.

---

## 3. Hyperparamètres (dans `src/utils/config.py`)

| Paramètre | Valeur | Où |
|-----------|--------|----|
| `sequence_length` | 1000 | `DataConfig` |
| `stride` | 500 | `DataConfig` |
| `kernel_sizes` | [3, 7, 13] | `ModelConfig` |
| `num_filters` | 64 | `ModelConfig` |
| `pool_size` | 2 | `ModelConfig` |
| `num_conv_layers` | 3 | `ModelConfig` |
| `dropout` | 0.3 | `ModelConfig` |
| `batch_size` | 32 | `TrainingConfig` |
| `learning_rate` | 1e-3 | `TrainingConfig` |
| `weight_decay` | 1e-4 | `TrainingConfig` |
| `num_epochs` | 50 | `TrainingConfig` |
| `patience` | 10 | `TrainingConfig` |
| `num_classes` | 41 | 30 à 70 ans |

---

## 4. Ce qui est déjà codé vs ce qu'il reste à faire

### Déjà fait :
- `src/data/parser.py` — Lecture du corpus, extraction des métadonnées (nom, âge, etc.)
- `src/data/features.py` — Extraction des 14 features par mot
- `src/data/dataset.py` — Dataset PyTorch (fenêtre glissante, padding, labels)
- `src/data/split.py` — Split train/test stratifié par âge
- `src/models/cnn.py` — Le modèle CNN (classe `AgeCNN`)
- `src/utils/config.py` — Toute la configuration centralisée

### À coder :
- **`train.py`** — La boucle d'entraînement

---

## 5. Guide pour coder `train.py`

Voici le squelette de ce qu'il faut faire :

```python
# 1. Charger la config
from src.utils.config import get_default_configs
data_cfg, feat_cfg, model_cfg, train_cfg = get_default_configs()

# 2. Charger le corpus et splitter
from src.data.parser import load_corpus
from src.data.split import stratified_split
entries = load_corpus(data_cfg.corpus_dir)
train_entries, test_entries = stratified_split(entries, n_test_per_age=data_cfg.n_test_per_age)

# 3. Construire les datasets
from src.data.dataset import build_datasets
train_ds, test_ds = build_datasets(
    train_entries, test_entries,
    sequence_length=data_cfg.sequence_length,
    train_stride=data_cfg.stride,
)

# 4. DataLoaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=train_cfg.batch_size, shuffle=False)

# 5. Modèle, loss, optimizer
from src.models.cnn import AgeCNN
model = AgeCNN(model_cfg).to(train_cfg.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=train_cfg.learning_rate,
    weight_decay=train_cfg.weight_decay,
)

# 6. Boucle d'entraînement
for epoch in range(train_cfg.num_epochs):
    model.train()
    for features, labels in train_loader:
        features = features.to(train_cfg.device)  # (batch, 1000, 14)
        labels   = labels.to(train_cfg.device)     # (batch,)

        logits = model(features)                   # (batch, 41)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 7. Évaluation sur le test set
    model.eval()
    # ... calculer accuracy, MAE, etc.
```

### Points importants :
- Le dataset renvoie `(features, label)` où features est shape `(1000, 14)`
  et label est un int entre 0 et 40 (= âge - 30)
- Le modèle attend `(batch, 1000, 14)` et renvoie `(batch, 41)` logits
- La loss est `CrossEntropyLoss` (classification en 41 classes)
- Penser à ajouter un **learning rate scheduler** (`ReduceLROnPlateau`)
  et de l'**early stopping** (patience = 10 epochs)
