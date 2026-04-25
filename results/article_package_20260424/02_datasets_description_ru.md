# Описание Датасетов

Источники:
- структура загрузчиков: `utils.py` (`DriveDataset`, `ArcadeVesselDataset`);
- фактические размеры в `data/*` (подсчитаны по файлам и ARCADE json).

## 1) DRIVE

- Директория: `data/DRIVE`
- Формат: `training/` и `test/`, внутри `images/`, `1st_manual/`, `mask/`
- Размеры:
  - training images: **20**
  - test images: **20**
- Валидация в коде строится из training через `val_fraction` (в benchmark-конфиге `0.2`):
  - train subset: 16
  - val subset: 4
- Валидная область для метрик: маска FOV (`use_fov_mask: true`).

## 2) CHASE-DB1

- Директория: `data/CHASE_DB1`
- Формат такой же, как у DRIVE (`images`, `1st_manual`, `mask`).
- Размеры:
  - training images: **8**
  - test images: **20**
- В benchmark-конфиге `val_fraction: 0.25`:
  - train subset: 6
  - val subset: 2

## 3) FIVES

- Директория: `data/FIVES`
- Формат такой же, как у DRIVE (`images`, `1st_manual`, `mask`).
- Размеры:
  - training images: **600**
  - test images: **200**
- В benchmark-конфиге `val_fraction: 0.15`:
  - train subset: 510
  - val subset: 90

## 4) ARCADE Syntax

- Фактический root по загрузчику: `data/ARCADE/arcade`
- Формат: `syntax/{train,val,test}/images` + `annotations/{split}.json`
- Размеры по json:
  - train: **1000** images, **4976** annotations
  - val: **200** images, **1168** annotations
  - test: **300** images, **1672** annotations
- Маски строятся из polygon annotation (COCO-like segmentation).

## 5) ARCADE Stenosis

- Формат: `stenosis/{train,val,test}/images` + `annotations/{split}.json`
- Размеры по json:
  - train: **1000** images, **1625** annotations
  - val: **200** images, **406** annotations
  - test: **300** images, **386** annotations
- Логика загрузки и маскообразования такая же, как в ARCADE Syntax.

## 6) Единый preprocessing-протокол

Общее:
- RGB вход;
- нормализация ImageNet mean/std;
- бинарные маски (`>127` -> 1).

Ретинальные датасеты (DRIVE/CHASE/FIVES):
- поддержка patch-crop;
- foreground/thin-vessel/hard-mining bias sampling;
- флипы/повороты и фотометрические джиттеры для train.

ARCADE:
- опциональный resize (`arcade_image_size`);
- patch-crop для train (`arcade_patch_size`);
- флипы/повороты для train;
- valid mask = единицы (поле обзора не ограничивается отдельной FOV-маской).

## 7) Текст для статьи (готовый абзац)

`We evaluate retinal and coronary vessel segmentation on DRIVE, CHASE-DB1, FIVES, and ARCADE (syntax/stenosis). For retinal datasets, we use paired RGB images, manual vessel masks, and field-of-view masks. For ARCADE, binary masks are rasterized from polygon annotations in split-wise JSON files. Across datasets, images are normalized using ImageNet statistics; train-time augmentation includes flips/rotations and dataset-specific crop sampling.`

