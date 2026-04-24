# Гайд по визуализациям (понятно для читателя)

## 1) Geometry Direction Field
- Что показывает: в каком направлении локально «смотрит» геометрический блок.
- Как читать: стрелки и цвет показывают доминирующую ориентацию структуры.
- Интерпретация: согласованные направления вдоль сосуда/линейного объекта должны быть непрерывными.

## 2) Geometry Contribution Map
- Что показывает: где геометрическая анизотропия усиливает полезный сигнал и где подавляет шум.
- Цвета:
  - оранжевый: усиление вдоль полезной оси (elongation);
  - синий: подавление поперечного шума/фоновых ответов (suppression).

## 3) Local Geometry Regimes
- Что показывает: какой локальный геометрический режим (правило) доминирует в каждом пикселе.
- Зачем: видно, как модель делит изображение на области с разной локальной геометрией.

## 4) Regime Confidence
- Что показывает: насколько уверенно выбран доминирующий режим.
- Как читать: тёплые цвета = высокая уверенность, холодные = низкая.

## 5) Geometry Contribution vs Baseline
- Что показывает: разницу между предложенным методом и baseline.
- Цвета:
  - зелёный: предложенный метод исправил ошибку baseline на целевом объекте;
  - красный: предложенный метод ухудшил результат;
  - голубой: убран ложный позитив baseline;
  - оранжевый: добавлен новый ложный позитив.

## 6) Готовые подписи для статьи
- Figure 1 (pipeline): `Overall segmentation pipeline with geometry-aware local aggregation and final binary mask generation.`
- Figure 2 (comparison): `Comparison with baseline on hard local regions. The proposed method better preserves thin elongated structures and recovers missed branches.`
- Figure 3 (XAI): `Layer-wise geometry attention view. Direction maps show dominant local orientation; contribution maps show where anisotropic geometry amplifies vessel-like evidence and suppresses cross-structure noise.`

## 7) Как объяснить обычному читателю в 2 фразах
- `Модель не просто ищет яркие пиксели, а учитывает локальное направление структуры и усиливает согласованные вытянутые фрагменты.`
- `За счёт этого лучше сохраняется связность тонких ветвей и уменьшается поперечный шум по сравнению с baseline.`
