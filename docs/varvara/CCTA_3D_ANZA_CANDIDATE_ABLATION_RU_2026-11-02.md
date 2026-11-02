# 3D ANZA-LIRA для CCTA: математический оператор и контролируемый CNN baseline

Статус: **исследовательская реализация, не валидированная на реальных КТ**. Ветка `feature/ccta-3d-anza-encoder-2026-11-02`; PR #3; построена поверх PR #2 и PR #1. Ветка Варвары `research/coronary-connectivity-repair` и её выбранный K10/15 не изменены.

## 1. Зачем добавлен собственный 3D ANZA

Исходная CT28 Graph-LIRA извлекает geometry/CT признаки вручную, применяет локальные HGB и 84-признаковый HGB для relation type. Это рабочий baseline, **но не научное доказательство пользы архитектуры ANZA** на 3D КТ.

В `models/azconv3d.py` реализован новый локальный fuzzy anisotropic 3D operator. Он по локальным данным вычисляет для каждого правила r:
- вес принадлежности `mu_r(p)` через softmax;
- нормированный физический 3D вектор `u_r(p)`;
- обучаемую ограниченную анизотропию `h_r(p)`;
- локально согласованное соседское ядро.

Метрический тензор

\[
G_r(p)=e^{2h_r(p)}u_ru_r^T+e^{-h_r(p)}(I-u_ru_r^T)
\]

имеет собственные значения `e^{2h}, e^{-h}, e^{-h}`, потому **det G=1**. Следовательно увеличение направленной чувствительности компенсируется поперечным сжатием. Это корректное математическое ограничение, **не доказательство существования динамики Аносова, гиперболического аттрактора или эргодичности**.

Для физического смещения `delta` между вокселями

\[
K_r(p,q)=\mu_r(p)\mu_r(q)
\exp[-\beta(\delta^TG_r(p)\delta+\delta^TG_r(q)\delta)/2].
\]

Этот kernel симметричен при перестановке `p,q` с изменением знака `delta`. Направления и принадлежности извлекаются из локального CT feature field; используется массив физических 3D offset-векторов с заданным affine basis. Выход — rule-wise нормализованная агрегация признаков с обучаемым pointwise mix и residual.

**Уже проверяются математические инварианты:** det=1, симметрия pair kernel, физическое spacing/affine, autograd, нулевая orientation fallback, анизотропная vs изотропная абляция, корректные размеры и конечные значения.

## 2. Что сравниваем и почему

`models/ccta3d_candidate_encoder.py` предоставляет два классификатора для одного кандидата:

| Вариант | CT representation |
|---|---|
| `conv3d` | Stem Conv3d + depthwise separable 3D convolution + одна и та же FC-head |
| `anza3d` | Stem Conv3d + предлагаемый AZConv3d + та же FC-head |

Оба получают **один и тот же** 2-канальный patch:
1. интенсивность CT после clip HU [−300, 1200] и нормализации в [−1,1];
2. гауссовы локальные 3D heatmaps концов текущего кандидата (без имён, метки класса и source IDs).

Патч строится в физических LPS-миллиметрах относительно centroid эндпоинтов; локальная система координат ортонормальная, **det=+1**. Размер по умолчанию 24³ при 1 мм на voxel. Для PAIR и JUNCTION обучаются отдельные бинарные heads.

Модельно-ёмкостный контроль: параметров двух альтернатив должно различаться не более чем на 20%. У обеих моделей одинаковые патчи, train/validation пациенты, optimizer AdamW, pos_weight из train, learning rate, число epochs и seed. Validation **не используется для выбора лучшего чекпоинта**.

## 3. Команды на Linux у Варвары

В корне клона ANZA-LIRA:

```bash
git fetch origin
git switch feature/ccta-3d-anza-encoder-2026-11-02
source .venv-ccta/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pytest -q tests/test_azconv3d.py tests/test_ccta3d_candidate_encoder.py tests/test_ccta_3d_an_patches.py tests/test_ccta_3d_an_patch_cache.py tests/test_ccta_3d_an_train.py
```

Если устройство не поддерживает CUDA 12.8, выбрать совместимую установку PyTorch для текущей CUDA. На CPU использовать `--device cpu`.

Сначала извлечь patch cache **только для 17 train CT28 пациентов** из локального уже проверенного ImageCAS архива:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_patch_cache.py
```

Подготовить 5 validation-patient patches **отдельно**, только для post-train evaluation:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_patch_cache.py \
  --patients 957 961 965 966 974 --include-validation
```

Оба шага записывают `patches_<patient>.npz` и `patches_<patient>.json` в игнорируемый Git путь `varvara/anza_lira_local_results/ccta_3d_an_patches`. В `.json` сохраняются SHA источников, CT/patch geometry и факт отсутствия labels в сетевом входе. Все six held-out test CT28 пациентов заблокированы.

После этого сравнить кандидаты PAIR:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_train.py \
  --candidate-type PAIR --device cuda \
  --epochs 8 --batch-size 8 --seed 20261102 \
  --output-dir varvara/anza_lira_local_results/ccta_3d_an_pair
```

Аналогично JUNCTION:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_train.py \
  --candidate-type JUNCTION --device cuda \
  --epochs 8 --batch-size 8 --seed 20261102 \
  --output-dir varvara/anza_lira_local_results/ccta_3d_an_junction
```

Значения 8 epochs, seed и batch — **начальный проверяемый экспериментальный режим**, не доказанная настройка по качеству. Для сопоставимого исследования лучше зафиксировать заранее число epochs, learning-rate schedule и train-only cross-validation, затем выполнить несколько фиксированных seeds на обеих архитектурах. Не менять выбор только потому, что ANZA хуже на пяти validation пациентах.

В каждом output каталоге:
- `experiment.json`: архитектура, параметры и число весов, train/validation пациенты, ROC-AUC/AP, по-пациентные результаты, CPU/GPU, время и SHA файлов;
- `<TYPE>_anza3d.pt` и `<TYPE>_conv3d.pt`: model state dict;
- `<TYPE>_anza3d_val_scores.csv` и `<TYPE>_conv3d_val_scores.csv`: validation candidate scores и labels для сравнения.

## 4. Чего этот эксперимент ещё НЕ делает

Это **локальная candidate classification**. Она не подключена к relation head K10/15 и не может заменить старые HGB CT scores без нового протокола patient-level OOF train candidate scoring. Иначе появится in-sample leakage при обучении следующего уровня.

Чтобы проверить реальный вклад 3D ANZA в Graph-LIRA, нужна последовательность:

1. TRAIN: для каждой из 17 train CT извлечь OOF CNN и OOF ANZA candidate scores из моделей, которые **не обучались на этом пациенте**.
2. На одинаковых OOF score streams обучить CT-conditioned relation head; архитектура/пороги/consistency должны быть сопоставимыми и зафиксированными.
3. VALIDATION: оценить frozen K10 baseline, CNN-score variant и ANZA-score variant на одинаковых controlled сценах; затем на full-volume damaged 3D masks.
4. Freeze протокол выбора и отказа от repair, затем одноразовый Graph-LIRA held-out test. Тест нельзя использовать для выбора ANZA гиперпараметров.

**Научно недопустимо:** объявить ANZA победителем по абстрактной симметрии ядра или train-only примерам, подменить 52/0 на метрики нового encoder без полного pipeline, не указать генератор искусственных сцен/shortcuts.

Сейчас в PR #3 предоставлены математический 3D оператор, сопоставимый CNN baseline и реальная функция обучения на train-кандидатах. Для медицинских метрик требуется локальный архив Варвары, которого нет в GitHub.
