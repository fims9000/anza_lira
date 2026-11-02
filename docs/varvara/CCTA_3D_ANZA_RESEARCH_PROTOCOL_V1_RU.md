# CT28 / 3D ANZA-LIRA — исследовательский протокол и готовность статьи

**Обновлено:** 2026-11-02. Ветка: `feature/ccta3d-research-contract-2026-11-02` (над PR #3 → PR #2 → PR #1). Никаких изменений в рабочей `research/coronary-connectivity-repair` Варвары; held-out test `954/958/972/973/980/984` не открывался.

## 1. Что именно исследуем

**Задача:** восстановить недостающие связи коронарной сосудистой сети на реальном 3D CCTA объёме, имея повреждённую маску сегментации, но не используя эталонную маску на этапе inference. Требуется одновременно (1) корректно определять тип и идентичность связей (PAIR/JUNCTION/BOTH/NONE), (2) конструировать анатомически правдоподобный путь в КТ и (3) избегать ложных соединений других сосудов, выбирая отказ при недостатке уверенности.

**Утверждение, которое надо проверить:** может ли локальная трёхмерная нечётко-анизотропная агрегация физических CCTA признаков (AZConv3d) улучшить селективное восстановление связности по сравнению с сопоставимой по размеру обычной 3D CNN, **при одинаковых сценах, физической геометрии, групповых patient-level разделениях, порогах и CT path/mask правилах**?

**Не заявлять заранее:** пользу ANZA относительно CNN, клиническую безопасность, свойство Аносова/эргодичность, независимую held-out генерализацию либо высокую точность на естественных разрывах сегментации. Все эти утверждения пока требуют результатов.

## 2. Математическая модель и правомерный вклад

В каждой физической точке `p` 3D encoder предсказывает для нечёткого режима `r`: вес принадлежности `mu_r(p)`, единичное направление `u_r(p)` и ограниченный параметр `h_r(p)`. Задаётся положительно определённый оператор

```math
G_r(p)=e^{2h_r(p)}u_r(p)u_r(p)^T+
e^{-h_r(p)}(I-u_r(p)u_r(p)^T),
\qquad \det G_r(p)=1.
```

Собственные значения `e^{2h},e^{-h},e^{-h}`, поэтому продольное растяжение компенсируется поперечным сжатием; гиперобъём метрического шара контролируется. Для физического смещения `delta=p-q` вес симметричного пространственного взаимодействия

```math
K_r(p,q)=\mu_r(p)\mu_r(q)
\exp\!\left[
-\frac{\beta}{2}
\left(
\delta^T G_r(p)\delta+\delta^T G_r(q)\delta
\right)
\right].
```

Далее по `r` выполняется нормированная агрегация признаков `sum_q K_r(p,q) f(q) / sum_q K_r(p,q)` и pointwise mixing с остаточной связью. Ядро симметрично при одновременном обмене `p,q` и `delta \to -delta`. В коде физические расстояния задаются в миллиметрах посредством `voxel_mm`, а направление при нулевой норме нормализуется через безопасную единичную ось. Это **конструкция/ограничение для нейросети**, а не доказательство динамической гиперболичности.

Сравниваемые candidate encoders `anza3d` и `conv3d` получают одинаковый `2 × D × H × W` патч: нормализованные HU и безымянную Gaussian-карту endpoint-координат. Патчи инвариантны к перестановке/именованию endpoints. Для PAIR и JUNCTION предусмотрены отдельные головы с бинарной weighted BCE.

Тренировочные предсказания для второго уровня должны быть **OOF по пациентам**: пациент с кодом `i` не входит в обучение candidate encoder, выдающего score для его же train scenes. Для пяти validation CT используется full-train encoder (17 пациентов). Затем CT score stream поступает в 84-признаковый relation HGB. На этапе volume inference сохраняются пороги `tau=.85`, perturbation consistency `>=.60`, запрет повторного endpoint через exact binary set-packing и выбранный CT path v2 с topology/mask veto. MILP использует `confidence × consistency` как **эвристическую utility**, а не калиброванную клиническую вероятность успеха.

**Существенная оговорка:** сейчас neural relation HGB переобучается на *clean* patient-OOF CT-признаках, тогда как базовый K10/15 relation HGB получен с 15 joint train-augmentations. Прямой сравнительный результат ANN vs K10/15, если он появится, будет **пилотным, не полностью capacity/augmentation-matched**. ANN vs CNN можно сравнивать непосредственно, поскольку у них одинаковый протокол обучения.

## 3. Что уже установлено численно

| Сравнение/этап | Число | Что именно доказывает |
|---|---:|---|
| Frozen CT28 cohort | 17 train / 5 val / 6 unopened test | Источники, разделённые по пациентам |
| Контролируемые train+val сцены | 1 404, из них 368 val | Синтетический графовый benchmark |
| **Count-only negative control** | **270/368 = 73,37%** relation-class accuracy | Сильную уязвимость к числу кандидатов; **вообще без чтения CT** |
| K10/15 selected val structural repair | **52 exact / 206 repair-needed; 0 observed false / 368** после selective gate | Только *постселективный* результат графовых сцен на 5 пациентах |
| K10/15 CT path v2 | 52/52 локально топологически построенных маршрутов | Достижимость выбранных endpoint-наборов; **не** качество финальной 3D маски |
| 3D NIfTI train pilot: K10/15 | **Не измерено на реальном CT** | Исполняемый код проверен синтетическими тестами |
| 3D CNN vs 3D ANZA на пациентских CT | **Не измерено** | Нужен локальный ImageCAS и trained checkpoints |
| 5 val full-mask Dice/clDice/false bridges | **Не измерено** | Пять пациентов уже использовались для подбора K10/15 |
| Шесть held-out full-mask patients | **Не открывались** | Нельзя использовать до окончательной фиксации протокола |

**Нельзя сравнивать** 73,37% count-only *relation accuracy* c 52/206 *accepted exact structural repair* как будто это одна метрика. Count-only контроль является фальсифицируемой гипотезой о shortcut, не метрикой точности восстановления сосудов. Нулевое число ложных принятия на пяти пациентах не означает нулевой ожидаемый риск.

## 4. Минимальный воспроизводимый запуск на Linux Варвары

Зависимости, модели и исходные CCTA тома остаются локальными (в Git исходные медицинские тома/веса не загружаем).

```bash
git fetch origin
git switch feature/ccta3d-research-contract-2026-11-02
source .venv-ccta/bin/activate
python -m pip install -r requirements-ccta-3d.txt
python -m pytest -q tests/test_ccta_3d_an_real_scene_contract.py tests/test_ct28_cardinality_benchmark.py tests/test_ct28_topology_metrics.py
```

Установить подходящий для её CUDA PyTorch. Сначала выполнить готовый train-only pilot и базовые сравнения (без открытия validation/test NIfTI):

```bash
python scripts/research/ccta_graph_lira_safe_repair/ct28_cardinality_benchmark.py \
  --output-dir varvara/anza_lira_local_results/cardinality_negative_control

python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_study.py \
  --stage all --patient-id 953 --device cuda --epochs 8 --folds 5 --batch-size 8
```

Затем обучить **отрицательные CT-контроли**, начиная с endpoint-only. Это отдельное кандидатное исследование, а **не** замена основной medical 3D inference модели:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_controls.py \
  --source-cache varvara/anza_lira_local_results/ccta_3d_an_patches \
  --mode endpoint_only \
  --output-dir varvara/anza_lira_local_results/ccta_3d_neg_endpoint_only

python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_oof.py \
  --cache-dir varvara/anza_lira_local_results/ccta_3d_neg_endpoint_only \
  --candidate-type PAIR --variant anza3d --device cuda \
  --output-dir varvara/anza_lira_local_results/ccta_3d_neg_endpoint_only_pair_anza
```

Аналогично при необходимости `--mode ct_only` и `--mode scrambled_ct`, и для `conv3d/JUNCTION`. Изначальные labels и кандидаты остаются идентичны, проверяются хэши. Основной relation/volume pipeline **преднамеренно отклоняет** шам-CT OOF, чтобы такие веса не были случайно объявлены обученными на изображениях.

## 5. Заморозить и запустить пять development-validation full-volume пациентов

**Только после train-пилота и окончательного выбора параметров.** Текущая заморозка фиксирует демонстрационный train-only режим: 3 разрыва, радиус разрыва 1.5 мм, центр-линейный repair radius 0, максимальный бюджет 250 новых voxel на decision. Это не оптимальные медицинские значения, и не нужно подгонять их по validation.

Создать неподменяемый manifest выбранной модели, path-протокола, 5 val IDs и ImageCAS CT SHA:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_freeze.py \
  --output varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_v1.json
```

Затем запустить реальные volume-проходы selected K10/15 и агрегировать все 5 пациентов без сокрытия ошибок:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_cohort.py \
  --freeze varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_v1.json \
  --run
```

После готовности четырёх OOF ANN packs и двух clean relation heads из train этапа, те же пять повреждённых масок без подмены параметров:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_an_compare.py \
  --freeze varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_v1.json \
  --run --device cuda
```

Будут сохранены самостоятельные NIfTI `repaired.nii.gz`, `added_voxels.nii.gz`, `decisions.json`, `posthoc_metrics.json` по каждому пациенту и варианту, а в корне — `validation_cohort_results.md` и `matched_val_comparison.md`. Если хотя бы один пациент отсутствует или падает, сводка **не может** получить статус complete. Повторный запуск не переписывает результаты.

Если нужно выделить реальные 3 случая ошибок (не выдумывать их, если событий меньше):

```bash
OUT=varvara/anza_lira_local_results/ccta_3d_validation/patient_957
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_failure_triage.py \
  --patient-id 957 \
  --decisions "$OUT/decisions.json" \
  --metrics "$OUT/posthoc_metrics.json" \
  --output "$OUT/paper_failure_cases.json"
```

Для иллюстраций в 3D Slicer загрузить CT, damaged/repaired masks и added-voxel overlay. В `posthoc_metrics.json` также записаны **GT-inconsistent inter-component bridge pairs**, GT-unknown pairs и 3D координаты representative voxels для ручной проверки. Этот счётчик не выявляет ложных соединений между ветвями, которые уже принадлежат одной GT-компоненте; требуется анатомический аудит врача/эксперта.

## 6. Критерии пригодности для статьи

1. **Данные и воспроизводимость:** указаны версия ImageCAS, dataset ID, 17/5/6 split, source SHAs, affine/spacing, протокол controlled errors, реальные predicted masks если используются. Контроль `train-only 953` отдельно от `five development val` и **отдельно** от будущего blind held-out.
2. **Справедливое сравнение:** geometry/count-only, original radial CT K10, matched-size CNN, ANZA; одинаковый candidate pool/target, минимум 3 повторных seed на CNN и ANZA, patient-level OOF, explicit runtime VRAM/seconds, train/val scores без дублирования пациента.
3. **Защита от shortcut:** patient-invariant endpoint ID metamorphic test, count-only majority 270/368, endpoint-only/shuffled CT negative controls, subgroup analysis `candidate_count` и `PAIR/JUNCTION/BOTH`. Если CNN/ANZA не превосходит endpoint-only, **CT contribution не доказан**.
4. **Анатомия:** Dice, clDice, компонентный Betti-0/count difference, precision новых voxel vs GT, число GT-inconsistent/unknown inter-component bridges, минимум 3 *реально возникших* failure cases с CT overlay и причинным объяснением. Нулевой неверный bridge на коротком испытании не доказывает клиническую безопасность.
5. **Конкуренты:** сравнение с геодезическими методами восстановления сосудов важно для статьи. GeodVessel (STACOM 2026) — релевантный CT-path comparator, но **в этой реализации он не воспроизведён**, поэтому не приписывать себе преимущества над ним.
6. **Математика:** доказаны только свойства предложенного локального ядра (SPD, det=1, pairwise symmetry), а не обобщаемость, topological homotopy guarantee или Anosov hyperbolicity. Любую пользу архитектуры подтверждать реальными patient-level экспериментами и абляциями.
7. **Фиксация:** после завершения train/validation зафиксировать окончательные ANN augmentations, solver, threshold, candidate generator, патчи и веса; только потом проводить **однократное** blind evaluation 6 held-out пациентов без retuning.

## 7. Работы для цитирования и честное позиционирование

- Shit et al., CVPR 2021, *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*. Основной первоисточник clDice, отличать измерение clDice от строгой гарантии корректной анатомии. [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html).
- *GeodVessel: A Geodesic Cost-Map Framework for Refining Coronary Vessel Connectivity*, MICCAI 2026 STACOM. Прямо связан с CCTA восстановлением разрывов, интенсивностью и геодезическим путём. [MICCAI 2026](https://papers.miccai.org/miccai-2026-sat/STACOM_023.html).
- Qiu et al., *A topology-preserving three-stage framework for fully-connected coronary artery extraction*, Medical Image Analysis 2025, DOI 10.1016/j.media.2025.103578. Важен как полноценный coronary extraction comparator, а не идентичный patch postprocessing baseline. [Elsevier](https://www.sciencedirect.com/science/article/pii/S1361841525001252).

**Статус handoff:** код математики/OOF/3D NIfTI/масок/валидационных и негативных проверок реализован; actual five-patient full-mask metrics требуют локального архива Варвары. Результаты не заполнять искусственными значениями.
