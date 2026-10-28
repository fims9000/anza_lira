# ANZA–LIRA: полный исследовательский исходный код и подключение Graph-LIRA

**28 октября 2026** · ветка [`research/bubble-chamber-graph-lira`](https://github.com/fims9000/anza_lira/tree/research/bubble-chamber-graph-lira) · **RESEARCH ONLY, НЕ ИЗМЕНЯТЬ НАСТОЯЩИЕ ТРЕКИ АВТОМАТИЧЕСКИ**

Этот комплект включает **оригинальную ANZA**, разные исторические версии **LIRA**, Graph-LIRA для пузырьковой камеры, заменяемые детекторы, исследования S1–S85, скрипты, тесты и документацию. Это не просто Graph-LIRA-адаптер. Архив исходников формируется из конкретного Git-коммита с SHA-256 для каждого файла.

> Исходники и веса — разные вещи: **веса ANZA/S56/S85, H50Z BMP/PNG, измеренные film070-точки и экспертные ответы НЕ включаются**. Обученная модель и идентичные её калибровке входные данные необходимы для научного воспроизведения реальных экспериментов. Не подменяйте их генератором в отчётах.

## 1. Состав полного проекта

| Раздел | Назначение |
|---|---|
| `models/azconv.py`, `models/segmentation.py` | **Оригинальная ANZA:** AZConv и AZUNet, а также обычный BaselineUNet для честного сравнения |
| `anza2/`, `anza_*/`, `models/anza2/` | Разные исследовательские архитектуры/варианты ANZA; не считать автоматически обученными |
| `lira_final/`, `lira_graph_cut_v2/`, `lira_*/`, `anza_tracegraph/` | Исходные проекты LIRA/графовых методов для структур и сейсмики |
| `bubble_chamber/`, `scripts/research/bubble_chamber/` | Исследование 2D треков пузырьковой камеры S1–S85, S56/S59/S72, S69 аудитор, S80–S85 |
| `bubble_chamber/interop_*.py` | Независимый от детектора интерфейс 2D-полилиний, S56-пары, стратегия Graph-LIRA и калибровка |
| `bubble_chamber/interop_anza_frontend.py` | **НОВОЕ:** подлинный AZUNet checkpoint → сегментация → скелет → полилинии `(x,y)` → тот же interop API |
| `tests/`, `docs/`, `scripts/`, `configs/` | Контрактные и научные проверки, воспроизводимые настройки, отчёты, инструкции |

Оригинальная **сейсмическая LIRA** и экспериментальный **S56 Graph-LIRA** для H50Z имеют общее происхождение идей, но различаются задачей, признаками и протоколами. Не заменяйте один другим без повторной проверки.

## 2. Установка и минимальная проверка

В корне извлечённого полного исходного дерева:

```bash
python -m pip install -r requirements.txt -r requirements-interop.txt
python -m pytest -q tests/test_interop_*.py tests/test_s14_anza_frontend.py tests/test_anza_segmentation_adapter.py
python scripts/research/bubble_chamber/example_external_detector.py
python scripts/research/bubble_chamber/example_verified_external_detector.py
python scripts/research/bubble_chamber/example_anza_to_graph_lira.py
```

Последний пример строит **исходную архитектуру AZUNet с НЕОБУЧЕННЫМИ весами** и пропускает её через переходник к Graph-LIRA. Он специально использует `strategy='none'`; нулевые соединения здесь — нормальное поведение, не научный результат. Результат можно смотреть только как проверку совместимости кода и формата.

## 3. Реальная оригинальная ANZA → Graph-LIRA (после подключения закрытых данных)

```python
from pathlib import Path
from bubble_chamber.interop_anza_frontend import OriginalAZUNetDetector, verified_anza_to_graph
from bubble_chamber.interop_calibration import CalibrationReceipt
from bubble_chamber.interop_coordinates import CropTransform

crop = CropTransform(x0=160, y0=160, full_width=512, full_height=512)
patch = crop.extract(your_private_512_gray_uint8)

# ONLY a trusted original source-locked checkpoint, kept outside Git:
anza_detector, provenance = OriginalAZUNetDetector.from_private_checkpoint(
    Path('/PRIVATE/ANZA_SOURCE_LOCKED_AZUNET.pt'))

# Pair scorer and calibration MUST be newly trained and frozen for this EXACT
# detector_id, checkpoint, crop/preprocessing and target-domain split.
receipt = CalibrationReceipt.from_file(Path('/PRIVATE/ANZA_PAIR_CAL_RECEIPT.json'))
result = verified_anza_to_graph(
    patch, anza_detector, receipt=receipt,
    model_file=Path('/PRIVATE/ANZA_PAIR_S56_SCORER.bin'),
    source_cal_file=Path('/PRIVATE/ANZA_PAIR_SOURCE_CAL.json'),
    score_pairs=loaded_detector_specific_pair_scorer,
    expected_training_domain='YOUR_EXPLICIT_FROZEN_ANZA_TRAIN_DOMAIN',
)
public_counts = result.as_public_dict()  # NO raw pixels or coordinates
# result.lines / result.joins are PRIVATE advisory machine hypotheses only.
```

`CalibrationReceipt` содержит SHA-256 геометрической модели и калибровки, а `OriginalAZUNetDetector` включает SHA-256 **реального AZUNet checkpoint** и конфигурации извлечения в `detector_id`. При изменении весов в памяти он также прекращает инференс. Нельзя переиспользовать scorer или порог S72 с AZUNet или другим детектором.

**Ограничение:** `CalibrationReceipt.validation_level` — декларация происхождения, а не независимый сертификат эксперта; для физического использования требуется внешняя валидация.

## 4. Как сравнивать с альтернативами

Сравнивайте на одинаковых сценах, сплитах и правилах оценки:

- **Обычный U-Net vs оригинальный AZUNet** на одних и тех же доверенных масках: метрики сегментации, фрагментация, число разрывов, precision и восстановление полных траекторий. Не подменять неизвестные пиксели маски отрицательными.
- **Hessian/ridge vs LSD vs Hough** как источники 2D-фрагментов: длины, покрытие и качество концов, ложные дополнительные кандидаты.
- **None / node-mutual / port-greedy / global-MILP** на одних и тех же полилиниях и scorer: отдельно правильные стыки, остальные/неоднозначные стыки, полные цепочки и время расчёта.
- **Отдельный TRAIN/CAL для каждого детектора**. Вся стратегия fail-closed и read-only до получения экспертной GT.

### Что уже подтвердили эксперименты

* Искусственные сцены без H50Z: в контрольном сравнении Hessian/ridge собрал **10/40** полных искусственных цепочек без размытия, LSD **4/40**, Hough **0/40**. Это **не** преимущество ANZA над U-Net: AZUNet в этом эксперименте не участвовал.
* После отдельного обучения геометрического scorer для каждого из трёх детекторов все три **не прошли** CAL-ограничение на ошибочные стыки и воздержались от соединений.
* Исторические S27–S36 сравнения **оригинальной ANZA и обычного U-Net** на искусственно вставленных треках поверх H50Z не доказывают реальную физическую точность; обычный U-Net нередко оставался конкурентоспособным.
* Заявленные в исходном README сейсмические AUROC **0.9923**, recovery **67.2%**, ложные соединения **0.78%** относятся к **другой задаче и историческому протоколу LIRA**, НЕ к физической реконструкции H50Z.

Подробности в `docs/bubble_chamber/FULL_ANZA_LIRA_HANDOFF_2026_10_28.md`, `DEVELOPER_INTEGRATION_2026_10_28.md`, отчётах S27–S85 и `S85_INTEROP_*.md`.

## 5. Что может сделать другая команда прямо сейчас

1. Подключить детектор к `Detector.detect(uint8[192,192]) -> list[ndarray[N,2]]` и получить **неизменяемые 2D-гипотезы**.
2. Запустить локальные тесты без исходных фотографий.
3. Обучить геометрический scorer на собственном разрешённом TRAIN и заморозить CAL для **конкретной** версии детектора.
4. На независимом TEST рассчитать не только верные соединения, но и ошибочные, полные цепочки, bootstrap по целым сценам.
5. Открыть приватный просмотрщик с реальными H50Z и дать физикам подтвердить ограниченный набор уже предложенных связей.

**NO-GO:** без независимых подтверждённых физических треков нельзя разрешать автоматическое соединение, заявлять точность реальной реконструкции, определять частицы, импульс или 3D.
