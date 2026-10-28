# Полная ANZA–LIRA: инженерная интеграция и границы научной применимости

**Версия:** 28.10.2026, ветка `research/bubble-chamber-graph-lira`.

## От исходной ANZA до Graph-LIRA

```text
                         PRIVATE 2D image
                                |
           +--------------------+----------------------+
           |                    |                      |
  original AZUNet         BaselineUNet           Hessian/LSD/Hough
  models.azconv +         models.segmentation     interop_detectors
  models.segmentation     (requires weights)     (classical)
           |                    |                      |
      binary mask            binary mask              |
           |                    |                      |
      skeletonization + open-curve fragmentation       |
           |                    |                      |
           +------------- ndarray[N,2], x/y ----------+
                                |
             detector-specific feature distribution
                                |
            TRAIN scorer for THIS detector and domain
                                |
                detector-specific frozen source CAL
                                |
                        strict S59 pair gate
                                |
              S56 geometric graph solver or no-join
                                |
                advisory 2D links: STATUS UNKNOWN
```

Оригинальные `anza_*` и `lira_*` пакеты в репозитории содержат самостоятельные исследования другой постановки, прежде всего тонких сейсмических структур. Код `bubble_chamber/s56_*`, `s59_*`, `s72_*` — отдельный, более поздний эксперимент по двумерной связности в пузырьковой камере, а не доказанная end-to-end замена всей исходной LIRA.

## Как не допустить скрытой подмены модели

- Исходная сеть AZUNet используется именно из `models/segmentation.py`, а AZConv из `models/azconv.py`. Проверка source-locked checkpoint через `s14_anza_frontend.load_original_az_checkpoint` требует совпадения Git blob hash оригинального кода.
- `OriginalAZUNetDetector.from_private_checkpoint()` формирует `detector_id` из SHA-256 checkpoint и всех параметров mask-to-polyline; сохранённый hash state_dict проверяется при каждом вызове.
- Веса исходного AZUNet и веса scorer пар — **два независимых объекта**, и оба должны соответствовать источнику TRAIN. `CalibrationReceipt` проверяет SHA-256 scorer-файла и CAL. Для сегментации detector_id меняется с checkpoint/конфигурацией.
- `process_verified` не сертифицирует физическую истинность. Независимая валидация должна проводиться вне этого интерфейса.
- `ScoreBinding(trained_for_detector=...)` не позволит тихо применить scorer другого детектора. `strategy='none'` — отказ от соединений, не фиктивная 100% точность.

## Допустимые типы доказательств

| Тип теста | Что проверяет | Чего НЕ подтверждает |
|---|---|---|
| `OriginalAZUNetDetector` на случайно инициализированной оригинальной сети | Импорт оригинального кода, форма тензоров, извлечение Nx2, переход в граф | Качество обнаружения на H50Z |
| Имитирующий маску `DarkStrokeMock` | Перестановка координат y/x → x/y, нулевые/многочисленные фрагменты, отказ без CAL | Качество ANZA |
| Реальные H50Z-фоны с инъецированными artificial film070 линиями (S27–S85) | Робастность pipeline к фотометрии/детекторным ошибкам на 2 фонах одного запуска | Правильность **реальных** физических соединений |
| Сгенерированные сцены без исходных H50Z (S85 interop) | Поменяемый детектор, сравнительные стратегии, TRAIN/CAL контроль | Превосходство AZUNet, перенос на физическую камеру |
| Отдельный физический запуск с expert-reviewed связями (пока не выполнен) | Может обосновать real-data precision/recall, перенос на запуск | 3D/PID без дополнительной физики |

## На какой базе лучше строить работу сейчас

1. **S72 Hessian/ridge** — честный и наиболее изученный baseline двухмерной реконструкции в данной ветке. В предыдущих H50Z-background экспериментах он сравнивался с добавочными S80–S84 линиями; безопасное улучшение не подтвердилось. В новом generated-only S85 сравнивался только с LSD/Hough, а не с обученным AZUNet.
2. **Оригинальный AZUNet** — исследовательский кандидат для front-end сегментации. Он может лучше моделировать анизотропию и непрерывность тонких линий, но без обученных/доступных checkpoint и native GT нельзя заявить, что он лучше классического ridge или обычного U-Net для реальных H50Z. В исторических S27–S36 на synthetic injection обычный U-Net конкурентоспособен.
3. **LSD/Hough** — прозрачные быстрые контрольные методы. На генераторе изогнутых линий проигрывали ridge; это не означает, что они хуже на всех изображениях или в задачах с преимущественно прямыми треками.
4. **Original LIRA seismic vs Graph-LIRA bubble** — не подставлять исторические сейсмические 0.9923/67.2%/0.78% в вывод по H50Z; это невалидный перенос между предметными областями.

## Предлагаемая схема корректного научного сравнения ANZA vs alternative

- **Split:** разные физические запуски/кадры (сейчас нет достаточных независимых размеченных H50Z), геометрические искусственные source-фреймы целиком разделены TRAIN/CAL/TEST.
- **Detector arms:** source-locked AZUNet, matched BaselineUNet, classical Hessian/ridge, LSD/Hough. Зафиксировать размеры сетей, seed, тренировочное время, количество положительных/отрицательных и UNKNOWN масок.
- **Pipeline arms:** detector-only, detector+`none`, detector+`node_mutual`, detector+`port_greedy`, detector+`global_milp`. Отдельный scorer/CAL для каждого детектора; сеть не должна видеть TEST labels.
- **Primary:** полные правильно соединённые цепочки при ограничении роста wrong/ambiguous-joins; secondary — длины фрагментов, endpoint error, coverage, throughput/latency и доля сценных воздержаний.
- **Uncertainty:** bootstrap по независимым сценам/физическим запускам, не по пикселям. На одном физическом H50Z запуске нельзя делать вывод о межзапусковой переносимости.
- **Acceptance:** FAIL-CLOSED, если ни один scorer не проходит CAL; сохранять S72 baseline. Нельзя ослабить порог исходя из TEST.

## Конкретное решение о передаче

- **GO** — код, конфигурации, адаптеры и обучение detector-specific моделей для исследований и приватного экспертного просмотра.
- **NO-GO** — использовать AZUNet или Graph-LIRA для автоматической корректировки настоящих физических траекторий, выдавать физическую accuracy или выполнять 3D/PID по одному кадру.
- **Что отсутствует** — source images H50Z 005/006 (39 частных crop-участков одного запуска), measured film070 geometry, обученные приватные checkpoint, экспертные связи на независимых физических запусках.

Источники внутри полного проекта: `README.md` (исходная seismic ANZA–LIRA), `README_INTEROP.md`, `docs/bubble_chamber/README.md`, `docs/bubble_chamber/S27_S29_ANZA_H50Z_ROBUST_TRACK_RESEARCH_2026_10_25.md`, `docs/bubble_chamber/S85_INTEROP_COMPARISON_RESULTS_2026_10_28.md`, `docs/bubble_chamber/S85_INTEROP_CALIBRATED_RESULT_2026_10_28.md`.
