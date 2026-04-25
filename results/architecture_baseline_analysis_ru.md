# Анализ архитектуры AZ и сравнение с baseline

Дата среза: 2026-04-25

## Короткий вывод

Математическая идея AZConv выглядит согласованной: локальная fuzzy-совместимость умножается на геометрическое anisotropic-kernel поле, затем агрегируются соседние признаки. Проблема текущих результатов не в том, что геометрия "математически неправильная", а в том, как этот prior встроен в U-Net и оптимизацию.

На текущих честных прогонах baseline остается очень сильным. После implementation-fix `az_thesis` уже близко к baseline и в одном DRIVE multiseed-пакете слегка выше по Dice, но делает это ценой меньшего recall и примерно 3x более медленного forward.

## Что показывают результаты

### Final pack 2026-04-24

| Dataset | Baseline Dice | AZ-Thesis Dice | Delta |
|---|---:|---:|---:|
| DRIVE | 0.7432 | 0.5949 | -0.1483 |
| CHASE-DB1 | 0.6725 | 0.6479 | -0.0246 |
| FIVES | 0.7502 | 0.7199 | -0.0302 |
| ARCADE Syntax | 0.6522 | 0.5963 | -0.0559 |
| ARCADE Stenosis | 0.3034 | 0.2075 | -0.0959 |

Вывод: старая версия `az_thesis` не обгоняла baseline ни на одном датасете.

### DRIVE implementation-fix, multiseed 41/42/43, 20 эпох

| Variant | Dice mean | Precision | Recall | Balanced Acc | Params | GMACs | Forward batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7456 | 0.7374 | 0.7540 | 0.8638 | 1.54M | 19.97 | 0.068s |
| az_thesis policyfix | 0.7498 | 0.7799 | 0.7220 | 0.8510 | 2.20M | 27.07 | 0.213s |
| az_thesis recallfloor | 0.7377 | 0.7388 | 0.7449 | 0.8589 | 2.20M | 27.07 | 0.224s |

Вывод: `policyfix` дает +0.0042 Dice к baseline, но ниже recall и balanced accuracy. Это слабое, но реальное улучшение Dice, не универсальная победа.

### CHASE-DB1 policyfix, multiseed 41/42/43, 20 эпох

| Variant | Dice mean | Precision | Recall | Balanced Acc | Forward batch |
|---|---:|---:|---:|---:|---:|
| baseline | 0.6303 | 0.6099 | 0.6547 | 0.8053 | 0.209s |
| az_thesis | 0.6277 | 0.5814 | 0.6829 | 0.8156 | 0.916s |

Вывод: на CHASE `az_thesis` почти равен baseline по Dice, выше recall/balanced accuracy, но ниже precision и намного медленнее.

## Что устроено правильно

1. `AZConv2d` строит совместимость:
   `compat = mu_center * mu_neighbor * kernel * valid_mask`.

2. В local hyperbolic mode геометрия строится симметрично по center/neighbor:
   направление усредняется через `cos(2 theta)` и `sin(2 theta)`, что корректно для orientation-field без различения theta и theta + pi.

3. Padding bug исправлен:
   `compatibility_floor` теперь добавляется только к реальным соседям через `valid_un`, а не к padded-пикселям.

4. `az_thesis` больше не pure-AZ везде. Лучшие прогоны используют hybrid encoder:
   обычная conv-ветка плюс AZ-ветка с `hybrid_mix_init ~= 0.10`, ASPP bottleneck и residual decoder.

## Что, вероятно, не так

1. Геометрия слишком легко становится фильтром precision, а не recall.
   На DRIVE policyfix AZ имеет precision 0.7799 против 0.7374 у baseline, но recall 0.7220 против 0.7540. Для тонких сосудов и дорог это означает пропуски слабых/тонких сегментов.

2. Global normalization может смывать локальный геометрический смысл.
   При `normalize_mode=global` все rules и все соседи конкурируют за одну сумму. Это стабилизирует обучение, но превращает геометрию в attention-like перераспределение массы. `per_rule` звучит математически чище, но текущие короткие пробы дали хуже Dice, значит нужна не простая замена, а более аккуратная нормализация.

3. AZ-ветка дорогая и медленная.
   На DRIVE forward batch примерно 0.213s против 0.068s у baseline. Если метрика выше только на 0.004 Dice, это пока слабый trade-off.

4. Слишком маленькая доля AZ фактически помогает.
   Лучшие policyfix-прогоны держат `hybrid_mix_alpha` около 0.10. Это говорит, что модель выигрывает от небольшой геометрической добавки, но не хочет отдавать основную работу AZ.

5. Старые final-pack результаты были до сильного implementation/policy fix.
   Поэтому их нельзя использовать как финальный приговор архитектуре, но они показывают риск: чистая/агрессивная AZ-архитектура ломает практическую сегментацию.

## Что проверять дальше

1. Для DRIVE: закрепить `drive_implfix_policyfix_ms_414243_e20` полным повтором на свежем коде и, если сохраняется +Dice, писать как "small Dice gain with precision-recall trade-off".

2. Для GIS roads: проверить, помогает ли AZ именно на дорогах. Это сильный тест идеи, потому что дороги похожи на сосуды как тонкие связные структуры, но имеют другую текстуру.

3. Добавить connectivity/topology метрики, а не только Dice/IoU. Если геометрия реально полезна, она может проявиться в связности дорог/сосудов, даже когда Dice близок.

4. Пробовать recall-friendly threshold selection и loss:
   ограничение `min_recall`, Tversky с большим beta, либо selection metric типа Dice + balanced accuracy.

5. Сравнивать не только `az_thesis` против baseline, а baseline + lightweight geometry regularizer, чтобы понять, нужна ли тяжелая AZConv или достаточно геометрического auxiliary loss.
