Да. Здесь **старый `STOP_BEFORE_TRAINING` снимаем**. Кодер правильно остановился по старому контракту, но теперь мы официально меняем научный протокол.

И важный момент: это изменение не «костыль из-за 40 масок». Сама статья CRACKS прямо предусматривает нетрадиционную постановку, где изображения могут использоваться при обучении с novice/practitioner-разметками, а затем на тех же сейсмических изображениях модель должна приблизить экспертную интерпретацию. Авторы также отдельно рассматривают ограниченное expert fine-tuning.  Причём их обычные SegFormer, U-Net, U-Net++ и DeepLabV3 на section-wise segmentation показывают довольно слабые результаты, и авторы прямо указывают на необходимость новых training paradigms.

Это нам подходит намного лучше, чем искусственно пытаться сделать `200/25/40`.

И одновременно я бы **сейчас окончательно менял научный центр ANZA-LIRA**. Просто «адаптивная вытянутая свёртка» недостаточна: deformable convolution уже обучает геометрию выборки, Dynamic Snake Convolution специально предназначена для тонких извилистых структур, а совместное обучение segmentation+orientation для линий тоже давно изучалось. ([CVPR Open Access][1]) Даже сохранение topology через clDice уже отдельная известная линия. ([arXiv][2])

**Наша следующая версия должна отвечать не только “куда вытянуть окно”, а “какую структурную ветвь продолжить через сложную локальную геометрию”.**

Ниже уже **единое полное ТЗ**, которое можно целиком отдать кодеру вместо старого CRACKS ТЗ.

---

# MASTER SPECIFICATION

# ANZA-LIRA v2

## Mode-Resolved Hyperbolic Fuzzy Transport for Seismic Fault Delineation

## 0. РЕШЕНИЕ ПО ТЕКУЩЕМУ BLOCKER

Текущий `BLOCKER_REPORT.md` не удалять.

Изменить статус blocker на:

```text
RESOLVED_BY_PROTOCOL_REVISION
```

Причина:

```text
The previous expert-only 200/25/40 protocol is invalid for the
actual released CRACKS archive.

The research protocol has been revised to use CRACKS according
to its intended crowd-to-expert learning setting and to introduce
a separate controlled structural benchmark for branch identity,
junction routing and trace continuity.
```

`40 expert masks` больше НЕ являются причиной остановки.

Продолжить работу автоматически.

---

# 1. ГЛАВНАЯ ЦЕЛЬ РАБОТЫ

Не ставить целью просто:

```text
increase Dice using another anisotropic convolution
```

Новая исследовательская задача:

> Разработать локальный режимно-разрешённый оператор, который переносит признаки вдоль отдельных направленных структур, не смешивает пересекающиеся ветви преждевременно, сохраняет несколько локальных гипотез в точках пересечения и позволяет восстанавливать структурную трассу разлома, а не только бинарную foreground-маску.

Исследование должно проверить четыре уровня:

```text
LEVEL 1
pixel segmentation

LEVEL 2
centerline / topology preservation

LEVEL 3
branch identity and junction continuation

LEVEL 4
model structural uncertainty vs human disagreement
```

---

# 2. ОСНОВНОЕ ОТЛИЧИЕ ОТ ANZA-LIRA v1

ANZA-LIRA v1 оставить неизменным baseline.

Не переписывать существующий `AZConv2d`.

В v1 локальные режимы в конечном итоге достаточно рано объединяются в один локальный отклик.

Это хорошо работает для:

```text
single elongated fragment
```

но принципиально ограничено в ситуации:

```text
\ /
 X
/ \
```

Проблема не только в обнаружении двух направлений.

Главный вопрос:

```text
какая входящая ветвь должна продолжиться какой исходящей ветвью?
```

ANZA-LIRA v2 должна сохранять mode identity внутри нескольких последовательных локальных преобразований.

---

# 3. НАУЧНОЕ НАЗВАНИЕ ВНУТРИ КОДА

Рабочее:

```text
ANZA-LIRA v2
Mode-Resolved Hyperbolic Fuzzy Transport
```

Класс слоя:

```text
AZConv2dV2
```

или:

```text
ModeResolvedAZConv2d
```

Не использовать в названии:

```text
Anosov convolution
ergodic convolution
ergodic ANZA
```

---

# 4. ЧТО ИМЕННО БЕРЁМ ОТ ГИПЕРБОЛИЧЕСКОЙ / «СЕДЛОВОЙ» ИДЕИ

Для режима (r) в точке (p):

[
u_r(p)=
\begin{bmatrix}
\cos\theta_r(p)\
\sin\theta_r(p)
\end{bmatrix},
]

[
s_r(p)=
\begin{bmatrix}
-\sin\theta_r(p)\
\cos\theta_r(p)
\end{bmatrix}.
]

Ввести локальную гиперболическую матрицу:

[
H_r(p)
======

R(\theta_r)
\begin{bmatrix}
e^{h_r(p)}&0\
0&e^{-h_r(p)}
\end{bmatrix}
R(\theta_r)^T.
]

Ключевое свойство:

[
\det H_r(p)=1.
]

То есть локальный масштаб растягивается вдоль одной оси и компенсирующе сжимается вдоль сопряжённой.

Это **saddle-inspired local geometry**.

Это НЕ доказательство:

```text
Anosov dynamics
ergodicity
global measure preservation
```

---

# 5. ЛОКАЛЬНЫЕ МАСШТАБЫ

Сохраняем существующую идею:

[
\sigma_{u,r}(p)=b_r e^{h_r(p)},
]

[
\sigma_{s,r}(p)=b_r e^{-h_r(p)}.
]

Следовательно:

[
\sigma_{u,r}\sigma_{s,r}=b_r^2.
]

Геометрическая часть:

[
G_r(p,q)
 ========

\exp
\left(
-\frac{
\langle q-p,u_r(p)\rangle^2
}{
2\sigma_{u,r}^2
}
-

\frac{
\langle q-p,s_r(p)\rangle^2
}{
2\sigma_{s,r}^2
}
\right).
]

Не ломать рабочую реализацию v1 ради косметической идентичности формул.

---

# 6. FUZZY MODES

Пусть mode head выдаёт logits (a_r(p)).

Принадлежности:

[
\mu_r(p)
 ========

\frac{\exp a_r(p)}
{\sum_k\exp a_k(p)}.
]

Требовать:

[
\mu_r(p)\ge0,
]

[
\sum_r\mu_r(p)=1.
]

Проверить unit tests.

---

# 7. ГЛАВНОЕ ИЗМЕНЕНИЕ V2: НЕ СУММИРОВАТЬ MODES СРАЗУ

Пусть входной feature vector:

[
x(p)\in\mathbb R^C.
]

Вместо:

[
x
\rightarrow
\sum_r \text{local response}_r
]

строим отдельные mode states:

[
z_r^{(0)}(p)
 ============

\mu_r(p)W_0x(p).
]

И сохраняем:

[
z_1,z_2,\ldots,z_R
]

через несколько v2 blocks.

Объединение modes выполнять только позднее:

```text
end of AZ stack
or
decoder fusion boundary
```

---

# 8. V2A: MODE-RESOLVED AXIAL TRANSPORT

Сначала реализовать минимальную новую идею без half-directions.

Переход:

[
T_{r\leftarrow s}(p,q).
]

Он означает:

> насколько режим (s) у источника (q) совместим с режимом (r) у получателя (p).

---

# 9. AXIAL ORIENTATION DISTANCE

Так как fault orientation является осью, а не стрелкой:

[
\theta\equiv\theta+\pi.
]

Использовать:

[
d_\pi(\theta_1,\theta_2)
 ========================

\frac12
\arccos
\left[
\cos 2(\theta_1-\theta_2)
\right].
]

Unit tests:

```text
d(theta, theta) = 0
d(theta, theta + pi) = 0
d(0, pi/2) = pi/2
```

---

# 10. ORIENTATION COMPATIBILITY

Ввести:

[
C^\theta_{rs}(p,q)
 ==================

\exp
\left[
-\kappa_\theta
\sin^2
\left(
\theta_r(p)-\theta_s(q)
\right)
\right].
]

Свойства:

```text
same axis        → close to 1
opposite axis    → close to 1
orthogonal axis  → strongly suppressed
```

Это должно быть отдельным тестом.

---

# 11. SYMMETRIC LOCAL GEOMETRY

Для связи двух modes использовать:

[
G_{rs}(p,q)
 ===========

\sqrt{
G_r(p,q)
G_s(q,p)
}.
]

Это не обязательно единственно возможный вариант, но primary implementation должен быть именно таким.

Не менять после просмотра test results.

---

# 12. MODE TRANSITION SCORE V2A

[
\widetilde T_{r\leftarrow s}(p,q)
 =================================

\mu_r(p)
\mu_s(q)
G_{rs}(p,q)
C^\theta_{rs}(p,q).
]

Нормировать:

[
\alpha_{r\leftarrow s}(p,q)
 ===========================

\frac{
\widetilde T_{r\leftarrow s}(p,q)
}{
\varepsilon+
\sum_{q'\in\Omega(p)}
\sum_{s'}
\widetilde T_{r\leftarrow s'}(p,q')
}.
]

Обновление:

[
z_r^{(l+1)}(p)
 ==============

\phi
\left[
W_{\mathrm{self}}z_r^{(l)}(p)
+
\sum_{q,s}
\alpha_{r\leftarrow s}(p,q)
W_{\mathrm{msg}}z_s^{(l)}(q)
\right].
]

Использовать residual connection, если это согласуется с существующим AZ block.

---

# 13. V2B: HALF-MODE DIRECTIONAL TRANSPORT

После PASS v2a добавить directed state.

Для каждой оси:

[
t_{r,+}=u_r,
]

[
t_{r,-}=-u_r.
]

Состояния:

[
z_{r,+},
\qquad
z_{r,-}.
]

Теперь различаются:

```text
axis identity
```

и:

```text
local travel direction
```

---

# 14. SOURCE → DESTINATION DIRECTION

Для сообщения:

[
q\rightarrow p
]

определить:

[
d_{q\rightarrow p}
 ==================

\frac{p-q}
{|p-q|+\varepsilon}.
]

Для source half-mode:

[
C^{src}_{s,\xi}(q,p)
 ====================

\exp
\left[
-\kappa_d
\left(
1-
\langle
d_{q\rightarrow p},
t_{s,\xi}(q)
\rangle
\right)^2
\right].
]

Для destination:

[
C^{dst}_{r,\eta}(p,q)
 =====================

\exp
\left[
-\kappa_d
\left(
1-
\langle
d_{q\rightarrow p},
t_{r,\eta}(p)
\rangle
\right)^2
\right].
]

---

# 15. FULL DIRECTIONAL TRANSITION

[
\widetilde T_
{r,\eta\leftarrow s,\xi}
(p,q)
=====

\mu_r(p)
\mu_s(q)
G_{rs}(p,q)
C^\theta_{rs}(p,q)
C^{src}*{s,\xi}(q,p)
C^{dst}*{r,\eta}(p,q).
]

Дополнительно разрешается проверить:

[
C^h_{rs}(p,q)
 =============

\exp
\left(
-\kappa_h
|h_r(p)-h_s(q)|
\right),
]

но это отдельная ablation.

Не включать сразу в full model без проверки.

---

# 16. TRANSITION НОРМИРОВКА

Предпочтительная интерпретация — local transport probabilities.

Для source state:

[
P
\left(
p,r,\eta
\mid
q,s,\xi
\right)
 =======

\frac{
\widetilde T_
{r,\eta\leftarrow s,\xi}
(p,q)
}{
\varepsilon+
\sum_{p',r',\eta'}
\widetilde T_
{r',\eta'\leftarrow s,\xi}
(p',q)
}.
]

Требовать:

[
P\ge0,
]

и:

[
\sum_{p,r,\eta}
P(p,r,\eta\mid q,s,\xi)
\approx1.
]

Это **row-stochastic local transport**.

Не называть это доказательством эргодичности.

---

# 17. MODE MESSAGE UPDATE

[
z^{(l+1)}_{r,\eta}(p)
 =====================

\phi
\left[
W_{\mathrm{self}}z^{(l)}*{r,\eta}(p)
+
\sum*{q,s,\xi}
P(p,r,\eta\mid q,s,\xi)
W_{\mathrm{msg}}
z^{(l)}_{s,\xi}(q)
\right].
]

Не делать отдельную полную (C\times C) матрицу для каждого (r,s,\eta,\xi).

Это взорвёт параметры.

Использовать shared или low-rank message projection.

---

# 18. MODE FUSION

В segmentation head:

[
z_{\mathrm{fused}}(p)
 =====================

\sum_{r,\eta}
\pi_{r,\eta}(p)
z_{r,\eta}(p).
]

(\pi) должна быть нормированной.

Primary:

[
\pi_{r,+}
 =========

# \pi_{r,-}

\frac12\mu_r.
]

Learned fusion допускается только отдельной ablation.

---

# 19. JUNCTION SCORE ИЗ САМОЙ ANZA-GEOMETRY

Не добавлять отдельный CNN junction detector как главный механизм.

Нормированные modes:

[
\bar\mu_r
 =========

\frac{\mu_r}
{\sum_k\mu_k+\varepsilon}.
]

Mode diversity:

[
D(p)
====

\frac{
1-\sum_r\bar\mu_r(p)^2
}{
1-1/R
}.
]

Ограничить:

[
D\in[0,1].
]

---

# 20. ANGULAR DIVERSITY

[
A(p)
====

\frac{
\sum_{r<s}
\bar\mu_r
\bar\mu_s
\sin^2(\theta_r-\theta_s)
}{
\sum_{r<s}
\bar\mu_r
\bar\mu_s+\varepsilon
}.
]

Итого:

[
J(p)=D(p)A(p).
]

Требовать:

[
0\le J(p)\le1.
]

---

# 21. ИНТЕРПРЕТАЦИЯ JUNCTION SCORE

Тесты:

```text
one dominant mode
→ J low

two strong parallel modes
→ J low

two strong orthogonal modes
→ J high

three separated orientations
→ J high
```

Это одна из центральных проверок.

---

# 22. HYPERBOLIC CONE FIELD

Для mode (r):

[
\mathcal C_r(p)
 ===============

\left{
v:
d_\pi(v,u_r(p))
\le
\alpha(p)
\right}.
]

Cone half-angle сделать junction-aware:

[
\alpha(p)
 =========

\alpha_0
+
(\alpha_J-\alpha_0)J(p).
]

То есть:

```text
straight fragment
→ narrow allowed continuation

junction
→ wider set of admissible continuations
```

---

# 23. SOFT CONE CONSISTENCY

Для mode (r) в (p) и modes соседней точки (q):

[
c_r(p,q)
 ========

\sum_s
\bar\mu_s(q)
\exp
\left[
-\kappa_c
\left(
\max
\left(
0,
d_\pi(\theta_r(p),\theta_s(q))
-\alpha(p)
\right)
\right)^2
\right].
]

Loss:

[
L_{\mathrm{cone}}
 =================

*

\frac{
\sum_{p,q,r}
w_{pqr}
\log(c_r(p,q)+\varepsilon)
}{
\sum_{p,q,r}w_{pqr}+\varepsilon
}.
]

Вес:

[
w_{pqr}
 =======

P_f(p)
P_f(q)
\bar\mu_r(p)
G_r(p,q).
]

---

# 24. ROUTING ENTROPY

Для каждого source state с переходами (P_j):

[
H_T
===

*

\frac{
\sum_j P_j\log(P_j+\varepsilon)
}{
\log N
}.
]

Требовать:

[
0\le H_T\le1.
]

Интерпретация:

```text
H_T low
→ one dominant structural continuation

H_T high
→ several competing continuations
```

Не использовать это сразу как loss на CRACKS.

Сначала diagnostic.

---

# 25. НОВЫЙ CONTROLLED BENCHMARK

# CROSSINGTRACEBENCH

Это ОБЯЗАТЕЛЬНАЯ часть работы.

Реальный CRACKS не должен использоваться для доказательства instance identity, если released masks не содержат достоверный fault-instance ID.

Поэтому создать controlled procedural benchmark.

Файлы:

```text
synthetic/
    crossing_trace_bench.py
    seismic_background.py
    geometry_generator.py
    instance_targets.py
```

---

# 26. BENCHMARK НЕ ДОЛЖЕН БЫТЬ «БЕЛЫЕ ЛИНИИ НА ЧЁРНОМ»

Нужно приблизить задачу к seismic geometry.

Базовый layered signal:

[
I_0(x,y)
 ========

\sum_{m=1}^{M}
a_m
\sin
\left[
2\pi f_m y+\phi_m(x)
\right].
]

Разрешается использовать Gaussian-layer representation вместо sine, если она визуально ближе к seismic sections.

---

# 27. SYNTHETIC FAULT THROW

Для fault (i) со signed-distance field (s_i(x,y)):

[
\Delta y_i(x,y)
 ===============

\frac{\Delta_i}{2}
\tanh
\left(
\frac{s_i(x,y)}
{\tau_i}
\right).
]

Совокупное warped coordinate:

[
y'
==

y+
\sum_i\Delta y_i(x,y).
]

Изображение:

[
I(x,y)
======

I_0(x,y')
+
\epsilon(x,y).
]

Это создаёт displacement/discontinuity структуры слоёв около fault.

Synthetic generator не должен заявляться как физическая модель F3.

Это controlled structural benchmark.

---

# 28. GEOMETRY TYPES

Генерировать:

```text
single straight fault
curved fault
two crossing faults
X junction
T junction
Y junction
near-parallel faults
close non-intersecting faults
fault with gap
two faults with different throws
curved crossing
short distractor
multiple minor branches
```

---

# 29. SYNTHETIC DATA SIZE

Генерировать on the fly.

Не сохранять тысячи PNG.

Fixed seeds:

```text
train: 10000 samples
validation: 2000
test: 2000
```

Размер:

```text
128x128
```

Если GPU throughput позволяет, 192×192 разрешён, но не менять после начала сравнений.

---

# 30. SYNTHETIC TARGETS

Для каждого sample автоматически доступны:

```text
binary fault mask
fault instance map
centerline map
tangent orientation map
junction map
endpoint map
branch IDs
branch pairing at every junction
fault instance IDs
gap mask
```

Это exact ground truth.

---

# 31. INSTANCE ID НЕ МЕНЯЕТСЯ ПОСЛЕ ПЕРЕСЕЧЕНИЯ

Пример:

```text
fault A enters junction
fault A exits on its geometric continuation
```

GT должен хранить это явно.

Именно это позволяет проверить branch identity.

---

# 32. STRUCTURAL METRICS

Обязательные:

```text
Dice
IoU
clDice
skeleton F1
orientation error
junction F1
endpoint F1
branch-pairing accuracy
false-merge rate
false-split rate
identity-switch rate
gap-recovery rate
fragmentation
```

---

# 33. BRANCH-PAIRING ACCURACY

В junction имеется множество incident branches:

[
B={b_1,\ldots,b_K}.
]

GT задаёт relation:

[
\Pi^*(b_i,b_j)=1
]

если ветви принадлежат одному fault instance через junction.

Из transition scores построить predicted continuation.

Метрика:

[
Acc_{\mathrm{pair}}
 ===================

\frac{
N_{\mathrm{correct\ pairings}}
}{
N_{\mathrm{GT\ pairings}}
}.
]

---

# 34. FALSE MERGE

Predicted trace component считается merged, если он существенно покрывает больше одного GT instance.

Использовать фиксированный overlap threshold:

```text
20%
```

Он фиксируется до test.

[
FMR
===

\frac{
N_{\mathrm{merged\ predicted\ traces}}
}{
N_{\mathrm{predicted\ traces}}
}.
]

---

# 35. FALSE SPLIT

Для GT fault (i) считать число значимых predicted components:

[
n_i.
]

[
FS_i=\max(0,n_i-1).
]

[
FSR
===

\frac1N
\sum_i FS_i.
]

---

# 36. IDENTITY SWITCH

Для GT fault, проходящего через junction, проверить predicted trace identity до и после junction.

[
IDSW
====

\frac{
N_{\mathrm{wrong\ continuation}}
}{
N_{\mathrm{junction\ continuations}}
}.
]

Это одна из центральных метрик v2.

---

# 37. SYNTHETIC LOSSES

Segmentation:

[
L_{\mathrm{seg}}
 ================

L_{\mathrm{BCE}}
+
\lambda_D L_{\mathrm{Dice}}.
]

Topology:

[
L_{\mathrm{top}}
 ================

L_{\mathrm{clDice}}.
]

Orientation:

[
L_{\theta}
 ==========

\frac{
\sum_p w(p)
\left[
1-\cos2(\theta(p)-\theta^*(p))
\right]
}{
\sum_pw(p)+\varepsilon
}.
]

Junction:

[
L_J
===

BCE(J,J^*).
]

---

# 38. ROUTING SUPERVISION

На synthetic junctions имеется GT continuation destination (j^*).

[
L_{\mathrm{route}}
 ==================

-\log
P(j^*\mid source).
]

Это крайне важная часть.

Она напрямую учит:

```text
не просто найти линии,
а сохранить identity через пересечение
```

---

# 39. FULL SYNTHETIC OBJECTIVE

[
L_{\mathrm{synthetic}}
 ======================

L_{\mathrm{seg}}
+
\lambda_TL_{\mathrm{top}}
+
\lambda_\theta L_\theta
+
\lambda_JL_J
+
\lambda_RL_{\mathrm{route}}
+
\lambda_CL_{\mathrm{cone}}.
]

Не тюнить пять (\lambda) независимо огромным grid.

Использовать bounded candidates.

---

# 40. INCREMENTAL IMPLEMENTATION

Не писать сразу full V2.

Строгая последовательность:

```text
V1
↓
V2A mode-resolved axial
↓
V2B half-mode directional
↓
junction score
↓
routing supervision
↓
cone consistency
↓
structural replay
```

После каждого:

```text
unit test
synthetic smoke
targeted comparison
self-review
```

---

# 41. BASELINES НА SYNTHETIC

Обязательные:

```text
U-Net baseline
ANZA-LIRA v1
ANZA-LIRA v2
Deformable U-Net
```

Deformable U-Net использовать через стабильную существующую PyTorch/torchvision реализацию, если она доступна в текущем окружении.

Не создавать новое virtual environment.

---

# 42. DYNAMIC SNAKE

DSC / Dynamic Snake baseline желателен, потому что этот класс методов специально адаптирует receptive geometry к тонким извилистым структурам. ([arXiv][3])

Но:

```text
не копировать случайную стороннюю реализацию,
не добавлять конфликтующий CUDA stack.
```

Если официальная/надёжная реализация интегрируется изолированно без разрушения environment:

```text
include dsc_unet
```

Иначе:

```text
record NOT_INCLUDED_DEPENDENCY_SCOPE
```

и не блокировать основную работу.

---

# 43. ЗАЧЕМ DEFORMABLE BASELINE

Deformable convolution уже учит spatial offsets и тем самым является обязательным сравнением с утверждением «наша геометрия адаптивна». ([CVPR Open Access][1])

Поэтому `deformable_unet` ОБЯЗАТЕЛЕН.

Сравнение должно отвечать:

> Даёт ли явное mode identity + structural transport что-то сверх свободного изменения sampling geometry?

---

# 44. SYNTHETIC DEVELOPMENT BUDGET

Не бесконечно тюнить до победы.

Первичная последовательность candidates:

```text
C0: ANZA v1

C1:
V2A
R=4
axial compatibility

C2:
V2B
R=4
half-mode direction

C3:
V2B
+ junction score
+ route supervision

C4:
C3
+ cone consistency

C5:
C4
+ synthetic structural replay setup
```

Если конкретные числовые (\kappa) требуют выбора:

начальный ограниченный набор:

```text
kappa_theta: {2, 4, 8}
kappa_d:     {2, 4, 8}
```

Но не полный Cartesian grid.

Максимум 6 development configurations total после C0.

---

# 45. SYNTHETIC QUALITY GATE

До CRACKS v2 должна:

1. не иметь catastrophic segmentation degradation;
2. улучшать именно structural identity.

Desired gate:

```text
branch-pairing accuracy:
>= +5 percentage points over V1
OR

false-merge rate:
>= 15% relative reduction vs V1
OR

identity-switch rate:
>= 15% relative reduction vs V1
```

и одновременно:

```text
clDice must not decrease by > 0.01 absolute
```

Оценить на synthetic validation.

После выбора architecture test открыть один раз.

---

# 46. ЕСЛИ QUALITY GATE НЕ ДОСТИГНУТ

Не останавливаться после первой неудачи.

Выполнить:

```text
failure localization
routing visualization
junction subset metrics
straight-line subset metrics
crossing subset metrics
```

Понять, где ломается механизм.

Разрешено максимум:

```text
6 predeclared architecture candidates
```

После этого:

```text
freeze best candidate
```

и продолжить real study даже при отрицательном результате.

Не подглядывать в synthetic test для настройки.

---

# 47. CRACKS: НОВЫЙ НАУЧНЫЙ ПРОТОКОЛ

Старый:

```text
expert-only
200 / 25 / 40
```

УДАЛИТЬ ИЗ ACTIVE SPEC.

Фактические released data:

```text
396 images
12603 annotation masks
35 annotators
40 available expert masks
```

являются source of truth для текущего эксперимента.

---

# 48. ЗАЧЕМ МОЖНО ИСПОЛЬЗОВАТЬ CROWD LABELS

CRACKS специально создавался для использования novice/practitioner annotations как noisy approximations экспертной интерпретации. Авторы прямо пишут, что те же seismic images могут использоваться с non-expert labels при обучении для последующего приближения expert labels.

Поэтому наше обучение на crowd annotations не является leakage в рамках этого primary task.

Но результат нужно правильно называть:

```text
crowd-to-expert annotation transfer
```

а НЕ:

```text
generalization to unseen seismic images
```

---

# 49. СНАЧАЛА РАЗОБРАТЬСЯ С 40 EXPERT MASKS

Кодер уже нашёл 40.

Теперь необходимо проверить:

```text
their exact section IDs
whether they correspond to the 40 expert-training sections
described by the paper
whether another released directory contains expert test masks
whether test expert masks are absent from the public archive
```

Не угадывать.

Создать:

```text
results/cracks_study/expert_availability_audit.json
```

---

# 50. ЕСЛИ ДРУГИХ EXPERT MASKS НЕТ

Продолжать.

Не считать это blocker.

Назвать:

```text
available_expert_subset = 40 sections
```

И использовать новый protocol ниже.

---

# 51. LABEL SEMANTICS

Из первичного источника известны:

```text
blue   = fault certain
green  = fault uncertain
orange = no fault certain
```

В оригинальной основной серии экспериментов авторы не использовали orange no-fault labels и объединяли certain+uncertain fault; в дополнительном label-fusion анализе certain annotations получали в 1.5 раза больший вес, а practitioner labels в fusion получили больший вес, чем novice labels.

Это использовать как reference, а не придумывать свои значения с нуля.

---

# 52. WHITE 51.71% НЕЛЬЗЯ МОЛЧА НАЗНАЧАТЬ КЛАССОМ

Текущий audit обнаружил white.

Проверить:

```text
palette index
alpha
official preprocessing code
mask creation code
relation to untouched/unannotated canvas
```

Создать:

```text
MASK_SEMANTICS.md
```

Для каждого RGB:

```text
observed count
official meaning
evidence
training meaning
evaluation meaning
```

---

# 53. WHITE FALLBACK

Если официальный released code однозначно показывает:

```text
white = unannotated/background
```

использовать это.

Если это нельзя доказать:

не останавливать всю разработку.

Создать две явно названные policies:

```text
paper_like
conservative
```

И проверить sensitivity.

Ни одна неизвестная кодировка не должна молча превращаться в ground truth.

---

# 54. PRIMARY CROWD TARGET

Для annotator (a) и пикселя (p):

[
y_a(p)=
\begin{cases}
1,&\text{certain fault},\
1,&\text{uncertain fault},\
0,&\text{resolved background},\
\text{ignore},&\text{unknown / excluded}.
\end{cases}
]

---

# 55. CONFIDENCE WEIGHT

Source-compatible secondary weighting:

[
c_a(p)=
\begin{cases}
1.5,&\text{fault certain},\
1.0,&\text{fault uncertain}.
\end{cases}
]

Orange policy отдельно определяется mask semantics contract.

---

# 56. EXPERTISE WEIGHT

Использовать source-inspired weighting:

[
e_a=
\begin{cases}
2,&\text{practitioner},\
1,&\text{novice}.
\end{cases}
]

Это не наша научная новизна.

Одинаково применять ко всем architectures.

---

# 57. CROWD SOFT TARGET

Для valid annotators:

[
w_a(p)=e_a c_a(p).
]

[
q(p)
====

\frac{
\sum_a w_a(p)y_a(p)
}{
\sum_aw_a(p)+\varepsilon
}.
]

Также:

[
n_{\mathrm{eff}}(p)
 ===================

\sum_a\mathbf1[a\text{ valid at }p].
]

---

# 58. CROWD DISAGREEMENT

Для categorical annotation distribution:

[
H_{\mathrm{human}}(p)
 =====================

*

\frac{
\sum_c p_c(p)\log(p_c(p)+\varepsilon)
}{
\log K
}.
]

Только если:

```text
at least 5 valid annotators
```

Иначе:

```text
insufficient_annotation_support
```

---

# 59. PRIMARY CRACKS SETTING A

# CROWD-ONLY → EXPERT

Это основной real-data benchmark.

Обучение:

```text
ONLY novice/practitioner labels
NO expert gradient
```

Architecture/hyperparameters должны быть выбраны:

```text
synthetic validation
+
non-expert validation
```

До открытия expert scores.

После freeze:

```text
evaluate against all 40 available expert masks
```

Если crowd labels для этих же images использовались при training, это разрешено именно для этой постановки.

В отчёте писать:

> crowd-to-expert reconstruction on the same seismic sections

---

# 60. НЕ НАЗЫВАТЬ SETTING A IMAGE GENERALIZATION

Запрещено:

```text
generalizes to unseen sections
```

если images этих expert sections использовались с crowd labels.

---

# 61. CROWD VALIDATION БЕЗ EXPERT

Разделить annotators, а не expert masks.

Выбрать deterministic held-out annotators:

```text
at least 1 practitioner
at least 2 novices
```

не используемых при crowd target training.

Выбор по стабильному hash от annotator ID.

Проверить coverage.

Если выбранный annotator имеет недостаточное покрытие:

перейти к следующему hash candidate.

Не выбирать по model performance.

---

# 62. SETTING B

# LIMITED EXPERT FINE-TUNING

После завершения Setting A.

Использовать 40 available expert sections в cross-validation.

Primary:

```text
5 folds × 8 expert sections
```

В каждом fold:

```text
28 expert train
4 expert validation
8 expert test
```

Разбиение deterministic по numeric section ID / predefined fold rule.

Не менять после результатов.

---

# 63. FINE-TUNING

Каждый model начинает с crowd-only checkpoint.

Затем expert fine-tuning:

```text
small LR
max 20 epochs
early stop on expert validation
```

Одинаковый protocol для всех architectures.

---

# 64. SETTING C

# IMAGE-DISJOINT ROBUSTNESS

Это secondary stronger test.

Минимум для:

```text
U-Net
ANZA v1
ANZA v2 full
```

Seed:

```text
42
```

Для held-out expert fold:

исключить из training:

```text
image
all crowd annotations for this section
expert label
```

Также по возможности:

```text
±2 neighboring section IDs
```

как guard.

Этот setting проверяет именно:

```text
unseen-section generalization
```

---

# 65. НЕ ПУТАТЬ ТРИ SETTING

В таблицах:

```text
A: crowd-to-expert same-image transfer
B: crowd pretraining + expert fine-tuning
C: image-disjoint robustness
```

Никогда не смешивать их метрики без маркировки.

---

# 66. REAL-DATA TRAINING

Images не растягивать в square.

Actual:

```text
255 x 701
```

Padding:

```text
256 x 704
```

Metrics после unpad.

---

# 67. TRAINING CROPS

Для memory efficiency:

```text
256 x 256
```

Но test/evaluation на полном section.

Foreground-aware crop sampling:

```text
70% fault-aware
30% random
```

Использовать crowd fused target только из training annotations.

---

# 68. FULL-SECTION INFERENCE

На validation/test:

```text
full 256x704
```

если помещается.

Если нет:

```text
overlap tiled inference
overlap >= 64 px
weighted blending
```

После:

```text
unpad to 255x701
```

---

# 69. REAL-DATA LOSS

Для всех models одинаковый:

[
L_{\mathrm{real}}
 =================

L_{\mathrm{seg}}
+
\lambda_TL_{\mathrm{clDice}}.
]

Не включать synthetic routing ground truth на CRACKS, которого там нет.

---

# 70. STRUCTURAL REPLAY

Чтобы v2 при real fine-tuning не забыл junction routing, разрешить mixed training.

Например:

```text
3 real CRACKS batches
1 CrossingTraceBench batch
```

Real batch:

```text
segmentation/topology losses
```

Synthetic batch:

```text
full structural objective
```

Назвать:

```text
structural replay
```

---

# 71. STRUCTURAL REPLAY — ABLATION

Обязательно сравнить:

```text
v2 without replay
v2 with replay
```

Иначе нельзя утверждать, что controlled structural supervision реально влияет на real task.

---

# 72. MAIN MODEL MATRIX

На controlled benchmark:

```text
U-Net
Deformable U-Net
ANZA v1
ANZA v2a
ANZA v2b
ANZA v2 full
```

На CRACKS primary:

```text
U-Net
Deformable U-Net
ANZA v1
ANZA v2 full
```

Дополнительные ablations:

```text
v2 no fuzzy
v2 no directional half-mode
v2 no junction
v2 no cone
v2 no structural replay
```

---

# 73. НЕ ЗАПУСКАТЬ ВСЁ СРАЗУ

Сначала seed 42:

```text
U-Net
Deformable
V1
V2 full
```

После sanity PASS:

```text
41
42
43
```

для основных четырёх.

Ablations seed 42.

---

# 74. ОСНОВНЫЕ CRACKS METRICS

Не раздувать список бесконечно.

Primary table:

```text
Dice
IoU
clDice
skeleton F1 @ 2px
fragmentation
orientation median error
```

Это шесть.

---

# 75. SECONDARY METRICS

Хранить:

```text
precision
recall
Hausdorff / symmetric distance
endpoint F1
junction F1
skeleton precision
skeleton recall
```

Но не тащить все в главный тезис.

---

# 76. REAL ORIENTATION REFERENCE

Из expert skeleton:

локальная PCA / tangent estimation.

Window:

```text
radius = 5 px primary
3 and 7 sensitivity
```

Axial error:

[
E_\theta
 ========

d_\pi(\theta_{\mathrm{pred}},\theta_{\mathrm{GT}}).
]

---

# 77. REAL TRACE EXTRACTION

Reuse существующий:

```text
mask
→ skeleton
→ graph
→ trace branches
```

Не называть branch:

```text
unique geological fault instance
```

если released annotations не дают instance ID.

Использовать:

```text
candidate fault trace
fault trace branch
```

---

# 78. FRAGMENTATION

Для GT skeleton component (i):

пусть (n_i) — число predicted components, пересекающих 2px dilation GT component.

[
F_{\mathrm{frag}}
 =================

\frac1N
\sum_i
\max(0,n_i-1).
]

Lower is better.

---

# 79. HUMAN BASELINE

На всех available expert sections:

для каждого novice/practitioner, имеющего annotation:

вычислить тем же evaluator:

```text
Dice vs expert
clDice vs expert
skeleton F1 vs expert
orientation agreement if meaningful
```

Получить distributions:

```text
novice
practitioner
model
```

---

# 80. КОРРЕКТНАЯ ФОРМУЛИРОВКА «ЛУЧШЕ ЧЕЛОВЕКА»

Никогда:

```text
AI is better than humans
```

Даже если число выше.

Разрешено:

> The model showed higher agreement with the available expert annotation than the median novice/practitioner annotation under the same metric.

Только если реально подтверждено.

---

# 81. MODEL UNCERTAINTY VS HUMAN DISAGREEMENT

На expert sections сравнить:

[
H_{\mathrm{human}}
]

с:

[
H_T
]

и:

[
1-\rho.
]

Также:

```text
segmentation error
junction score J
anisotropy strength A
```

---

# 82. НЕ ИСПОЛЬЗОВАТЬ ПИКСЕЛИ КАК НЕЗАВИСИМЫЕ N

Основной correlation analysis:

агрегировать по section.

Для каждого section:

```text
mean human disagreement
mean routing entropy
mean 1-rho
Dice
clDice
error rate
```

Spearman.

Bootstrap by section.

---

# 83. ОСНОВНОЙ HUMAN-DISAGREEMENT ВОПРОС

Проверить:

> Повышается ли внутренняя структурная неоднозначность ANZA-LIRA v2 в тех же областях, где расходятся человеческие интерпретации?

Не заявлять заранее, что повышается.

---

# 84. АНОСОВ / ЭРГОДИЧНОСТЬ

Строго запрещено делать из текущей работы утверждение:

```text
AZConv is an Anosov map
AZConv is ergodic
```

Что можно:

> The local geometry uses a determinant-one paired expansion/contraction parameterization inspired by hyperbolic stable/unstable directional splitting.

И отдельно:

> Cone consistency is used to maintain locally admissible continuation directions.

---

# 85. МАТЕМАТИЧЕСКИЕ LIMIT CASES

Обязательно проверить экспериментально/unit-tests.

### Isotropic

[
h_r=0
]

даёт:

[
\sigma_u=\sigma_s.
]

### No fuzzy differentiation

[
\mu_r=\text{const}.
]

### No orientation routing

[
\kappa_\theta=0.
]

### No directional routing

[
\kappa_d=0.
]

### One mode

[
R=1
]

должен приближать обычную directional local aggregation.

---

# 86. COMPLEXITY AUDIT

Для каждой модели сохранить:

```text
parameter count
peak VRAM
training sec/epoch
inference ms/section
```

V2 не должна улучшать качество ценой неограниченного роста.

---

# 87. EFFICIENCY TARGET

Если v2 требует >3× VRAM v1:

оптимизировать implementation.

Если >3× inference time:

профилировать mode-pair computation.

Разрешено:

```text
top-k destination modes
vectorized unfold/einsum
shared message projection
```

Но математически результат должен оставаться эквивалентным выбранной формулировке.

---

# 88. TESTS V2

Добавить:

```text
test_v2_mode_normalization.py
test_v2_axial_periodicity.py
test_v2_orientation_compatibility.py
test_v2_directional_transport.py
test_v2_transport_mass.py
test_v2_junction_score.py
test_v2_cone_consistency.py
test_v2_limit_cases.py
test_v2_gradient_finite.py
```

---

# 89. CROSSINGTRACE TESTS

```text
single line
X crossing
T junction
Y junction
two parallel lines
close non-crossing lines
gap
curved crossing
```

Ожидания должны быть explicit.

---

# 90. NO TEST LEAKAGE

Synthetic:

```text
train/val/test RNG seeds disjoint
test generator config frozen
```

CRACKS:

```text
expert test scores not used for tuning
```

Setting B/C folds:

```text
fold IDs frozen before training
```

---

# 91. OUTPUT STRUCTURE

```text
results/anza_v2_study/
    protocol.json
    environment.txt

    synthetic/
        config.json
        model_results.csv
        structural_metrics.csv
        candidate_search.csv
        junction_cases/
        figures/

    cracks/
        archive_audit/
        mask_semantics/
        crowd_target/
        setting_A/
        setting_B/
        setting_C/
        human_comparison/
        disagreement/
        traces/
        figures/

    tables/
        main_cracks.csv
        structural_benchmark.csv
        ablations.csv
        human_comparison.csv
        disagreement_correlations.csv
        efficiency.csv

    THESIS_NUMBERS.json
    THESIS_EVIDENCE.md
    FINAL_REPORT.md
    SCIENTIFIC_AUDIT.md
```

---

# 92. THESIS_NUMBERS

Автоматически:

```json
{
  "synthetic": {
    "unet": {},
    "deformable": {},
    "anza_v1": {},
    "anza_v2": {},
    "branch_pairing": {},
    "false_merge": {},
    "false_split": {},
    "identity_switch": {}
  },
  "cracks": {
    "setting_A": {},
    "setting_B": {},
    "setting_C": {},
    "topology": {},
    "orientation": {},
    "fragmentation": {}
  },
  "human": {
    "novice": {},
    "practitioner": {},
    "expert_agreement": {},
    "disagreement": {}
  },
  "efficiency": {},
  "limitations": []
}
```

---

# 93. ОСНОВНЫЕ ТАБЛИЦЫ ДЛЯ БУДУЩИХ ТЕЗИСОВ

Не больше трёх ключевых.

### TABLE 1 — Real CRACKS

```text
U-Net
Deformable U-Net
ANZA v1
ANZA v2

Dice
clDice
Skeleton F1
Fragmentation
Orientation error
```

### TABLE 2 — Controlled structure

```text
V1
V2

branch pairing
false merge
false split
identity switch
gap recovery
```

### TABLE 3 — Ablation

```text
v2 full
-no fuzzy
-no direction
-no junction
-no cone
-no replay
```

---

# 94. FIGURE 1

Смысл оператора:

```text
same local crossing

V1:
mode evidence collapses

V2:
mode states remain separate
and route through junction
```

Сделать программно, scientific white-background schematic.

Не AI art.

---

# 95. FIGURE 2

Controlled benchmark:

```text
input seismic-like sample
GT instances
V1 traces
V2 traces
```

Особенно X/T intersection.

---

# 96. FIGURE 3

Real CRACKS:

```text
seismic section
expert annotation
U-Net
ANZA v1
ANZA v2
```

Median case.

Не cherry-pick best.

---

# 97. FIGURE 4

V2 internal geometry:

```text
orientation modes
junction score
routing entropy
final trace graph
```

---

# 98. FIGURE 5

Human disagreement:

```text
expert
crowd disagreement
routing entropy
model error
```

Только если результат содержательный.

---

# 99. SCIENTIFIC SUCCESS НЕ РАВЕН «DICE +0.1»

Работа является сильной, если подтвердится хотя бы одна из следующих историй:

### Story A

```text
similar Dice
but clearly better topology/fragmentation
```

### Story B

```text
similar segmentation
but fewer branch identity errors on controlled benchmark
```

### Story C

```text
V2 particularly improves junction/crossing cases
```

### Story D

```text
internal routing ambiguity agrees with human uncertainty
```

### Story E

```text
V2 gives better expert agreement than median non-expert
```

---

# 100. ЧТО БУДЕТ ПЛОХИМ РЕЗУЛЬТАТОМ

Не считать успехом:

```text
Dice +0.002
and no structural advantage
```

Тогда нужно честно сказать:

```text
new mechanism is not yet justified
```

и посмотреть ablations/root cause.

---

# 101. DEVELOPMENT LOOP

Если v2 не показывает структурное преимущество:

```text
DO NOT immediately tweak everything.
```

Сначала определить:

```text
Does orientation head fail?
Does mode collapse occur?
Does transition become uniform?
Does junction J fail?
Does half-mode routing flip?
Does segmentation erase structural advantage?
```

Вывести diagnostics.

---

# 102. MODE COLLAPSE METRIC

Обязательно считать:

[
N_{\mathrm{eff}}(p)
 ===================

\exp
\left[
-\sum_r
\mu_r(p)
\log(\mu_r(p)+\varepsilon)
\right].
]

Если:

```text
N_eff ≈ 1 everywhere
```

multimode mechanism фактически не используется.

Считать распределение:

```text
straight regions
junctions
background
```

---

# 103. JUNCTION SPECIALIZATION CHECK

На synthetic GT:

ожидаем:

```text
N_eff junction > N_eff straight
J junction > J straight
routing entropy junction > straight
```

Если нет:

v2 mechanism не работает как задуман.

Не переходить молча к CRACKS.

---

# 104. GENERALIZATION OF ROUTING

На synthetic test отдельно strata:

```text
seen angle ranges
unseen crossing angles
higher noise
larger gaps
different line widths
```

Это дешёвая OOD structural check.

---

# 105. СТАТИСТИКА

Synthetic:

unit = generated sample.

CRACKS:

unit = seismic section.

Не pixel.

Seeds:

```text
41
42
43
```

Основные model comparisons paired на одинаковых sections.

Bootstrap:

```text
2000
```

---

# 106. ONE-COMMAND ORCHESTRATION

Основной:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/anza_v2_study.py full
```

Он должен:

```text
verify archives
resolve label semantics
prepare crowd targets
build synthetic benchmark
run synthetic candidate development
freeze V2
run CRACKS crowd training
run expert evaluation
run fine-tuning CV
run image-disjoint robustness
run trace metrics
run human comparison
run disagreement analysis
run bootstrap
generate tables
generate figures
generate report
run scientific audit
run final validator
```

---

# 107. RESUME

Любая стадия:

```text
same config hash + COMPLETE
→ SKIP

interrupted
→ RESUME

changed config
→ new run ID
```

Никакого переобучения из-за изменения рисунка.

---

# 108. AGENT DISCIPLINE

Сохраняются старые правила:

```text
AGENTS.md
skills
RTK
TASK_STATE
EVIDENCE
small verified changes
reviewer mode
```

Не загружать этот master spec целиком на каждом шаге.

Сохранить:

```text
docs/research/anza_v2_master_spec.md
```

---

# 109. SKILLS

Создать/расширить только необходимые:

```text
cracks-data
crossing-trace-bench
anza-v2-transport
structural-metrics
cracks-experiment
human-disagreement
scientific-validation
thesis-evidence
```

Сначала проверить существующие и не плодить duplicates.

---

# 110. TOKEN DISCIPLINE

Training logs на диск.

Agent видит только:

```text
phase
model
seed
epoch
val metric
structural metric
status
ETA
```

RTK для шумных команд.

---

# 111. НЕ ДЕЛАТЬ КОММИТЫ

Текущее правило остаётся.

```text
NO intermediate commits
NO push
```

Branch:

```text
feature/cracks-final
```

Main не трогать.

Backup patch обновлять после больших PASS-фаз.

---

# 112. ЕДИНСТВЕННЫЙ FINAL COMMIT

Только после:

```text
all tests pass
synthetic benchmark complete
V2 frozen
CRACKS settings complete
statistics complete
figures complete
scientific audit pass
final validator pass
```

---

# 113. FINAL VALIDATOR

Создать:

```bash
python scripts/validate_anza_v2_study.py
```

Он проверяет:

```text
archives verified
mask semantics documented
synthetic generator deterministic
synthetic train/val/test separated
V1 baseline complete
deformable baseline complete
V2 complete
structural metrics complete
controlled test opened only after freeze
CRACKS crowd target complete
expert masks untouched during crowd-only model selection
Setting A complete
Setting B complete
Setting C complete
human baseline complete
disagreement analysis complete
bootstrap complete
figures complete
THESIS_NUMBERS complete
THESIS_EVIDENCE complete
SCIENTIFIC_AUDIT PASS
no NaN
no Inf
no TODO
no fake metrics
```

---

# 114. FINAL OUTPUT

Последняя строка:

```text
 ====================================================
ANZA-LIRA V2 STUDY STATUS: COMPLETE
 ====================================================
```

До неё задача НЕ завершена.

---

# 115. STOP CONDITIONS

Остановиться разрешено только:

### Real external blocker

```text
corrupt verified archive
physical GPU failure
disk inaccessible
missing essential released label semantics with no valid fallback
```

Но не:

```text
40 expert masks
metric lower than expected
model worse than baseline
one implementation bug
OOM
test failure
```

Это рабочие ситуации.

---

# 116. ВАЖНО: НЕ ОПТИМИЗИРОВАТЬ ДО «КРАСИВОГО RESULT»

Если после bounded development V2 хуже:

закончить эксперимент.

Отрицательный результат сохранить.

Не менять test.

Не пересобирать split.

Не выбирать seed задним числом.

---

# 117. SCIENTIFIC AUDIT

Перед финалом агент переходит в reviewer mode.

Ответить:

```text
What exactly is new relative to V1?

Why is this not just deformable convolution?

Why is this not just orientation prediction?

Why is this not just clDice/topology loss?

Did controlled benchmark really test branch identity?

Did real CRACKS data support the same mechanism?

Did synthetic structural supervision leak into real test?

Are expert labels hidden where claimed?

Are human comparisons phrased correctly?

Is Anosov language limited to the local geometric analogy?

Is every claim backed by a stored metric?
```

---

# 118. EXPECTED NOVELTY STATEMENT

Если эксперимент подтверждает гипотезу, будущая формулировка примерно такая:

> ANZA-LIRA v2 extends anisotropic fuzzy local aggregation from direction-sensitive weighting to mode-resolved structural transport. Instead of collapsing competing local orientation modes before aggregation, the operator maintains separate directional states and routes them through neighboring regions according to fuzzy membership, anisotropic geometry, axial orientation compatibility and local continuation direction. This allows the model to preserve competing structural hypotheses at crossings and to suppress transfers between incompatible branches.

Не вставлять это как готовый доказанный вывод до результатов.

---

# 119. ЕЩЁ БОЛЕЕ ВАЖНАЯ НОВИЗНА

Если branch metrics подтвердятся:

> The proposed operator addresses a limitation that is not captured by pixel overlap alone: preservation of line identity through intersections.

Вот это уже реально сильнее «ещё одна анизотропная свёртка».

---

# 120. ОДНО ПРЕДЛОЖЕНИЕ О ВСЕЙ РАБОТЕ

Кодер должен держать в `TASK_STATE`:

> We test whether preserving and transporting separate fuzzy orientation modes inside a local anisotropic operator can reduce branch mixing and structural fragmentation in intersecting line-like structures, and whether this mechanism remains useful for expert-level seismic fault delineation under noisy crowdsourced supervision.

---

## Что я здесь сознательно поменял

Главная вещь: **40 expert masks больше не надо насильно превращать в обычный большой supervised dataset**.

Это как раз соответствует исходной идее CRACKS: noisy novice/practitioner labels должны помогать приближать expert interpretation, причём авторы сами подчёркивают, что разные уровни expertise расходятся особенно в сложной геометрии и соединениях faults.

А для вопроса:

> «A и C — одна линия через X или две разные?»

мы больше **не делаем вид, что semantic CRACKS mask может дать нам правильный instance answer**. Для этого есть controlled `CrossingTraceBench`, где instance identity известна по построению.

Получается очень чистая доказательная линия:

**CrossingTraceBench доказывает механизм → CRACKS проверяет перенос на реальную сейсмику → crowd/expert disagreement проверяет, ведёт ли внутренняя неопределённость модели себя осмысленно.**

И самое важное: **`Anosov` мы не прикручиваем для красоты**. Оттуда берётся только содержательная локальная идея парного растяжения/сжатия и cone field. Эргодичность не заявляем.

Именно это ТЗ я бы теперь дал кодеру и **больше концепцию не менял до первых нормальных результатов**.

[1]: https://openaccess.thecvf.com/content_iccv_2017/html/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.html?utm_source=chatgpt.com "ICCV 2017 Open Access Repository"
[2]: https://arxiv.org/abs/2003.07311?utm_source=chatgpt.com "clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"
[3]: https://arxiv.org/abs/2307.08388?utm_source=chatgpt.com "Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation"
