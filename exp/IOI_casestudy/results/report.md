---
title: IOI Feature Case Study — 具体因果效果
date: 2026-03-19
---

# 目的

验证 VASAE 特征的 vocab alignment 与其因果效果之间的对应关系。
具体地：如果一个 feature 的 decoder 方向对齐女性名字（如 `herself`, `Anne`, `Elizabeth`），
那么消融该 feature 后，模型在 **correct name 为女性** 的 IOI prompt 上应该表现更差（logit diff 下降），
而在 **correct name 为男性** 的 prompt 上影响应该较小。反之亦然。

# 干预方式

对每个目标 feature $f$ 在 layer $l$：

1. 用 clean prompt 前向传播到 layer $l$，得到 hidden state $h$
2. SAE encode：$h \to z$（topk=32），其中 $z_f$ 为 feature 激活强度
3. **干预**：$z_f \leftarrow 0$（仅置零该 feature，其余不变）
4. SAE decode 并 patch 回 layer $l$，继续前向传播
5. 测量 $\Delta u = \text{logit}(\text{correct}) - \text{logit}(\text{wrong})$ 的变化

**关键指标**：消融后 correct name 的 logit 降幅 vs wrong name 的 logit 降幅。
如果 feature 确实编码了特定性别的名字信息，那么匹配性别的 name logit 应该降得**更多**。

# L7 Feature 3733

**Vocab alignment**：` herself`(0.2579)、` husband`(0.1828)、` Anne`(0.1633)、` Elizabeth`(0.1489)、` breasts`(0.1437)

**激活情况**：55/100 个 prompt （correct=女 46 个，correct=男 9 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 46 | **-0.9327** | -5.1600 | -4.2273 |
| 男 | 9 | **+0.0217** | -0.0756 | -0.0973 |

**解读**：
该 feature 的 decoder 方向对齐 **女性名字/性别** token。
消融后，correct=女性名的 prompt Δu 平均下降 -0.933，而 correct=男性名仅下降 0.022。**符合预期**：女性名字 feature 选择性地影响女性 correct name。

![Logit Diff Shift](L7_F3733_logitdiff.pdf)

![Rank Shift](L7_F3733_ranks.pdf)

## 代表性案例

### Prompt #8（correct=女，wrong=男）

> After Elizabeth and John went to the hospital, John gave a snack to Elizabeth

- correct=` Elizabeth`（女），wrong=` John`（男），strength=**27.7927**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -121.3608 | -121.7681 | **0.4073** | 280 | 387 |
| 干预后 | -129.5468 | -127.2047 | **-2.3422** | 1718 | 228 |
| 变化 | -8.1860 | -5.4366 | **-2.7495** | +1438 | -159 |

**不对称性**：correct name logit 降幅（-8.19）大于 wrong name（-5.44），差值 2.75。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #89（correct=女，wrong=男）

> Then, Samuel and Mary were working at the station. Samuel decided to give a drink to Mary

- correct=` Mary`（女），wrong=` Samuel`（男），strength=**30.5498**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -127.9131 | -130.8618 | **2.9487** | 180 | 1922 |
| 干预后 | -134.6963 | -135.1013 | **0.4050** | 388 | 555 |
| 变化 | -6.7832 | -4.2395 | **-2.5437** | +208 | -1367 |

**不对称性**：correct name logit 降幅（-6.78）大于 wrong name（-4.24），差值 2.54。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #19（correct=男，wrong=男）

> Then, Mark and Jesse had a lot of fun at the garden. Mark gave a drink to Jesse

- correct=` Jesse`（男），wrong=` Mark`（男），strength=**5.7031**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -122.3453 | -121.7801 | **-0.5653** | 1863 | 1127 |
| 干预后 | -122.1937 | -121.8033 | **-0.3904** | 1601 | 1121 |
| 变化 | +0.1516 | -0.0232 | **+0.1749** | -262 | -6 |

**不对称性**：correct name logit 降幅（+0.15）大于 wrong name（-0.02），差值 0.13。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #88（correct=男，wrong=男）

> Then, Justin and Nicholas had a lot of fun at the store. Justin gave a basketball to Nicholas

- correct=` Nicholas`（男），wrong=` Justin`（男），strength=**8.3283**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -123.0920 | -122.5838 | **-0.5082** | 2920 | 2022 |
| 干预后 | -123.1164 | -122.5351 | **-0.5813** | 2678 | 1766 |
| 变化 | -0.0244 | +0.0487 | **-0.0731** | -242 | -256 |

correct 和 wrong name logit 降幅接近（-0.02 vs +0.05），该 feature 对两个名字的影响较对称。

# L4 Feature 4649

**Vocab alignment**：`pher`(0.1488)、` Anne`(0.1459)、` Elizabeth`(0.1191)、`athed`(0.1179)、` Marie`(0.1175)

**激活情况**：35/100 个 prompt （correct=女 35 个，correct=男 0 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 35 | **-0.9956** | -3.0871 | -2.0916 |

**解读**：
该 feature 的 decoder 方向对齐 **女性名字/性别** token。

![Logit Diff Shift](L4_F4649_logitdiff.pdf)

![Rank Shift](L4_F4649_ranks.pdf)

## 代表性案例

### Prompt #89（correct=女，wrong=男）

> Then, Samuel and Mary were working at the station. Samuel decided to give a drink to Mary

- correct=` Mary`（女），wrong=` Samuel`（男），strength=**11.7999**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -110.1791 | -115.6507 | **5.4716** | 19 | 1530 |
| 干预后 | -113.0301 | -114.6386 | **1.6085** | 71 | 303 |
| 变化 | -2.8510 | +1.0121 | **-3.8631** | +52 | -1227 |

**不对称性**：correct name logit 降幅（-2.85）大于 wrong name（+1.01），差值 1.84。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #8（correct=女，wrong=男）

> After Elizabeth and John went to the hospital, John gave a snack to Elizabeth

- correct=` Elizabeth`（女），wrong=` John`（男），strength=**10.1354**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -115.5357 | -116.9784 | **1.4427** | 102 | 351 |
| 干预后 | -121.9660 | -120.8421 | **-1.1239** | 525 | 201 |
| 变化 | -6.4303 | -3.8637 | **-2.5666** | +423 | -150 |

**不对称性**：correct name logit 降幅（-6.43）大于 wrong name（-3.86），差值 2.57。该 feature 不成比例地提升了 correct name 的 logit。

# L0 Feature 5628

**Vocab alignment**：` Nicole`(0.1538)、` herself`(0.1446)、` Anne`(0.1376)、` Mae`(0.1355)、` daughter`(0.1266)

**激活情况**：46/100 个 prompt （correct=女 46 个，correct=男 0 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 46 | **-0.9148** | -1.4732 | -0.5584 |

**解读**：
该 feature 的 decoder 方向对齐 **女性名字/性别** token。

![Logit Diff Shift](L0_F5628_logitdiff.pdf)

![Rank Shift](L0_F5628_ranks.pdf)

## 代表性案例

### Prompt #8（correct=女，wrong=男）

> After Elizabeth and John went to the hospital, John gave a snack to Elizabeth

- correct=` Elizabeth`（女），wrong=` John`（男），strength=**14.8519**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -96.7779 | -97.3917 | **0.6138** | 102 | 156 |
| 干预后 | -97.3480 | -94.6847 | **-2.6633** | 530 | 78 |
| 变化 | -0.5701 | +2.7070 | **-3.2771** | +428 | -78 |

**不对称性**：wrong name logit 降幅（+2.71）大于 correct name（-0.57）。该 feature 主要影响 wrong name 方向。

### Prompt #38（correct=女，wrong=男）

> Then, Richard and Emily had a long argument, and afterwards Richard said to Emily

- correct=` Emily`（女），wrong=` Richard`（男），strength=**15.5898**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -78.0291 | -79.2077 | **1.1786** | 113 | 218 |
| 干预后 | -78.6736 | -77.1512 | **-1.5225** | 461 | 158 |
| 变化 | -0.6445 | +2.0565 | **-2.7011** | +348 | -60 |

**不对称性**：wrong name logit 降幅（+2.06）大于 correct name（-0.64）。该 feature 主要影响 wrong name 方向。

# L6 Feature 5628

**Vocab alignment**：` Anne`(0.191)、` Marie`(0.1696)、` herself`(0.1631)、` Margaret`(0.1584)、` Mary`(0.1581)

**激活情况**：46/100 个 prompt （correct=女 46 个，correct=男 0 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 46 | **-0.6501** | -4.0859 | -3.4358 |

**解读**：
该 feature 的 decoder 方向对齐 **女性名字/性别** token。

![Logit Diff Shift](L6_F5628_logitdiff.pdf)

![Rank Shift](L6_F5628_ranks.pdf)

## 代表性案例

### Prompt #89（correct=女，wrong=男）

> Then, Samuel and Mary were working at the station. Samuel decided to give a drink to Mary

- correct=` Mary`（女），wrong=` Samuel`（男），strength=**19.4146**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -122.0789 | -127.2954 | **5.2165** | 45 | 2908 |
| 干预后 | -129.0580 | -131.6948 | **2.6368** | 141 | 1227 |
| 变化 | -6.9791 | -4.3994 | **-2.5797** | +96 | -1681 |

**不对称性**：correct name logit 降幅（-6.98）大于 wrong name（-4.40），差值 2.58。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #8（correct=女，wrong=男）

> After Elizabeth and John went to the hospital, John gave a snack to Elizabeth

- correct=` Elizabeth`（女），wrong=` John`（男），strength=**17.5819**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -112.8463 | -113.7988 | **0.9525** | 146 | 297 |
| 干预后 | -118.0402 | -116.9477 | **-1.0925** | 466 | 198 |
| 变化 | -5.1939 | -3.1489 | **-2.0450** | +320 | -99 |

**不对称性**：correct name logit 降幅（-5.19）大于 wrong name（-3.15），差值 2.05。该 feature 不成比例地提升了 correct name 的 logit。

# L5 Feature 13

**Vocab alignment**：` Mae`(0.1462)、` Maria`(0.1292)、` Nicole`(0.1221)、` Marie`(0.1215)、` Jane`(0.1203)

**激活情况**：47/100 个 prompt （correct=女 45 个，correct=男 2 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 45 | **-0.8105** | -5.2622 | -4.4517 |
| 男 | 2 | **+0.2211** | -0.8346 | -1.0557 |

**解读**：
该 feature 的 decoder 方向对齐 **女性名字/性别** token。
消融后，correct=女性名的 prompt Δu 平均下降 -0.810，而 correct=男性名仅下降 0.221。**符合预期**：女性名字 feature 选择性地影响女性 correct name。

![Logit Diff Shift](L5_F13_logitdiff.pdf)

![Rank Shift](L5_F13_ranks.pdf)

## 代表性案例

### Prompt #63（correct=女，wrong=男）

> Then, Alicia and Aaron went to the house. Aaron gave a ring to Alicia

- correct=` Alicia`（女），wrong=` Aaron`（男），strength=**24.0133**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -117.6026 | -117.0778 | **-0.5248** | 4706 | 3379 |
| 干预后 | -122.8810 | -119.6469 | **-3.2341** | 14075 | 1985 |
| 变化 | -5.2784 | -2.5691 | **-2.7093** | +9369 | -1394 |

**不对称性**：correct name logit 降幅（-5.28）大于 wrong name（-2.57），差值 2.71。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #89（correct=女，wrong=男）

> Then, Samuel and Mary were working at the station. Samuel decided to give a drink to Mary

- correct=` Mary`（女），wrong=` Samuel`（男），strength=**25.1839**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -119.9626 | -123.3982 | **3.4355** | 38 | 909 |
| 干预后 | -127.1923 | -128.2487 | **1.0564** | 149 | 333 |
| 变化 | -7.2297 | -4.8505 | **-2.3791** | +111 | -576 |

**不对称性**：correct name logit 降幅（-7.23）大于 wrong name（-4.85），差值 2.38。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #31（correct=男，wrong=女）

> After Lindsey and Jose went to the station, Lindsey gave a snack to Jose

- correct=` Jose`（男），wrong=` Lindsey`（女），strength=**6.0376**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -117.4175 | -117.5939 | **0.1764** | 2325 | 2622 |
| 干预后 | -118.1742 | -118.7694 | **0.5952** | 2248 | 3465 |
| 变化 | -0.7567 | -1.1755 | **+0.4188** | -77 | +843 |

**不对称性**：wrong name logit 降幅（-1.18）大于 correct name（-0.76）。该 feature 主要影响 wrong name 方向。

### Prompt #88（correct=男，wrong=男）

> Then, Justin and Nicholas had a lot of fun at the store. Justin gave a basketball to Nicholas

- correct=` Nicholas`（男），wrong=` Justin`（男），strength=**6.0641**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -123.7166 | -123.1414 | **-0.5752** | 3028 | 1974 |
| 干预后 | -124.6291 | -124.0773 | **-0.5518** | 3066 | 1984 |
| 变化 | -0.9125 | -0.9359 | **+0.0234** | +38 | +10 |

correct 和 wrong name logit 降幅接近（-0.91 vs -0.94），该 feature 对两个名字的影响较对称。

# L3 Feature 278

**Vocab alignment**：`pher`(0.1411)、` Anne`(0.128)、`athed`(0.127)、` herself`(0.1193)、`ding`(0.1152)

**激活情况**：32/100 个 prompt （correct=女 32 个，correct=男 0 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 32 | **-0.7492** | -0.6360 | +0.1132 |

**解读**：
该 feature 的 decoder 方向对齐 **女性名字/性别** token。

![Logit Diff Shift](L3_F278_logitdiff.pdf)

![Rank Shift](L3_F278_ranks.pdf)

## 代表性案例

### Prompt #38（correct=女，wrong=男）

> Then, Richard and Emily had a long argument, and afterwards Richard said to Emily

- correct=` Emily`（女），wrong=` Richard`（男），strength=**9.3112**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -103.9855 | -104.1614 | **0.1759** | 78 | 91 |
| 干预后 | -104.7952 | -102.5575 | **-2.2377** | 205 | 37 |
| 变化 | -0.8097 | +1.6039 | **-2.4136** | +127 | -54 |

**不对称性**：wrong name logit 降幅（+1.60）大于 correct name（-0.81）。该 feature 主要影响 wrong name 方向。

### Prompt #97（correct=女，wrong=男）

> Then, Sean and Samantha had a lot of fun at the school. Sean gave a computer to Samantha

- correct=` Samantha`（女），wrong=` Sean`（男），strength=**12.0023**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -116.2844 | -117.4943 | **1.2099** | 205 | 567 |
| 干预后 | -116.3284 | -115.4282 | **-0.9001** | 616 | 290 |
| 变化 | -0.0440 | +2.0661 | **-2.1100** | +411 | -277 |

**不对称性**：wrong name logit 降幅（+2.07）大于 correct name（-0.04）。该 feature 主要影响 wrong name 方向。

# L8 Feature 3477

**Vocab alignment**：` herself`(0.2465)、`pher`(0.1818)、` Anne`(0.1759)、` husband`(0.1743)、` Elizabeth`(0.1664)

**激活情况**：42/100 个 prompt （correct=女 42 个，correct=男 0 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 42 | **-0.4785** | -3.3614 | -2.8828 |

**解读**：
该 feature 的 decoder 方向对齐 **女性名字/性别** token。

![Logit Diff Shift](L8_F3477_logitdiff.pdf)

![Rank Shift](L8_F3477_ranks.pdf)

## 代表性案例

### Prompt #89（correct=女，wrong=男）

> Then, Samuel and Mary were working at the station. Samuel decided to give a drink to Mary

- correct=` Mary`（女），wrong=` Samuel`（男），strength=**20.6887**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -109.8328 | -114.2695 | **4.4367** | 132 | 4125 |
| 干预后 | -117.6050 | -119.9357 | **2.3307** | 305 | 2140 |
| 变化 | -7.7722 | -5.6662 | **-2.1060** | +173 | -1985 |

**不对称性**：correct name logit 降幅（-7.77）大于 wrong name（-5.67），差值 2.11。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #8（correct=女，wrong=男）

> After Elizabeth and John went to the hospital, John gave a snack to Elizabeth

- correct=` Elizabeth`（女），wrong=` John`（男），strength=**15.8891**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -103.4400 | -104.6175 | **1.1775** | 183 | 475 |
| 干预后 | -110.7717 | -110.5892 | **-0.1825** | 474 | 404 |
| 变化 | -7.3317 | -5.9717 | **-1.3600** | +291 | -71 |

**不对称性**：correct name logit 降幅（-7.33）大于 wrong name（-5.97），差值 1.36。该 feature 不成比例地提升了 correct name 的 logit。

# L1 Feature 5386

**Vocab alignment**：`pher`(0.174)、`ding`(0.1562)、` daughter`(0.1336)、`ded`(0.1288)、` husband`(0.1262)

**激活情况**：34/100 个 prompt （correct=女 34 个，correct=男 0 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 34 | **-0.7397** | +0.7928 | +1.5325 |

**解读**：
该 feature 的 decoder 方向对齐 **女性名字/性别** token。

![Logit Diff Shift](L1_F5386_logitdiff.pdf)

![Rank Shift](L1_F5386_ranks.pdf)

## 代表性案例

### Prompt #63（correct=女，wrong=男）

> Then, Alicia and Aaron went to the house. Aaron gave a ring to Alicia

- correct=` Alicia`（女），wrong=` Aaron`（男），strength=**5.3768**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -87.0433 | -88.8142 | **1.7709** | 10 | 78 |
| 干预后 | -97.3817 | -96.4757 | **-0.9060** | 116 | 51 |
| 变化 | -10.3384 | -7.6615 | **-2.6769** | +106 | -27 |

**不对称性**：correct name logit 降幅（-10.34）大于 wrong name（-7.66），差值 2.68。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #8（correct=女，wrong=男）

> After Elizabeth and John went to the hospital, John gave a snack to Elizabeth

- correct=` Elizabeth`（女），wrong=` John`（男），strength=**8.8916**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -94.6094 | -93.4394 | **-1.1700** | 16 | 5 |
| 干预后 | -96.6011 | -93.2083 | **-3.3928** | 40 | 5 |
| 变化 | -1.9917 | +0.2311 | **-2.2228** | +24 | +0 |

**不对称性**：correct name logit 降幅（-1.99）大于 wrong name（+0.23），差值 1.76。该 feature 不成比例地提升了 correct name 的 logit。

# L6 Feature 3042

**Vocab alignment**：` Francis`(0.1302)、` James`(0.1284)、` F`(0.1273)、` Henry`(0.1272)、` Daniel`(0.1226)

**激活情况**：86/100 个 prompt （correct=女 32 个，correct=男 54 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 32 | **-0.0938** | -0.6370 | -0.5432 |
| 男 | 54 | **-0.4378** | -5.8219 | -5.3841 |

**解读**：
该 feature 的 decoder 方向对齐 **男性名字** token。
消融后，correct=男性名的 prompt Δu 平均下降 -0.438，而 correct=女性名仅下降 -0.094。**符合预期**：男性名字 feature 选择性地影响男性 correct name。

![Logit Diff Shift](L6_F3042_logitdiff.pdf)

![Rank Shift](L6_F3042_ranks.pdf)

## 代表性案例

### Prompt #69（correct=女，wrong=男）

> Then, Jeremy and Amber went to the school. Jeremy gave a computer to Amber

- correct=` Amber`（女），wrong=` Jeremy`（男），strength=**6.2231**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -106.1562 | -108.7074 | **2.5512** | 618 | 4025 |
| 干预后 | -104.7699 | -106.4040 | **1.6341** | 997 | 2990 |
| 变化 | +1.3863 | +2.3034 | **-0.9171** | +379 | -1035 |

**不对称性**：wrong name logit 降幅（+2.30）大于 correct name（+1.39）。该 feature 主要影响 wrong name 方向。

### Prompt #89（correct=女，wrong=男）

> Then, Samuel and Mary were working at the station. Samuel decided to give a drink to Mary

- correct=` Mary`（女），wrong=` Samuel`（男），strength=**17.4766**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -122.0789 | -127.2954 | **5.2165** | 45 | 2908 |
| 干预后 | -129.2723 | -133.8783 | **4.6060** | 99 | 3196 |
| 变化 | -7.1934 | -6.5829 | **-0.6105** | +54 | +288 |

**不对称性**：correct name logit 降幅（-7.19）大于 wrong name（-6.58），差值 0.61。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #58（correct=男，wrong=男）

> Then, Jason and Joseph had a long argument, and afterwards Jason said to Joseph

- correct=` Joseph`（男），wrong=` Jason`（男），strength=**26.2883**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -105.3449 | -108.8513 | **3.5065** | 41 | 556 |
| 干预后 | -106.4450 | -107.9309 | **1.4859** | 45 | 124 |
| 变化 | -1.1001 | +0.9204 | **-2.0206** | +4 | -432 |

**不对称性**：correct name logit 降幅（-1.10）大于 wrong name（+0.92），差值 0.18。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #67（correct=男，wrong=男）

> Then, Richard and Joseph had a long argument, and afterwards Richard said to Joseph

- correct=` Joseph`（男），wrong=` Richard`（男），strength=**27.5269**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -103.5998 | -105.5958 | **1.9960** | 52 | 228 |
| 干预后 | -106.3428 | -106.7464 | **0.4037** | 50 | 67 |
| 变化 | -2.7430 | -1.1506 | **-1.5923** | -2 | -161 |

**不对称性**：correct name logit 降幅（-2.74）大于 wrong name（-1.15），差值 1.59。该 feature 不成比例地提升了 correct name 的 logit。

# L0 Feature 783

**Vocab alignment**：` James`(0.2005)、` Paul`(0.188)、` Thomas`(0.1752)、` Patrick`(0.1739)、` Howard`(0.1724)

**激活情况**：100/100 个 prompt （correct=女 46 个，correct=男 54 个）

## 性别分组汇总

| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |
| --- | --- | --- | --- | --- |
| 女 | 46 | **+0.2460** | +0.5048 | +0.2587 |
| 男 | 54 | **-0.4345** | -4.8132 | -4.3788 |

**解读**：
该 feature 的 decoder 方向对齐 **男性名字** token。
消融后，correct=男性名的 prompt Δu 平均下降 -0.434，而 correct=女性名仅下降 0.246。**符合预期**：男性名字 feature 选择性地影响男性 correct name。

![Logit Diff Shift](L0_F783_logitdiff.pdf)

![Rank Shift](L0_F783_ranks.pdf)

## 代表性案例

### Prompt #70（correct=女，wrong=男）

> Then, Lindsay and Joseph went to the store. Joseph gave a snack to Lindsay

- correct=` Lindsay`（女），wrong=` Joseph`（男），strength=**12.0603**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -99.1728 | -97.1260 | **-2.0468** | 1151 | 235 |
| 干预后 | -93.3547 | -89.8130 | **-3.5417** | 3419 | 213 |
| 变化 | +5.8181 | +7.3130 | **-1.4949** | +2268 | -22 |

**不对称性**：wrong name logit 降幅（+7.31）大于 correct name（+5.82）。该 feature 主要影响 wrong name 方向。

### Prompt #41（correct=女，wrong=男）

> When Charles and Sara got a bone at the house, Charles decided to give it to Sara

- correct=` Sara`（女），wrong=` Charles`（男），strength=**7.4063**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -108.1393 | -107.1077 | **-1.0316** | 1373 | 606 |
| 干预后 | -102.7455 | -103.1875 | **0.4420** | 542 | 774 |
| 变化 | +5.3938 | +3.9202 | **+1.4736** | -831 | +168 |

**不对称性**：correct name logit 降幅（+5.39）大于 wrong name（+3.92），差值 1.47。该 feature 不成比例地提升了 correct name 的 logit。

### Prompt #62（correct=男，wrong=女）

> Then, Michael and Amber had a lot of fun at the restaurant. Amber gave a computer to Michael

- correct=` Michael`（男），wrong=` Amber`（女），strength=**18.0029**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -113.0122 | -118.3679 | **5.3557** | 64 | 2484 |
| 干预后 | -99.1919 | -101.0818 | **1.8899** | 243 | 959 |
| 变化 | +13.8203 | +17.2861 | **-3.4658** | +179 | -1525 |

**不对称性**：wrong name logit 降幅（+17.29）大于 correct name（+13.82）。该 feature 主要影响 wrong name 方向。

### Prompt #53（correct=男，wrong=男）

> After Jose and Brandon went to the hospital, Jose gave a snack to Brandon

- correct=` Brandon`（男），wrong=` Jose`（男），strength=**12.1812**

| | logit(correct) | logit(wrong) | $\Delta u$ | rank(correct) | rank(wrong) |
| --- | --- | --- | --- | --- | --- |
| 干预前 | -102.0944 | -102.2498 | **0.1554** | 396 | 449 |
| 干预后 | -113.0054 | -110.4560 | **-2.5494** | 50 | 9 |
| 变化 | -10.9110 | -8.2062 | **-2.7048** | -346 | -440 |

**不对称性**：correct name logit 降幅（-10.91）大于 wrong name（-8.21），差值 2.70。该 feature 不成比例地提升了 correct name 的 logit。

# 总结

| Feature | Vocab 方向 | 类型 | correct=女 Δu | correct=男 Δu | 符合预期？ |
| --- | --- | --- | --- | --- | --- |
| L7 F3733 | herself/husband/Anne | 女性 | -0.933 | +0.022 | 是 |
| L4 F4649 | pher/Anne/Elizabeth | 女性 | -0.996 | +0.000 | 是 |
| L0 F5628 | Nicole/herself/Anne | 女性 | -0.915 | +0.000 | 是 |
| L6 F5628 | Anne/Marie/herself | 女性 | -0.650 | +0.000 | 是 |
| L5 F13 | Mae/Maria/Nicole | 女性 | -0.810 | +0.221 | 是 |
| L3 F278 | pher/Anne/athed | 女性 | -0.749 | +0.000 | 是 |
| L8 F3477 | herself/pher/Anne | 女性 | -0.479 | +0.000 | 是 |
| L1 F5386 | pher/ding/daughter | 女性 | -0.740 | +0.000 | 是 |
| L6 F3042 | Francis/James/F | 男性 | -0.094 | -0.438 | 是 |
| L0 F783 | James/Paul/Thomas | 男性 | +0.246 | -0.434 | 是 |
