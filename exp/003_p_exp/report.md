---
title: 
date:
id: 
dependencies:
---

## 目的

## 方法

## 结果


第一步，先排除一个最危险的假阳性：你是不是只是把 decoder 拉到 embedding 上了，但 latent/feature 并没有变得更有用或更可解释。

立刻补 3 个检查。

一，看 activation-level interpretability，不是只看 decoder 几何。
对高对齐 feature，取 top activating contexts，检查这些上下文是否真的和 top-1 token 或其词义相关。
如果 feature 只是方向上贴住某个 token embedding，但激活时并不对应那个 token/语义，那这个 regularizer 学到的是“几何贴脸”，不是解释性。

二，看 feature usage distribution。
统计每个 feature 的平均激活频率、激活强度、dead feature 比例。
如果加 anchoring 后，大量 feature 贴到 token embedding 上，但其实几乎不被用，结果也不能算成功。

三，看 encoder-decoder consistency。
对每个强对齐 feature d_i，检查其高激活样本里，输入 token 是否更常落在对应 top token 附近，或者是否有稳定语义模式。
你现在证明了 decoder 可被拉齐，但还没证明整个 feature 通道变得“词汇可解释”。

第二步，做一个最关键的对照：这到底是“lexical interpretability 提升”，还是“任意外部字典都能这样吸过去”。

最少补两个 baseline：

一个是 random embedding baseline。
把 W_E 的 token 行随机打乱，或者直接换成同维度随机正交/高斯字典，跑同样 anchor loss。
如果也能几乎零代价把 cosine 拉到 0.99，那说明你现在看到的主要是“max-cos 吸附效应”，不是 vocab geometry 真有特殊性。

另一个是 frozen nearest-token assignment baseline 或更简单地说，训练后做 post-hoc token matching。
你已经有 plain SAE 的结果了。现在要比的是：
加训练时 anchoring 后，除了 cosine 上升，activation interpretability 是否真的比 post-hoc matching 更好。

第三步，做一个最小机制消融，决定后面往哪写。

只扫这三项就够：

一，\lambda 再细一点，只在有效区间扫。
现在看起来 10^{-4} 到 10^{-3} 已经很强，尤其 layer 11 几乎一碰就贴上去。
所以别再扫 10^{-2} 这种粗点了，改扫：
0,\ 3\times 10^{-5},\ 10^{-4},\ 3\times 10^{-4},\ 10^{-3}
你要找到的是 Pareto 最优点，不是证明“越大越对齐”。

二，anchor 目标从 max 改成 top-k soft version 做一个对照。
你现在这个
-\max_j \cos(d_i,e_j)
很容易把 feature 直接吸到某一个稀有 token 上。
这就是为什么你看到很多奇怪稀有 subword。
补一个更平滑版本，比如对 top-k token 的 log-sum-exp 或 softmax-weighted cosine。
如果平滑版能保住重构，同时 feature 更稳定、更不容易吸到垃圾 token，那就更像可发表的方法。

三，补一个 token frequency analysis。
统计强对齐 feature 对应 token 的词频分布。
现在示例里大量是稀有、碎片化、奇怪 subword。这个很危险。
如果最后发现模型最喜欢锚到低频怪 token，因为它们在 embedding 空间里更“孤立”、更容易拿到 0.999 cosine，那你现在的结果就不是“解释性变强”，而是“正则诱导 feature 去占据容易匹配的孤立词向量”。

所以最该先做的是：

A. context-based interpretability
高对齐 feature 的 top activating contexts，人工看 50 个。

B. random-dictionary control
把真实 W_E 换成随机字典，重跑一组 \lambda=10^{-4},10^{-3}。

C. token frequency / rarity analysis
高对齐 feature 对应 token 的频率分布，和 plain nearest-token baseline 比。

如果这三步结果都站得住，主线就清楚了：

普通 SAE 不会自然形成 lexical directions；
但很弱的 lexical anchoring 就能以极小重构代价诱导出 token-aligned features；
而且这种 alignment 不只是几何贴合，还转化成了更强的 activation-level interpretability。

如果 B 挂了，也就是随机字典同样有效，那你得立刻降温，因为那说明“vocab”不是关键。
如果 A 挂了，也就是上下文不支持 token 语义，那说明你现在只学到了 cosmetic alignment。
如果 C 显示全是稀有碎 token，那你要把目标从“token-specific”改成“lexically anchored”并重设计正则。


