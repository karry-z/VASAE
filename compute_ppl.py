import json
import numpy as np

# 读取文件
with open("out/gpt2_squad_val_eval.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 假设文件是一个由对象组成的列表，每个对象有 "perplexity"
ppls = [item["perplexity"] for item in data]

# 计算均值和总体方差
mean = float(np.mean(ppls))
var = float(np.var(ppls))  # 若想样本方差用 np.var(ppls, ddof=1)

print("mean:", mean)
print("variance:", var)