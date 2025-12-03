# 因果推断：IPS → DR → CATE（因果森林）复习总结
---

# 1. 潜在结果框架（Potential Outcome Framework）

用户特征：X  
策略（Treatment）：T ∈ {0,1}

- T=1：推自助  
- T=0：转人工  

潜在结果：
- Y(1)：如果推自助，会发生什么  
- Y(0)：如果推人工，会发生什么  

实际只观测到：
```
Y = T·Y(1) + (1–T)·Y(0)
```

目标：估计 CATE（Conditional Average Treatment Effect）
```
τ(x) = E[Y(1) – Y(0) | X=x]
```

---

# 2. IPS：反事实加权（纠偏）

历史策略 π₀ 产生偏置数据，例如“总是推人工”。

Propensity（展示概率）：
```
e(x) = P(T=1 | X=x)
```

IPS 无偏估计：
```
Y_IPS = T·Y / e(x)  –  (1–T)·Y / (1–e(x))
```

通俗理解：  
👉“展示少的样本应该加权，让训练数据变得公平。”

---

# 3. DR：双稳健估计（更稳定）

构建 outcome model：
```
μ̂_t(x) = E[Y | X=x, T=t]
```

DR 估计器：
```
Y_DR = 
    T (Y − μ̂₁(X)) / e(X)
  − (1−T)(Y − μ̂₀(X)) / (1−e(X))
  + (μ̂₁(X) − μ̂₀(X))
```

特点：  
👉只要 e(x) 或 μ̂_t(x) 任意一个对，最终估计就可靠（double robustness）。

通俗理解：  
👉“真实数据 + 预测器”互相兜底，结果不乱跳。

---

# 4. 因果森林 = DR score → 学习 CATE

伪结果（pseudo outcome）φ_i 和 DR 完全一致：
```
φ_i = 
    T_i (Y_i − μ̂₁(X_i)) / e(X_i)
  − (1−T_i)(Y_i − μ̂₀(X_i)) / (1−e(X_i))
  + (μ̂₁(X_i) − μ̂₀(X_i))
```

因果森林的目标：
```
τ(x) = E[φ | X=x]
```

通俗理解：  
👉“DR 给我们公平的数据，因果森林用它来找不同用户的策略效果差异。”

---

# 5. Honesty（诚实性）机制

将数据分成 S₁、S₂：

- S₁：用于 split（切树）  
- S₂：用于估计 treatment effect  

数学形式：
```
Split on S₁  
Estimate on S₂
```

通俗理解：  
👉“训练用练习册，考试用新题”，避免模型自己骗分，CATE 更可信。

---

# 6. 最终个性化决策：用 CATE 选择策略

假设自助收益 = R_self  
转人工成本 = C_agent

策略收益：
```
U(x, T) = E[Y(T) | X=x] − Cost(T)
```

选择最佳策略：
```
T* = argmax U(x, T)
```

CATE 进入决策：
```
如果 τ(x) > C_agent → 推自助  
否则 → 尽早转人工
```

通俗理解：  
👉“对这个用户，自助能省多少钱？如果比人工成本高，就推自助。”

---

# 7. 一句话面试总结

**“我先用 IPS 去掉历史策略偏差，再用 DR 让估计更稳，然后把 DR 的 pseudo-outcome 输入因果森林，利用 honesty 估计不同用户的 CATE。最后根据 CATE 和 agent 成本，决定给这个用户走自助还是人工，实现个性化成本优化。”**

---

