# GARFIELD 算法

## 二元离散变量 (Case|Control)

负对数似然 (Negetive Log Likelihood, NLL) 作为评估函数:
$$
NLL=\sum{w_i*[ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ]}
$$
其中
$$
p_0 = mean(y | pred=0) \\
p_1 = mean(y | pred=1) \\
$$

NLL的推导：

- 首先是单个样本的似然 (两种可能性, `0`或`1`的概率)
$$P(y_i \mid p_i) = p_i^{y_i}(1-p_i)^{1-y_i}$$

- 随后累乘获得全部样本（独立）似然
$$L = \prod_i p_i^{y_i}(1-p_i)^{1-y_i}$$

- 最后取负对数得到NLL
$$-\log L = -\sum_i \big[ y_i \log p_i + (1-y_i)\log(1-p_i) \big]$$

## 连续变量

$$
\mu_0 = mean(y | pred=0) \\
\mu_1 = mean(y | pred=1) \\
MSE=\frac{\sum{(y_i-\mu_i)^2}}{n}
$$
