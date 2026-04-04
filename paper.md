因为 $M$ 对称，所以任意向量 $x$ 都能写成
$$
x=\sum_{i=1}^n c_i v_i.
$$
于是
$$
x^\top x=\sum_{i=1}^n c_i^2,
$$
而
$$
x^\top Mx
=
\left(\sum_i c_i v_i\right)^\top
M
\left(\sum_j c_j v_j\right).
$$
由于 $Mv_i=\lambda_i v_i$，且特征向量正交归一，展开后得到
$$
x^\top Mx=\sum_{i=1}^n \lambda_i c_i^2.
$$
所以瑞利商
$$
R_M(x):=\frac{x^\top Mx}{x^\top x}
$$
可以写成
$$
R_M(x)=
\frac{\sum_{i=1}^n \lambda_i c_i^2}{\sum_{i=1}^n c_i^2}.
$$
这其实就是一个 **加权平均**，权重是 $c_i^2\ge 0$。

因此立刻有
$$
\lambda_1 \le R_M(x)\le \lambda_n.
$$
而且：

- 当 $x=v_1$ 时，取到最小值 $\lambda_1$；
- 当 $x=v_n$ 时，取到最大值 $\lambda_n$。

所以
$$
\lambda_1=\min_{x\neq 0}\frac{x^\top Mx}{x^\top x}.
$$