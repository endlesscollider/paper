"""
NumPy 手写 scaled dot-product attention。

运行：
  python code/numpy_attention.py

你会看到：
  1. 输入 X 的形状；
  2. Q/K/V 的形状；
  3. attention score 和 weight；
  4. 输出 token 表示。
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    out = weights @ V
    return out, scores, weights


def main():
    np.set_printoptions(precision=3, suppress=True)

    # 3 个 token，每个 token 4 维。
    # 你可以把它们想成：[红色] [方块] [盒子]
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ])

    # 为了教学清楚，这里手写固定权重。
    W_Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    W_K = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
    ])
    W_V = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ])

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    out, scores, weights = attention(Q, K, V)

    print("X shape:", X.shape)
    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    print("\nQ:\n", Q)
    print("\nK:\n", K)
    print("\nV:\n", V)
    print("\nScaled QK^T scores:\n", scores)
    print("\nAttention weights, each row sums to 1:\n", weights)
    print("\nOutput:\n", out)

    print("\n练习：修改 X 或 W_Q/W_K/W_V，观察 attention weights 如何变化。")


if __name__ == "__main__":
    main()
