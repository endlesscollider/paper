"""
无第三方依赖的 attention 小实验。

运行：
  python3 code/attention_no_dependencies.py
"""

from math import exp, sqrt


def matmul(A, B):
    rows, inner, cols = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def transpose(A):
    return [list(row) for row in zip(*A)]


def softmax(row):
    m = max(row)
    exps = [exp(x - m) for x in row]
    s = sum(exps)
    return [x / s for x in exps]


def attention(Q, K, V):
    d_k = len(Q[0])
    scores = matmul(Q, transpose(K))
    scores = [[x / sqrt(d_k) for x in row] for row in scores]
    weights = [softmax(row) for row in scores]
    out = matmul(weights, V)
    return out, scores, weights


def fmt(M):
    return "\n".join("[" + ", ".join(f"{x: .3f}" for x in row) + "]" for row in M)


def main():
    X = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
    W_Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    W_K = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]]
    W_V = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]

    Q, K, V = matmul(X, W_Q), matmul(X, W_K), matmul(X, W_V)
    out, scores, weights = attention(Q, K, V)

    print("Q:\n" + fmt(Q))
    print("\nK:\n" + fmt(K))
    print("\nV:\n" + fmt(V))
    print("\nScaled QK^T scores:\n" + fmt(scores))
    print("\nAttention weights:\n" + fmt(weights))
    print("\nOutput:\n" + fmt(out))


if __name__ == "__main__":
    main()
