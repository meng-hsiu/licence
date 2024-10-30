import torch

# 這是我用來測試我的一些模組是不是可以用CPU跑
# 強制使用 CPU
device = torch.device('cpu')

# 嘗試創建一個簡單的張量並進行一些操作
try:
    # 在 CPU 上創建一個張量
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = x * 2  # 簡單的運算

    print("Tensor created on CPU:", x)
    print("Result after operation:", y)
except Exception as e:
    print("Error during operation:", e)

