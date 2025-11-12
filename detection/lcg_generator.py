#!/usr/bin/env python3
"""
LCG 随机字节生成器（离差归一化，输出 .bin）

说明：
- 一次生成多组模数 (默认 2^26, 2^28, 2^30, 2^32, 2^34) 的随机字节序列
- 对每个序列的 LCG 状态做离差归一化（min-max）到 [0,255]
- 输出写入 detection/random_bits/

用法：
    python detection/lcg_generator.py              # 每个模数生成 5MB
    python detection/lcg_generator.py --mb 10      # 每个模数生成 10MB
    python detection/lcg_generator.py --mod-list 26 28 --mb 1  # 仅 2^26 与 2^28 各 1MB
    python detection/lcg_generator.py --seed 123 --a 1103515245 --c 12345
"""

import os
import argparse

# ========= 配置 =========
MOD_LIST = [26, 28, 30, 32, 34]    # 模数指数列表（可通过 --mod-list 覆盖）
SEED = 20                          # LCG 初始种子
A = 1103513245                     # 经典 LCG 参数 a
C = 12345                          # 经典 LCG 参数 c
OUT_DIR = os.path.join(os.path.dirname(__file__), "random_bits")
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 核心函数 =========
def lcg_bytes(length_bytes: int, m: int, seed: int = SEED, a: int = A, c: int = C) -> bytes:
    """
    生成 length_bytes 个 LCG 状态，并对当前样本做离差归一化（min-max）到 [0,255]，返回字节序列。
    采用两遍流式处理：第一遍统计 min/max，第二遍按相同序列生成并归一化后写入。
    """
    # 第一遍：统计 min/max
    x = seed % m
    min_v = None
    max_v = None
    for _ in range(length_bytes):
        x = (a * x + c) % m
        if min_v is None or x < min_v:
            min_v = x
        if max_v is None or x > max_v:
            max_v = x

    # 退化：序列常数
    if min_v == max_v:
        return bytes([0] * length_bytes)

    # 第二遍：归一化到 [0,255]
    x = seed % m
    out = bytearray()
    scale = 255.0 / (max_v - min_v)
    for _ in range(length_bytes):
        x = (a * x + c) % m
        y = int(round((x - min_v) * scale))
        if y < 0:
            y = 0
        elif y > 255:
            y = 255
        out.append(y)
    return bytes(out)

def save_bin(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

# ========= 主程序 =========
def main():
    p = argparse.ArgumentParser(description="生成多个模数的 LCG 随机 .bin 文件（离差归一化）")
    p.add_argument("--mb", type=int, default=1, help="每个文件生成大小(MB)，默认 5")
    p.add_argument("--mod-list", type=int, nargs="*", default=MOD_LIST, help="模数指数列表，例如: 26 28 30")
    p.add_argument("--seed", type=int, default=SEED, help="LCG 种子")
    p.add_argument("--a", type=int, default=A, help="LCG 参数 a")
    p.add_argument("--c", type=int, default=C, help="LCG 参数 c")
    args = p.parse_args()

    length_bytes = args.mb * 1024 * 1024

    print(f"开始生成，每个文件大小：{args.mb} MB，模数列表：{args.mod_list}，seed={args.seed}\n")
    for mod_exp in args.mod_list:
        modulus = 2 ** mod_exp
        data = lcg_bytes(length_bytes, m=modulus, seed=args.seed, a=args.a, c=args.c)
        out_name = f"lcg_m2e{mod_exp}_seed{args.seed}_{args.mb}MB.bin"
        out_path = os.path.join(OUT_DIR, out_name)
        save_bin(out_path, data)
        print(f"[✔] 模数 2^{mod_exp} 已生成二进制文件: {out_name}")

    print("\n全部生成完成 ✅")

if __name__ == "__main__":
    main()
