#!/usr/bin/env python3
"""
十六进制文本 → 二进制 .bin 转换工具（自包含版）

作用：
- 默认读取仓库根目录的 true_random_hex.txt
- 转换为二进制 .bin 并保存到 detection/random_bits/true_random.bin

使用：
  python detection/convert_hex_to_bin.py
  python detection/convert_hex_to_bin.py --input /path/to/file.txt --output /path/to/out.bin
"""

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HEX 文本转 BIN")
    script_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(script_dir)
    default_input = os.path.join(repo_root, "true_random_hex.txt")
    default_output = os.path.join(script_dir, "random_bits", "true_random.bin")
    parser.add_argument("--input", "-i", default=default_input, help="输入 HEX 文本路径，默认仓库根目录 true_random_hex.txt")
    parser.add_argument("--output", "-o", default=default_output, help="输出 BIN 文件路径，默认 detection/random_bits/true_random.bin")
    return parser.parse_args()


def sanitize_hex_text(text: str) -> str:
    """去除 0x/0X 前缀与所有空白，返回纯 16 进制字符串。若长度为奇数，则左侧补 0。"""
    # 快速去除常见前缀
    text = text.replace("0x", "").replace("0X", "")
    # 去掉所有空白（空格、换行、制表符）
    hex_str = "".join(text.split())
    # 奇数长度时补齐
    if len(hex_str) % 2 == 1:
        hex_str = "0" + hex_str
    return hex_str


def hex_text_to_bytes(text: str) -> bytes:
    """把任意换行/空格分隔的 16 进制文本转为原始字节；兼容含 0x 前缀的情况。"""
    hex_str = sanitize_hex_text(text)
    if not hex_str:
        return b""
    # bytes.fromhex 支持无空白的偶数长度字符串
    return bytes.fromhex(hex_str)


def main() -> int:
    args = parse_args()
    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)

    if not os.path.isfile(in_path):
        print(f"[错误] 找不到输入文件：{in_path}")
        print("请确认 --input 路径，或在仓库根目录放置 true_random_hex.txt")
        return 1

    print(f"[读取] {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    data = hex_text_to_bytes(raw)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)

    print(f"[保存] {out_path}")
    print(f"[信息] 字节长度：{len(data)} B")
    print("\n转换完成 ✅ (HEX → BIN)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
