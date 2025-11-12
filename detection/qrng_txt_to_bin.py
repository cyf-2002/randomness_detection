#!/usr/bin/env python3
"""
极简版：读取 `detection/qrng_output` 下的每个 .txt 文件（内容仅由 0/1 与空白组成的比特串），
清除所有空白后按 8 位打包为字节（不足 8 位直接丢弃），输出到 `detection/random_bin/*.bin`。

用法：
  python detection/qrng_txt_to_bin.py
  python detection/qrng_txt_to_bin.py -i detection/qrng_output -o detection/random_bin
"""

import os
import re
import argparse


def bits_to_bytes(bits: str) -> tuple[bytes, int]:
    """将 01 比特串转字节，丢弃尾部不足 8 位；返回 (bytes, dropped_bits)。"""
    bits = re.sub(r"\s+", "", bits)
    if not bits:
        return b"", 0
    rem = len(bits) % 8
    if rem:
        used_len = len(bits) - rem
        bits = bits[:used_len]
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i+8], 2))
    return bytes(out), rem


def convert_dir(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith('.txt'):
            continue
        in_path = os.path.join(input_dir, name)
        if not os.path.isfile(in_path):
            continue
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        # 仅保留 0/1
        cleaned = re.sub(r"[^01]", "", text)
        data, dropped = bits_to_bytes(cleaned)
        out_name = os.path.splitext(name)[0] + '.bin'
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'wb') as f:
            f.write(data)
        if dropped:
            print(f"[OK] {name} -> {out_name} ({len(data)} B, dropped {dropped} bits)")
        else:
            print(f"[OK] {name} -> {out_name} ({len(data)} B)")
    print('完成 ✅')


def parse_args():
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='将纯 01 文本批量转为 .bin')
    parser.add_argument('-i', '--input', default=os.path.join(script_dir, 'qrng_output'), help='输入目录 (默认 detection/qrng_output)')
    parser.add_argument('-o', '--output', default=os.path.join(script_dir, 'random_bin'), help='输出目录 (默认 detection/random_bin)')
    return parser.parse_args()


def main():
    args = parse_args()
    print(f'输入: {os.path.abspath(args.input)}')
    print(f'输出: {os.path.abspath(args.output)}')
    convert_dir(args.input, args.output)


if __name__ == '__main__':
    main()
