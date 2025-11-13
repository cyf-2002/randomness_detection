#!/usr/bin/env python3
"""
NIST SP 800-22 兼容的二进制文件转文本脚本

功能：
- 递归遍历 random_bin 目录下的所有 .bin 文件
- 将每个字节转换为8位二进制字符串
- 输出为单一长行的 .txt 文件（无换行符），用于 NIST 检测
- 支持自定义输出目录

用法：
    python detection/bin2txt.py                    # 使用默认输出目录 random_txt
    python detection/bin2txt.py --targetdir nist_data  # 输出到 nist_data 目录

输出示例：
    input:  random_bin/test.bin
    output: random_txt/test.txt (默认) 或 nist_data/test.txt (自定义)
    content:
        01001010111100000000111110101010...
"""

import os
import sys
import argparse
from pathlib import Path

def bin_to_nist_txt(bin_file: Path, txt_file: Path):
    """
    将单个 .bin 文件转换为 NIST 兼容的 .txt（单一长行）
    """
    print(f"Converting {bin_file} -> {txt_file}")

    with open(bin_file, 'rb') as f:
        data = f.read()

    # 将所有字节转换为连续的二进制字符串
    binary_str = ''.join(format(byte, '08b') for byte in data)

    # 确保输出目录存在
    txt_file.parent.mkdir(parents=True, exist_ok=True)

    # 写入 .txt 文件（单一长行）
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(binary_str)

    print(f"  -> Generated {len(binary_str)} bits (from {len(data)} bytes)")

def main():
    parser = argparse.ArgumentParser(description="Convert .bin files to NIST-compatible .txt files")
    parser.add_argument('--targetdir', type=str, default='random_txt',
                        help='Target directory for output .txt files (default: random_txt)')

    args = parser.parse_args()

    # 定义源目录和目标目录
    source_dir = Path("random_bin")
    target_dir = Path(args.targetdir)

    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist.")
        sys.exit(1)

    # 递归查找所有 .bin 文件
    bin_files = list(source_dir.rglob("*.bin"))

    if not bin_files:
        print(f"No .bin files found in {source_dir}")
        return

    print(f"Found {len(bin_files)} .bin files to process")
    print(f"Output directory: {target_dir}")

    # 处理每个 .bin 文件
    for bin_file in bin_files:
        # 计算对应的 .txt 文件路径（在目标目录下保持相对路径）
        relative_path = bin_file.relative_to(source_dir)
        txt_file = target_dir / relative_path.with_suffix('.txt')

        try:
            bin_to_nist_txt(bin_file, txt_file)
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")
            continue

    print(f"\nNIST-compatible conversion completed! Files saved to: {target_dir}")

if __name__ == "__main__":
    main()



