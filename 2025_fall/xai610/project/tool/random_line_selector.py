"""
random_line_selector.py

This script prints a specified number of random lines from a given text file
without loading the entire file into memory. It reads the file twice:
first to count total lines, then to print the selected lines.

Usage:
    python random_line_selector.py <file_path> <line_count>

Example:
    python random_line_selector.py sample.txt 5

Arguments:
    file_path: Path to the input text file.
    line_count: Number of random lines to print.

Raises:
    ValueError: If requested line_count is greater than total lines in the file.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import argparse

# Third-party imports
import numpy as np

# Custom imports


def random_lines_from_file_stream_sorted(file_path: str, l: int):
    """Print l random lines from a text file without loading entire file.

    Reads the file twice: first to count lines, second to print selected lines
    in ascending order to minimize memory usage.

    Args:
        file_path (str): Path to the input text file.
        l (int): Number of lines to randomly select and print.

    Raises:
        ValueError: If l is greater than total number of lines in the file.
    """
    line_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            line_count += 1

    if l > line_count:
        raise ValueError(
            f"Requested {l} lines, but file only contains {line_count} lines."
        )

    selected_indices = np.random.choice(line_count, l, replace=False)
    selected_indices.sort()

    with open(file_path, "r", encoding="utf-8") as f:
        current_target_idx = 0
        target_line_num = selected_indices[current_target_idx]

        for i, line in enumerate(f):
            if i == target_line_num:
                print(line, end="")
                current_target_idx += 1
                if current_target_idx == l:
                    break
                target_line_num = selected_indices[current_target_idx]


def main():
    """Parse command-line arguments and invoke the random line selector."""
    parser = argparse.ArgumentParser(
        description="Print l random lines from a given text file."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the input text file."
    )
    parser.add_argument(
        "line_count", type=int, help="Number of random lines to print."
    )
    args = parser.parse_args()

    np.random.seed(0)

    random_lines_from_file_stream_sorted(args.file_path, args.line_count)


if __name__ == "__main__":
    main()
