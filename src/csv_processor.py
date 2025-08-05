#!/usr/bin/env python3
"""
CSV Range Processor Utility
A utility for processing specific line ranges from CSV files
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


class CSVRangeProcessor:
    """A class for processing specific line ranges from CSV files"""

    def __init__(self, file_path):
        """Initialize with CSV file path"""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

    def get_total_lines(self):
        """Get total number of data rows (excluding header)"""
        try:
            df = pd.read_csv(self.file_path)
            return len(df)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return 0

    def read_range(self, start_line, end_line=None, include_header=True):
        """
        Read a specific range of lines from the CSV

        Args:
            start_line (int): Starting line number (1-indexed, excluding header)
            end_line (int, optional): Ending line number (1-indexed, excluding header)
            include_header (bool): Whether to include the header row

        Returns:
            pd.DataFrame: DataFrame containing the specified range
        """
        try:
            # Read the CSV file
            df = pd.read_csv(self.file_path)

            # Convert to 0-indexed for pandas
            start_idx = max(0, start_line - 1)

            if end_line is None:
                # Read from start_line to end
                result_df = df.iloc[start_idx:]
            else:
                # Read specific range
                end_idx = min(end_line, len(df))
                result_df = df.iloc[start_idx:end_idx]

            return result_df

        except Exception as e:
            print(f"Error reading CSV range: {e}")
            return None

    def save_range(self, start_line, end_line=None, output_path=None):
        """
        Save a specific range of lines to a new CSV file

        Args:
            start_line (int): Starting line number (1-indexed, excluding header)
            end_line (int, optional): Ending line number (1-indexed, excluding header)
            output_path (str, optional): Output file path. Auto-generated if None

        Returns:
            str: Path to the saved file
        """
        df = self.read_range(start_line, end_line)

        if df is None or df.empty:
            print("No data found in the specified range")
            return None

        if output_path is None:
            suffix = f"_lines_{start_line}_to_{end_line or 'end'}"
            output_path = self.file_path.with_suffix(f"{suffix}.csv")

        df.to_csv(output_path, index=False)
        print(f"Range saved to: {output_path}")
        return str(output_path)

    def print_range_info(self, start_line, end_line=None):
        """Print information about the specified range"""
        total_rows = self.get_total_lines()

        print(f"CSV File: {self.file_path}")
        print(f"Total data rows: {total_rows}")

        if start_line > total_rows:
            print(f"Error: Start line {start_line} exceeds total rows {total_rows}")
            return

        actual_end = end_line if end_line else total_rows
        actual_end = min(actual_end, total_rows)

        rows_to_process = actual_end - start_line + 1

        print(f"Processing range: line {start_line} to {actual_end}")
        print(f"Rows to process: {rows_to_process}")


def main():
    """Command-line interface for CSV range processing"""
    parser = argparse.ArgumentParser(
        description="Process specific line ranges from CSV files"
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        required=True,
        help="Starting line number (1-indexed, excluding header)",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        help="Ending line number (1-indexed, excluding header). If not provided, reads to end",
    )
    parser.add_argument("-o", "--output", help="Output file path for saving the range")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print information about the range without processing",
    )

    args = parser.parse_args()

    try:
        processor = CSVRangeProcessor(args.csv_file)

        if args.info:
            processor.print_range_info(args.start, args.end)
            return

        # Print range info
        processor.print_range_info(args.start, args.end)

        # Read the range
        df = processor.read_range(args.start, args.end)

        if df is not None and not df.empty:
            print(f"\nLoaded {len(df)} rows:")
            print(df.head())

            # Save if output path provided
            if args.output:
                output_path = processor.save_range(args.start, args.end, args.output)
            else:
                # Ask if user wants to save
                save_choice = input("\nSave this range to a new file? (y/n): ").lower()
                if save_choice == "y":
                    output_path = processor.save_range(args.start, args.end)
                    if output_path:
                        print(f"Range saved to: {output_path}")
                    else:
                        print("Failed to save the range.")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
