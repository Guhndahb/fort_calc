#!/usr/bin/env python3
"""
CSV Range Processor Utility
A utility for processing specific line ranges from CSV files with efficient memory usage
and comprehensive error handling.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd


class CSVProcessingError(Exception):
    """Base exception for CSV processing errors."""

    pass


class InvalidRangeError(CSVProcessingError):
    """Raised when invalid line range is provided."""

    pass


class FileAccessError(CSVProcessingError):
    """Raised when file cannot be accessed or read."""

    pass


class CSVRangeProcessor:
    """
    A class for processing specific line ranges from CSV files efficiently.

    This class provides methods to read, process, and save specific ranges
    from CSV files with optimized memory usage and comprehensive error handling.
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize the CSV processor with a file path.

        Args:
            file_path: Path to the CSV file to process

        Raises:
            FileNotFoundError: If the specified file does not exist
            FileAccessError: If the file cannot be accessed
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        if not self.file_path.is_file():
            raise FileAccessError(f"Path is not a file: {self.file_path}")
        if not self.file_path.suffix.lower() == ".csv":
            print(f"Warning: File does not have .csv extension: {self.file_path}")

    def get_total_lines(self) -> int:
        """
        Get the total number of data rows (excluding header).

        Returns:
            int: Total number of data rows in the CSV

        Raises:
            FileAccessError: If the file cannot be read
        """
        try:
            # Use chunksize to avoid loading entire file into memory
            chunk_iter = pd.read_csv(self.file_path, chunksize=10000)
            total_rows = 0
            for chunk in chunk_iter:
                total_rows += len(chunk)
            return total_rows
        except pd.errors.EmptyDataError:
            return 0
        except Exception as e:
            raise FileAccessError(f"Error reading CSV file: {e}")

    def _validate_line_range(
        self, start_line: Optional[int], end_line: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Validate and normalize line range parameters.

        Args:
            start_line: Starting line number (1-indexed, excluding header). If None, starts from first line
            end_line: Ending line number (1-indexed, excluding header). If None, reads to end

        Returns:
            Tuple[int, int]: Validated (start_line, end_line) tuple

        Raises:
            InvalidRangeError: If the range is invalid
        """
        total_lines = self.get_total_lines()

        # Handle None for start_line - default to 1 (first data row)
        if start_line is None:
            start_line = 1
        elif not isinstance(start_line, int) or start_line <= 0:
            raise InvalidRangeError(
                f"Start line must be a positive integer or None, got: {start_line}"
            )

        # Handle None for end_line - default to total_lines (last data row)
        if end_line is None:
            end_line = total_lines
        elif not isinstance(end_line, int) or end_line <= 0:
            raise InvalidRangeError(
                f"End line must be a positive integer or None, got: {end_line}"
            )

        # Validate range
        if start_line > total_lines:
            raise InvalidRangeError(
                f"Start line {start_line} exceeds total data rows {total_lines}"
            )

        if end_line < start_line:
            raise InvalidRangeError(
                f"End line {end_line} must be greater than or equal to start line {start_line}"
            )

        if end_line > total_lines:
            end_line = total_lines

        return start_line, end_line

    def read_range(
        self,
        start_line: Optional[int],
        end_line: Optional[int] = None,
        include_header: bool = True,
    ) -> pd.DataFrame:
        """
        Read a specific range of lines from the CSV efficiently.

        This method uses pandas skiprows and nrows parameters for memory-efficient
        range reading, avoiding loading the entire file into memory.

        Args:
            start_line: Starting line number (1-indexed, excluding header). If None, starts from first line
            end_line: Ending line number (1-indexed, excluding header). If None, reads to end
            include_header: Whether to include the header row in the result

        Returns:
            pd.DataFrame: DataFrame containing the specified range

        Raises:
            InvalidRangeError: If the range parameters are invalid
            FileAccessError: If the file cannot be read
        """
        try:
            start_line, end_line = self._validate_line_range(start_line, end_line)

            # Number of data rows to read
            n_rows = end_line - start_line + 1

            if include_header:
                # Preserve header row (line 0). Skip only preceding data rows (1..start_line-1).
                skiprows = None if start_line <= 1 else range(1, start_line)
                df = pd.read_csv(
                    self.file_path,
                    header=0,
                    skiprows=skiprows,
                    nrows=n_rows,
                )
            else:
                # Exclude header. Read raw data without column names.
                # Skip header (0) plus preceding data rows up to requested start (1..start_line).
                skiprows = range(0, start_line + 1)
                df = pd.read_csv(
                    self.file_path,
                    header=None,
                    skiprows=skiprows,
                    nrows=n_rows,
                )

            return df

        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception as e:
            raise FileAccessError(f"Error reading CSV range: {e}")

    def save_range(
        self,
        start_line: Optional[int],
        end_line: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        include_header: bool = True,
    ) -> str:
        """
        Save a specific range of lines to a new CSV file.

        Args:
            start_line: Starting line number (1-indexed, excluding header). If None, starts from first line
            end_line: Ending line number (1-indexed, excluding header). If None, reads to end
            output_path: Output file path. Auto-generated if None
            include_header: Whether to include the header row

        Returns:
            str: Path to the saved file

        Raises:
            InvalidRangeError: If the range is invalid
            FileAccessError: If the file cannot be written
        """
        df = self.read_range(start_line, end_line, include_header)

        if df.empty:
            raise InvalidRangeError("No data found in the specified range")

        if output_path is None:
            suffix = f"_lines_{start_line}_to_{end_line or 'end'}"
            output_path = self.file_path.with_stem(f"{self.file_path.stem}{suffix}")

        output_path = Path(output_path)

        try:
            df.to_csv(output_path, index=False)
            print(f"Range saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            raise FileAccessError(f"Error saving CSV file: {e}")

    def get_file_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the CSV file.

        Returns:
            Dict[str, Any]: Dictionary containing file information

        Raises:
            FileAccessError: If the file cannot be read
        """
        try:
            # Read first few rows to get column info
            sample_df = pd.read_csv(self.file_path, nrows=5)

            info = {
                "file_path": str(self.file_path),
                "file_size": self.file_path.stat().st_size,
                "total_rows": self.get_total_lines(),
                "columns": list(sample_df.columns),
                "column_count": len(sample_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
            }

            return info
        except Exception as e:
            raise FileAccessError(f"Error getting file info: {e}")

    def print_range_info(
        self, start_line: Optional[int], end_line: Optional[int] = None
    ) -> None:
        """
        Print information about the specified range.

        Args:
            start_line: Starting line number (1-indexed, excluding header). If None, starts from first line
            end_line: Ending line number (1-indexed, excluding header). If None, reads to end
        """
        try:
            file_info = self.get_file_info()
            start_line, end_line = self._validate_line_range(start_line, end_line)

            print(f"CSV File: {file_info['file_path']}")
            print(f"Total data rows: {file_info['total_rows']}")
            print(f"Columns: {file_info['columns']}")

            rows_to_process = end_line - start_line + 1
            print(f"Processing range: line {start_line} to {end_line}")
            print(f"Rows to process: {rows_to_process}")

        except (InvalidRangeError, FileAccessError) as e:
            print(f"Error: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # No cleanup needed for this class
        pass


def main() -> None:
    """Command-line interface for CSV range processing."""
    parser = argparse.ArgumentParser(
        description="Process specific line ranges from CSV files efficiently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python csv_processor.py data.csv -s 1 -e 100
  python csv_processor.py data.csv -s 50 -o output.csv
  python csv_processor.py data.csv --info -s 1 -e 10
        """,
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
        "--no-header", action="store_true", help="Exclude header row from output"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print information about the range without processing",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        with CSVRangeProcessor(args.csv_file) as processor:
            if args.verbose or args.info:
                processor.print_range_info(args.start, args.end)

            if args.info:
                return

            # Read the range
            df = processor.read_range(
                args.start, args.end, include_header=not args.no_header
            )

            if not df.empty:
                print(f"\nLoaded {len(df)} rows:")
                print(df.head())

                # Save if output path provided
                if args.output:
                    output_path = processor.save_range(
                        args.start,
                        args.end,
                        args.output,
                        include_header=not args.no_header,
                    )
                    if output_path:
                        print(f"Range saved to: {output_path}")
                else:
                    # Ask if user wants to save
                    save_choice = input(
                        "\nSave this range to a new file? (y/n): "
                    ).lower()
                    if save_choice == "y":
                        output_path = processor.save_range(
                            args.start, args.end, include_header=not args.no_header
                        )
                        if output_path:
                            print(f"Range saved to: {output_path}")
            else:
                print("No data found in the specified range")

    except (FileNotFoundError, FileAccessError, InvalidRangeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
