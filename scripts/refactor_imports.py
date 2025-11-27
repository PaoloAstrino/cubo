"""
Script to refactor all 'from src.cubo' imports to 'from cubo' for proper packaging.
This allows the package to work both in development and when installed via pip.
"""

import os
import re
from pathlib import Path


def refactor_imports(root_dir):
    """Refactor all Python files to remove 'src.' prefix from imports."""
    root_path = Path(root_dir)
    python_files = list(root_path.rglob("*.py"))

    # Patterns to replace
    patterns = [
        (r"from src\.cubo", "from cubo"),
        (r"import src\.cubo", "import cubo"),
    ]

    modified_files = []

    for py_file in python_files:
        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply all patterns
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)

            # Only write if content changed
            if content != original_content:
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(content)
                modified_files.append(str(py_file.relative_to(root_path)))
                print(f"✓ Refactored: {py_file.relative_to(root_path)}")

        except Exception as e:
            print(f"✗ Error processing {py_file}: {e}")

    return modified_files


if __name__ == "__main__":
    import sys

    # Get root directory from command line or use current directory
    root_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    print(f"🔧 Refactoring imports in: {root_dir}")
    print("=" * 60)

    modified = refactor_imports(root_dir)

    print("=" * 60)
    print(f"✅ Refactored {len(modified)} files")

    if modified:
        print("\nModified files:")
        for file in modified[:10]:  # Show first 10
            print(f"  - {file}")
        if len(modified) > 10:
            print(f"  ... and {len(modified) - 10} more")
