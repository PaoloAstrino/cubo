#!/usr/bin/env python3
"""
Benchmark Readiness Validator

Validates that all 3 datasets (BEIR, UltraDomain, RAGBench) are ready 
for overnight benchmark evaluation with ground truth and metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path


class DatasetValidator:
    """Validates dataset readiness for benchmarking."""
    
    def __init__(self):
        self.results = {
            "beir": {"ready": False, "issues": []},
            "ultradomain": {"ready": False, "issues": []},
            "ragbench": {"ready": False, "issues": []}
        }
    
    def validate_beir(self, data_dir: str = "data/beir") -> bool:
        """Validate BEIR dataset."""
        print("\n=== Validating BEIR Dataset ===")
        issues = []
        
        # Check corpus
        corpus_file = Path(data_dir) / "corpus.jsonl"
        if not corpus_file.exists():
            issues.append(f"Missing corpus file: {corpus_file}")
        else:
            try:
                with open(corpus_file, encoding='utf-8') as f:
                    line = f.readline()
                    if line:
                        data = json.loads(line)
                        if '_id' not in data or 'text' not in data:
                            issues.append("Corpus format invalid (missing _id or text)")
                        else:
                            print(f"  [OK] Corpus file valid")
                    else:
                        issues.append("Corpus file is empty")
            except Exception as e:
                issues.append(f"Corpus file error: {e}")
        
        # Check questions
        questions_file = Path(data_dir) / "questions.json"
        if not questions_file.exists():
            issues.append(f"Missing questions file: {questions_file}")
        else:
            try:
                with open(questions_file, encoding='utf-8') as f:
                    data = json.load(f)
                    if 'questions' not in data or 'metadata' not in data:
                        issues.append("Questions format invalid")
                    else:
                        total = data['metadata'].get('total_questions', 0)
                        print(f"  [OK] Questions file valid ({total} questions)")
            except Exception as e:
                issues.append(f"Questions file error: {e}")
        
        # Check ground truth
        gt_file = Path(data_dir) / "ground_truth.json"
        if not gt_file.exists():
            issues.append(f"Missing ground truth file: {gt_file}")
        else:
            try:
                with open(gt_file, encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        issues.append("Ground truth format invalid")
                    else:
                        print(f"  [OK] Ground truth file valid ({len(data)} queries)")
            except Exception as e:
                issues.append(f"Ground truth file error: {e}")
        
        self.results["beir"]["issues"] = issues
        self.results["beir"]["ready"] = len(issues) == 0
        
        if issues:
            for issue in issues:
                print(f"  [FAIL] {issue}")
        
        return len(issues) == 0
    
    def validate_ultradomain(self, data_dir: str = "data/ultradomain_processed") -> bool:
        """Validate UltraDomain dataset."""
        print("\n=== Validating UltraDomain Dataset ===")
        issues = []
        
        # Check if processed directory exists
        if not Path(data_dir).exists():
            issues.append(f"Processed directory not found: {data_dir}")
            issues.append("Run: python scripts/prepare_ultradomain_data.py --data-folder data/ultradomain --output-folder data/ultradomain_processed")
        else:
            # Check corpus
            corpus_file = Path(data_dir) / "corpus.jsonl"
            if not corpus_file.exists():
                issues.append(f"Missing corpus file: {corpus_file}")
            else:
                try:
                    line_count = sum(1 for line in open(corpus_file, encoding='utf-8') if line.strip())
                    print(f"  [OK] Corpus file valid ({line_count} documents)")
                except Exception as e:
                    issues.append(f"Corpus file error: {e}")
            
            # Check questions
            questions_file = Path(data_dir) / "questions.json"
            if not questions_file.exists():
                issues.append(f"Missing questions file: {questions_file}")
            else:
                try:
                    with open(questions_file, encoding='utf-8') as f:
                        data = json.load(f)
                        total = data['metadata'].get('total_questions', 0)
                        print(f"  [OK] Questions file valid ({total} questions)")
                except Exception as e:
                    issues.append(f"Questions file error: {e}")
            
            # Check ground truth
            gt_file = Path(data_dir) / "ground_truth.json"
            if not gt_file.exists():
                issues.append(f"Missing ground truth file: {gt_file}")
            else:
                try:
                    with open(gt_file, encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"  [OK] Ground truth file valid ({len(data)} queries)")
                except Exception as e:
                    issues.append(f"Ground truth file error: {e}")
        
        self.results["ultradomain"]["issues"] = issues
        self.results["ultradomain"]["ready"] = len(issues) == 0
        
        if issues:
            for issue in issues:
                print(f"  [FAIL] {issue}")
        
        return len(issues) == 0
    
    def validate_ragbench(self, data_dir: str = "data/ragbench") -> bool:
        """Validate RAGBench dataset."""
        print("\n=== Validating RAGBench Dataset ===")
        issues = []
        
        # Check if directory exists
        if not Path(data_dir).exists():
            issues.append(f"RAGBench directory not found: {data_dir}")
            issues.append("Run: python scripts/download_ragbench.py --config covidqa --split test --output-dir data/ragbench")
        else:
            # Check for parquet files
            parquet_files = list(Path(data_dir).glob("*.parquet"))
            if not parquet_files:
                issues.append("No parquet files found")
                issues.append("Run: python scripts/download_ragbench.py --config covidqa --split test --output-dir data/ragbench")
            else:
                print(f"  [OK] Found {len(parquet_files)} parquet file(s)")
                for pf in parquet_files:
                    print(f"    - {pf.name}")
                
                # Try to load one to verify format
                try:
                    import pandas as pd
                    df = pd.read_parquet(parquet_files[0])
                    
                    # Check required columns
                    required_cols = ['question', 'response', 'documents']
                    missing = [col for col in required_cols if col not in df.columns]
                    if missing:
                        issues.append(f"Missing columns: {missing}")
                    else:
                        print(f"  [OK] Parquet format valid ({len(df)} rows)")
                        print(f"       Columns: {', '.join(df.columns[:5])}...")
                        
                except ImportError:
                    issues.append("Pandas not installed (required for RAGBench)")
                except Exception as e:
                    issues.append(f"Error reading parquet: {e}")
        
        self.results["ragbench"]["issues"] = issues
        self.results["ragbench"]["ready"] = len(issues) == 0
        
        if issues:
            for issue in issues:
                print(f"  [FAIL] {issue}")
        
        return len(issues) == 0
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60 + "\n")
        
        all_ready = True
        for dataset, result in self.results.items():
            status = "READY" if result["ready"] else "NOT READY"
            print(f"{dataset.upper():15} {status}")
            all_ready = all_ready and result["ready"]
        
        print("\n" + "="*60)
        if all_ready:
            print("ALL DATASETS READY FOR OVERNIGHT EVALUATION")
            print("\nYou can now run the full benchmark with:")
            print("  python benchmarks/runner.py --datasets <dataset_paths> --configs <config_file>")
        else:
            print("SOME DATASETS ARE NOT READY")
            print("\nFix the issues above before running overnight evaluation.")
        print("="*60 + "\n")
        
        return all_ready


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark datasets readiness")
    parser.add_argument("--beir-dir", default="data/beir", help="BEIR dataset directory")
    parser.add_argument("--ultradomain-dir", default="data/ultradomain_processed", help="UltraDomain processed directory")
    parser.add_argument("--ragbench-dir", default="data/ragbench", help="RAGBench directory")
    args = parser.parse_args()
    
    validator = DatasetValidator()
    
    # Validate all datasets
    beir_ready = validator.validate_beir(args.beir_dir)
    ultra_ready = validator.validate_ultradomain(args.ultradomain_dir)
    ragbench_ready = validator.validate_ragbench(args.ragbench_dir)
    
    # Print summary
    all_ready = validator.print_summary()
    
    # Exit with error code if not ready
    sys.exit(0 if all_ready else 1)


if __name__ == "__main__":
    main()
