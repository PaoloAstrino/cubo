import os
import json
import time
import psutil
import argparse
from pathlib import Path
from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.utils.logger import logger

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def profile_format(format_name, file_path, output_dir):
    logger.info(f"Profiling format: {format_name}")
    start_mem = get_memory_usage()
    start_time = time.time()
    
    ingestor = DeepIngestor(output_dir=output_dir)
    ingestor.ingest([file_path])
    
    end_time = time.time()
    end_mem = get_memory_usage()
    
    results = {
        "format": format_name,
        "duration": end_time - start_time,
        "memory_delta_mb": end_mem - start_mem,
        "peak_mem_mb": end_mem,  # Simplification, should ideally monitor peak
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Profile memory usage across different file formats")
    parser.add_argument("--pdf", type=str, help="Path to a 100MB+ PDF file")
    parser.add_argument("--md", type=str, help="Path to a 100MB+ Markdown file")
    parser.add_argument("--jsonl", type=str, help="Path to a 100MB+ JSONL file")
    parser.add_argument("--output-dir", default="results/profiling", help="Directory for results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    reports = []
    
    if args.pdf:
        reports.append(profile_format("PDF", args.pdf, f"{args.output_dir}/pdf"))
    if args.md:
        reports.append(profile_format("Markdown", args.md, f"{args.output_dir}/md"))
    if args.jsonl:
        reports.append(profile_format("JSONL", args.jsonl, f"{args.output_dir}/jsonl"))
        
    summary_path = os.path.join(args.output_dir, "format_profile_summary.json")
    with open(summary_path, "w") as f:
        json.dump(reports, f, indent=2)
    
    print(f"Profiling complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
