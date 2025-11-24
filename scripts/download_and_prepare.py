#!/usr/bin/env python3
"""
Dataset and Model Downloader/Preparer for CUBO

Downloads datasets, verifies integrity, creates samples, and ensures models are present.
Supports skip-existing, force re-download, and sample generation.

Usage:
    python scripts/download_and_prepare.py --dataset ultradomain --skip-existing
    python scripts/download_and_prepare.py --dataset ultradomain --sample-percent 2
    python scripts/download_and_prepare.py --models bge-base-en-v1.5 --verify
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cubo.utils.logger import logger

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DatasetDownloader:
    """Handles dataset and model downloads with verification and sampling."""

    def __init__(self, manifest_path: str = "configs/datasets_manifest.json",
                 output_dir: str = "data", skip_existing: bool = False,
                 force: bool = False, verify: bool = True):
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.skip_existing = skip_existing
        self.force = force
        self.verify = verify
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load dataset manifest from JSON."""
        if not self.manifest_path.exists():
            logger.warning(f"Manifest not found: {self.manifest_path}")
            return {"datasets": {}, "models": {}}
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _print_status(self, message: str, color: str = Colors.OKBLUE, bold: bool = False):
        """Print colored status message."""
        prefix = Colors.BOLD if bold else ""
        print(f"{prefix}{color}[CUBO] {message}{Colors.ENDC}")

    def _check_marker(self, dataset_dir: Path) -> Tuple[bool, Optional[Dict]]:
        """Check if download completion marker exists."""
        marker_file = dataset_dir / ".download_complete"
        if marker_file.exists():
            try:
                with open(marker_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return True, metadata
            except Exception:
                return True, None
        return False, None

    def _write_marker(self, dataset_dir: Path, metadata: Dict):
        """Write download completion marker."""
        marker_file = dataset_dir / ".download_complete"
        with open(marker_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        self._print_status(f"Wrote completion marker: {marker_file}", Colors.OKGREEN)

    def _download_file(self, url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """Download file with progress bar and resume support."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if partial download exists
        temp_path = Path(str(dest_path) + ".part")
        resume_header = {}
        initial_pos = 0
        
        if temp_path.exists():
            initial_pos = temp_path.stat().st_size
            resume_header = {'Range': f'bytes={initial_pos}-'}
            self._print_status(f"Resuming download from {initial_pos} bytes", Colors.WARNING)

        try:
            response = requests.get(url, headers=resume_header, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])
            
            mode = 'ab' if initial_pos > 0 else 'wb'
            
            with open(temp_path, mode) as f, tqdm(
                desc=desc,
                initial=initial_pos,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Move completed download to final location
            shutil.move(str(temp_path), str(dest_path))
            self._print_status(f"Downloaded: {dest_path}", Colors.OKGREEN)
            return True
            
        except Exception as e:
            self._print_status(f"Download failed: {e}", Colors.FAIL)
            return False

    def _extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """Extract tar.gz or zip archive."""
        self._print_status(f"Extracting {archive_path.name}...", Colors.OKCYAN)
        
        try:
            if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
                import tarfile
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(path=extract_dir)
            elif archive_path.suffix == '.zip':
                import zipfile
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                self._print_status(f"Unsupported archive format: {archive_path.suffix}", Colors.FAIL)
                return False
            
            self._print_status(f"Extracted to: {extract_dir}", Colors.OKGREEN)
            return True
            
        except Exception as e:
            self._print_status(f"Extraction failed: {e}", Colors.FAIL)
            return False

    def download_dataset(self, dataset_name: str, url: Optional[str] = None) -> bool:
        """Download and prepare a dataset."""
        self._print_status(f"Preparing dataset: {dataset_name}", Colors.HEADER, bold=True)
        
        # Get dataset info from manifest
        dataset_info = self.manifest.get('datasets', {}).get(dataset_name)
        if not dataset_info and not url:
            self._print_status(f"Dataset '{dataset_name}' not found in manifest and no URL provided", Colors.FAIL)
            return False
        
        download_url = url or dataset_info.get('default_url')
        
        # Handle special URL types
        if download_url == "manual":
            return self._handle_manual_download(dataset_name, dataset_info)
        if download_url.startswith('hf://'):
            repo_id = download_url[5:]  # Remove 'hf://' prefix
            # Remove 'datasets/' prefix if present
            if repo_id.startswith('datasets/'):
                repo_id = repo_id[9:]
            return self._download_huggingface_dataset(dataset_name, repo_id, dataset_info)
        if download_url.startswith('huggingface://'):
            repo_id = download_url[14:]  # Remove 'huggingface://' prefix
            return self._download_huggingface_dataset(dataset_name, repo_id, dataset_info)
        
        if download_url == "local":
            self._print_status(f"Dataset '{dataset_name}' is local - skipping download", Colors.OKGREEN)
            return True
        
        dataset_dir = self.output_dir / dataset_name
        
        # Check existing marker
        exists, metadata = self._check_marker(dataset_dir)
        if exists and self.skip_existing:
            self._print_status(f"Dataset exists and skip_existing=True - skipping", Colors.WARNING)
            return True
        
        if exists and not self.force:
            self._print_status(f"Dataset exists. Use --force to re-download", Colors.WARNING)
            return True
        
        # Force re-download
        if self.force and dataset_dir.exists():
            self._print_status(f"Force re-download - removing existing directory", Colors.WARNING)
            shutil.rmtree(dataset_dir)
        
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download archive
        archive_name = Path(urlparse(download_url).path).name
        archive_path = dataset_dir.parent / archive_name
        
        if not self._download_file(download_url, archive_path, desc=f"Downloading {dataset_name}"):
            return False
        
        # Extract if it's an archive
        if archive_path.suffix in ['.gz', '.zip']:
            if not self._extract_archive(archive_path, dataset_dir):
                return False
            # Clean up archive after successful extraction
            archive_path.unlink()
        
        # Verify expected files
        if self.verify and dataset_info:
            expected_files = dataset_info.get('expected_files', [])
            missing_files = []
            for expected_file in expected_files:
                if not (dataset_dir / expected_file).exists():
                    # Check in subdirectories
                    found = list(dataset_dir.rglob(expected_file))
                    if not found:
                        missing_files.append(expected_file)
            
            if missing_files:
                self._print_status(f"Warning: Missing expected files: {missing_files}", Colors.WARNING)
            else:
                self._print_status(f"All expected files verified", Colors.OKGREEN)
            
            # For BeIR, check if files are in a subdirectory and move them up
            if dataset_name == 'beir':
                subdirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if subdirs:
                    self._print_status(f"Reorganizing BeIR files from subdirectory: {subdirs[0].name}", Colors.WARNING)
                    # Move files up if they're in a single subdirectory
                    if len(subdirs) == 1:
                        subdir = subdirs[0]
                        for file_path in subdir.rglob('*'):
                            if file_path.is_file():
                                relative_path = file_path.relative_to(subdir)
                                new_path = dataset_dir / relative_path
                                new_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(file_path), str(new_path))
                        # Remove empty subdirectory
                        shutil.rmtree(subdir)
                        self._print_status(f"BeIR files moved to root directory", Colors.OKGREEN)
        
        # Write completion marker
        marker_metadata = {
            'dataset': dataset_name,
            'url': download_url,
            'timestamp': time.time(),
            'size_gb': dataset_info.get('size_gb') if dataset_info else None,
            'verified': self.verify
        }
        self._write_marker(dataset_dir, marker_metadata)
        
        self._print_status(f"Dataset '{dataset_name}' ready!", Colors.OKGREEN, bold=True)
        return True

    def _handle_manual_download(self, dataset_name: str, dataset_info: Dict) -> bool:
        """Handle manual download datasets by providing instructions."""
        self._print_status(f"Manual download required for {dataset_name}", Colors.WARNING, bold=True)
        
        notes = dataset_info.get('notes', 'Please check documentation for download instructions.')
        self._print_status(f"Instructions: {notes}", Colors.OKBLUE)
        
        dataset_dir = self.output_dir / dataset_name
        
        # Check if user has already downloaded
        if dataset_dir.exists():
            expected_files = dataset_info.get('expected_files', [])
            found_files = []
            for expected_file in expected_files:
                if (dataset_dir / expected_file).exists():
                    found_files.append(expected_file)
            
            if found_files:
                self._print_status(f"Found {len(found_files)} expected files: {found_files}", Colors.OKGREEN)
                if len(found_files) == len(expected_files):
                    # Write completion marker
                    marker_metadata = {
                        'dataset': dataset_name,
                        'source': 'manual',
                        'timestamp': time.time(),
                        'size_gb': dataset_info.get('size_gb') if dataset_info else None,
                        'verified': True
                    }
                    self._write_marker(dataset_dir, marker_metadata)
                    self._print_status(f"Dataset '{dataset_name}' ready!", Colors.OKGREEN, bold=True)
                    return True
                else:
                    self._print_status(f"Missing files: {[f for f in expected_files if f not in found_files]}", Colors.WARNING)
            else:
                self._print_status(f"No expected files found in {dataset_dir}", Colors.WARNING)
        
        self._print_status(f"Please download and extract to: {dataset_dir}", Colors.OKBLUE)
        self._print_status("Run this command again after downloading to verify.", Colors.OKBLUE)
        return False

    def _download_huggingface_dataset(self, dataset_name: str, repo_id: str, dataset_info: Dict) -> bool:
        """Download dataset from HuggingFace Hub."""
        self._print_status(f"Downloading from HuggingFace: {repo_id}", Colors.OKCYAN)
        
        dataset_dir = self.output_dir / dataset_name
        
        # For RAGBench specifically, use snapshot download since it's a dataset repo
        if 'ragbench' in dataset_name.lower():
            return self._download_huggingface_hub_snapshot(dataset_name, repo_id, dataset_info)
        
        # Check if this is a datasets library dataset (has subsets)
        subsets = dataset_info.get('subsets', [])
        if subsets:
            return self._download_huggingface_datasets_library(dataset_name, repo_id, dataset_info)
        else:
            return self._download_huggingface_hub_snapshot(dataset_name, repo_id, dataset_info)
    
    def _download_huggingface_hub_snapshot(self, dataset_name: str, repo_id: str, dataset_info: Dict) -> bool:
        """Download dataset using huggingface_hub snapshot_download."""
        try:
            from huggingface_hub import snapshot_download, HfFileSystem
        except ImportError:
            self._print_status("huggingface_hub not installed. Install with: pip install huggingface_hub", Colors.FAIL)
            return False
        
        dataset_dir = self.output_dir / dataset_name
        
        # Special handling for large datasets like UltraDomain
        if "UltraDomain" in repo_id or dataset_info.get('size_gb', 0) > 1.0:
            return self._download_large_huggingface_dataset(dataset_name, repo_id, dataset_info)
        
        try:
            # Download the entire repository
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(dataset_dir),
                local_dir_use_symlinks=False
            )
            
            self._print_status(f"Downloaded HuggingFace dataset to {dataset_dir}", Colors.OKGREEN)
            
            # Verify expected files
            if self.verify and dataset_info:
                expected_files = dataset_info.get('expected_files', [])
                missing_files = []
                for expected_file in expected_files:
                    if not (dataset_dir / expected_file).exists():
                        missing_files.append(expected_file)
                
                if missing_files:
                    self._print_status(f"Warning: Missing expected files: {missing_files}", Colors.WARNING)
                else:
                    self._print_status(f"All expected files verified", Colors.OKGREEN)
            
            # Write completion marker
            marker_metadata = {
                'dataset': dataset_name,
                'repo_id': repo_id,
                'source': 'huggingface_hub',
                'timestamp': time.time(),
                'size_gb': dataset_info.get('size_gb') if dataset_info else None,
                'verified': self.verify
            }
            self._write_marker(dataset_dir, marker_metadata)
            
            self._print_status(f"Dataset '{dataset_name}' ready!", Colors.OKGREEN, bold=True)
            return True
            
        except Exception as e:
            self._print_status(f"HuggingFace download failed: {e}", Colors.FAIL)
            return False

    def _download_large_huggingface_dataset(self, dataset_name: str, repo_id: str, dataset_info: Dict) -> bool:
        """Download large HuggingFace datasets with batch processing to avoid timeouts."""
        self._print_status(f"Downloading large dataset {repo_id} in batches to avoid timeouts", Colors.WARNING)
        
        try:
            from huggingface_hub import HfFileSystem, hf_hub_download
            import concurrent.futures
            import time
        except ImportError:
            self._print_status("huggingface_hub not installed. Install with: pip install huggingface_hub", Colors.FAIL)
            return False
        
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use HfFileSystem to list all files
            fs = HfFileSystem()
            repo_path = f"datasets/{repo_id}"
            
            # List all files in the repository
            all_files = []
            try:
                for path in fs.glob(f"{repo_path}/**"):
                    if fs.isfile(path):
                        # Remove the repo path prefix to get relative path
                        rel_path = path.replace(f"{repo_path}/", "")
                        # Skip hidden files and metadata
                        if not rel_path.startswith('.') and not rel_path.startswith('_'):
                            all_files.append(rel_path)
            except Exception as e:
                self._print_status(f"Failed to list files: {e}", Colors.FAIL)
                return False
            
            if not all_files:
                self._print_status("No files found in repository", Colors.FAIL)
                return False
            
            self._print_status(f"Found {len(all_files)} files to download", Colors.OKBLUE)
            
            # Filter to expected files if specified
            expected_files = dataset_info.get('expected_files', [])
            if expected_files:
                filtered_files = [f for f in all_files if any(f.endswith(exp) for exp in expected_files)]
                if filtered_files:
                    all_files = filtered_files
                    self._print_status(f"Filtered to {len(all_files)} expected files", Colors.OKBLUE)
            
            # Download files in batches
            batch_size = 5  # Download 5 files at a time
            successful_downloads = 0
            
            for i in range(0, len(all_files), batch_size):
                batch = all_files[i:i + batch_size]
                self._print_status(f"Downloading batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}: {batch}", Colors.OKCYAN)
                
                # Download batch with retries
                for file_path in batch:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            local_path = hf_hub_download(
                                repo_id=repo_id,
                                repo_type="dataset",
                                filename=file_path,
                                local_dir=str(dataset_dir),
                                local_dir_use_symlinks=False
                            )
                            successful_downloads += 1
                            self._print_status(f"✓ Downloaded {file_path}", Colors.OKGREEN)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt  # Exponential backoff
                                self._print_status(f"Failed to download {file_path} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...", Colors.WARNING)
                                time.sleep(wait_time)
                            else:
                                self._print_status(f"Failed to download {file_path} after {max_retries} attempts: {e}", Colors.FAIL)
                
                # Small delay between batches
                if i + batch_size < len(all_files):
                    time.sleep(1)
            
            if successful_downloads == 0:
                self._print_status("No files were successfully downloaded", Colors.FAIL)
                return False
            
            self._print_status(f"Successfully downloaded {successful_downloads}/{len(all_files)} files", Colors.OKGREEN)
            
            # Verify expected files
            if self.verify and dataset_info:
                expected_files = dataset_info.get('expected_files', [])
                missing_files = []
                for expected_file in expected_files:
                    if not (dataset_dir / expected_file).exists():
                        missing_files.append(expected_file)
                
                if missing_files:
                    self._print_status(f"Warning: Missing expected files: {missing_files}", Colors.WARNING)
                else:
                    self._print_status(f"All expected files verified", Colors.OKGREEN)
            
            # Write completion marker
            marker_metadata = {
                'dataset': dataset_name,
                'repo_id': repo_id,
                'source': 'huggingface_hub_batch',
                'files_downloaded': successful_downloads,
                'total_files': len(all_files),
                'timestamp': time.time(),
                'size_gb': dataset_info.get('size_gb') if dataset_info else None,
                'verified': self.verify
            }
            self._write_marker(dataset_dir, marker_metadata)
            
            self._print_status(f"Large dataset '{dataset_name}' ready!", Colors.OKGREEN, bold=True)
            return True
            
        except Exception as e:
            self._print_status(f"Large dataset download failed: {e}", Colors.FAIL)
            return False

    def _download_huggingface_datasets_library(self, dataset_name: str, repo_id: str, dataset_info: Dict) -> bool:
        """Download dataset using datasets library (for datasets with subsets)."""
        self._print_status(f"Using datasets library for {repo_id} with subsets", Colors.OKCYAN)
        
        try:
            from datasets import load_dataset, DownloadMode
            import pandas as pd
        except ImportError:
            self._print_status("datasets library not installed. Install with: pip install datasets", Colors.FAIL)
            return False
        
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        subsets = dataset_info.get('subsets', [])
        if not subsets:
            self._print_status("No subsets specified for datasets library download", Colors.FAIL)
            return False
        
        self._print_status(f"Downloading {len(subsets)} subsets in batches: {subsets[:3]}...", Colors.OKBLUE)
        
        try:
            all_data = {}
            batch_size = 3  # Download in batches of 3 to avoid rate limits
            
            for i in range(0, len(subsets), batch_size):
                batch = subsets[i:i + batch_size]
                self._print_status(f"Processing batch {i//batch_size + 1}/{(len(subsets) + batch_size - 1)//batch_size}: {batch}", Colors.OKCYAN)
                
                for subset in batch:
                    self._print_status(f"Loading subset: {subset}", Colors.OKCYAN)
                    try:
                        # Load the subset with force download
                        dataset = load_dataset(repo_id, subset, download_mode=DownloadMode.FORCE_REDOWNLOAD)
                        all_data[subset] = dataset
                        
                        # Save to disk
                        subset_dir = dataset_dir / subset
                        subset_dir.mkdir(exist_ok=True)
                        
                        for split_name, split_data in dataset.items():
                            if hasattr(split_data, 'to_pandas'):
                                df = split_data.to_pandas()
                                output_file = subset_dir / f"{split_name}.parquet"
                                df.to_parquet(output_file)
                                self._print_status(f"Saved {subset}/{split_name} ({len(df)} rows)", Colors.OKGREEN)
                            else:
                                # Handle other data formats
                                output_file = subset_dir / f"{split_name}.json"
                                with open(output_file, 'w', encoding='utf-8') as f:
                                    # Convert to list of dicts if it's a Dataset
                                    if hasattr(split_data, '__iter__'):
                                        data_list = [item for item in split_data]
                                        json.dump(data_list, f, indent=2, default=str)
                                    else:
                                        json.dump(split_data, f, indent=2, default=str)
                                self._print_status(f"Saved {subset}/{split_name}", Colors.OKGREEN)
                                
                    except Exception as e:
                        self._print_status(f"Failed to load subset {subset}: {e}", Colors.WARNING)
                        continue
                
                # Small delay between batches to be respectful to the server
                if i + batch_size < len(subsets):
                    self._print_status("Waiting 2 seconds before next batch...", Colors.OKBLUE)
                    time.sleep(2)
            
            if not all_data:
                self._print_status("No subsets were successfully downloaded", Colors.FAIL)
                return False
            
            # Write completion marker
            marker_metadata = {
                'dataset': dataset_name,
                'repo_id': repo_id,
                'source': 'datasets_library',
                'subsets': list(all_data.keys()),
                'timestamp': time.time(),
                'size_gb': dataset_info.get('size_gb') if dataset_info else None,
                'verified': self.verify
            }
            self._write_marker(dataset_dir, marker_metadata)
            
            self._print_status(f"Dataset '{dataset_name}' with {len(all_data)} subsets ready!", Colors.OKGREEN, bold=True)
            return True
            
        except Exception as e:
            self._print_status(f"Datasets library download failed: {e}", Colors.FAIL)
            # Fallback to hub snapshot download
            self._print_status("Falling back to HuggingFace Hub snapshot download", Colors.WARNING)
            return self._download_huggingface_hub_snapshot(dataset_name, repo_id, dataset_info)

    def create_sample(self, dataset_name: str, sample_percent: int = 2,
                     include_ground_truth: bool = True) -> bool:
        """Create a percentage sample of a dataset."""
        self._print_status(f"Creating {sample_percent}% sample of '{dataset_name}'", Colors.HEADER, bold=True)
        
        dataset_dir = self.output_dir / dataset_name
        if not dataset_dir.exists():
            self._print_status(f"Dataset directory not found: {dataset_dir}", Colors.FAIL)
            return False
        
        sample_dir = self.output_dir / f"{dataset_name}_sample_{sample_percent}pct"
        if sample_dir.exists():
            if not self.force:
                self._print_status(f"Sample already exists: {sample_dir}", Colors.WARNING)
                return True
            shutil.rmtree(sample_dir)
        
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all data files (exclude JSON metadata)
        data_files = [f for f in dataset_dir.rglob('*.txt') if f.is_file()]
        data_files.extend([f for f in dataset_dir.rglob('*.md') if f.is_file()])
        data_files.extend([f for f in dataset_dir.rglob('*.pdf') if f.is_file()])
        
        if not data_files:
            self._print_status(f"No data files found in {dataset_dir}", Colors.FAIL)
            return False
        
        total_files = len(data_files)
        sample_count = max(1, int(total_files * sample_percent / 100))
        
        self._print_status(f"Total files: {total_files}, Sample size: {sample_count}", Colors.OKBLUE)
        
        # Load ground truth if available
        gt_files = set()
        ground_truth_path = dataset_dir / "ground_truth.json"
        if include_ground_truth and ground_truth_path.exists():
            try:
                with open(ground_truth_path, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                
                # Extract file references from ground truth
                for question_id, doc_ids in gt_data.items():
                    if isinstance(doc_ids, list):
                        for doc_id in doc_ids:
                            # Extract filename from doc_id (e.g., "doc2.txt_chunk_0" -> "doc2.txt")
                            filename = doc_id.split('_chunk_')[0] if '_chunk_' in doc_id else doc_id
                            gt_files.add(filename)
                
                self._print_status(f"Including {len(gt_files)} ground truth files", Colors.OKCYAN)
            except Exception as e:
                self._print_status(f"Could not load ground truth: {e}", Colors.WARNING)
        
        # Select files: ground truth + random sample
        import random
        random.seed(42)  # For reproducibility
        
        gt_file_paths = [f for f in data_files if f.name in gt_files]
        remaining_files = [f for f in data_files if f.name not in gt_files]
        
        additional_count = max(0, sample_count - len(gt_file_paths))
        if additional_count > len(remaining_files):
            additional_count = len(remaining_files)
        
        selected_files = gt_file_paths + random.sample(remaining_files, additional_count)
        
        # Copy selected files
        self._print_status(f"Copying {len(selected_files)} files...", Colors.OKCYAN)
        with tqdm(total=len(selected_files), desc="Copying files", unit="file") as pbar:
            for file_path in selected_files:
                dest_path = sample_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                pbar.update(1)
        
        # Copy questions.json if it exists
        questions_path = dataset_dir / "questions.json"
        if questions_path.exists():
            shutil.copy2(questions_path, sample_dir / "questions.json")
            self._print_status("Copied questions.json", Colors.OKGREEN)
        
        # Copy or filter ground_truth.json
        if ground_truth_path.exists():
            shutil.copy2(ground_truth_path, sample_dir / "ground_truth.json")
            self._print_status("Copied ground_truth.json", Colors.OKGREEN)
        
        # Write sample metadata
        sample_metadata = {
            'source_dataset': dataset_name,
            'sample_percent': sample_percent,
            'total_files': total_files,
            'sample_size': len(selected_files),
            'ground_truth_files': len(gt_file_paths),
            'timestamp': time.time()
        }
        with open(sample_dir / "sample_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(sample_metadata, f, indent=2)
        
        self._print_status(f"Sample created: {sample_dir}", Colors.OKGREEN, bold=True)
        return True

    def verify_model(self, model_name: str) -> bool:
        """Verify that a model is present and functional."""
        self._print_status(f"Verifying model: {model_name}", Colors.HEADER, bold=True)
        
        model_info = self.manifest.get('models', {}).get(model_name)
        if not model_info:
            self._print_status(f"Model '{model_name}' not found in manifest", Colors.WARNING)
            return False
        
        model_dir = Path("models") / model_name
        
        if model_info.get('source') == 'local':
            if model_dir.exists():
                self._print_status(f"Local model found: {model_dir}", Colors.OKGREEN)
                
                # Try to load and verify dimension
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(str(model_dir))
                    actual_dim = model.get_sentence_embedding_dimension()
                    expected_dim = model_info.get('dimension')
                    
                    if expected_dim and actual_dim != expected_dim:
                        self._print_status(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}", Colors.WARNING)
                    else:
                        self._print_status(f"Model verified: {actual_dim}-dimensional embeddings", Colors.OKGREEN)
                    
                    return True
                except Exception as e:
                    self._print_status(f"Could not verify model: {e}", Colors.WARNING)
                    return False
            else:
                self._print_status(f"Local model not found: {model_dir}", Colors.FAIL)
                return False
        
        elif model_info.get('source') == 'huggingface':
            hf_repo = model_info.get('hf_repo')
            if not hf_repo:
                self._print_status("No HuggingFace repo specified", Colors.FAIL)
                return False
            
            self._print_status(f"Downloading from HuggingFace: {hf_repo}", Colors.OKCYAN)
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(hf_repo, cache_folder=str(model_dir.parent))
                actual_dim = model.get_sentence_embedding_dimension()
                self._print_status(f"Model downloaded and verified: {actual_dim}-dimensional", Colors.OKGREEN)
                return True
            except Exception as e:
                self._print_status(f"Download failed: {e}", Colors.FAIL)
                return False
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets/models for CUBO performance testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download UltraDomain dataset
  python scripts/download_and_prepare.py --dataset ultradomain --skip-existing
  
  # Create 2% sample with ground truth
  python scripts/download_and_prepare.py --dataset ultradomain --sample-percent 2
  
  # Verify embedding model
  python scripts/download_and_prepare.py --models embeddinggemma-300m --verify
  
  # Force re-download
  python scripts/download_and_prepare.py --dataset ultradomain --force
        """
    )
    
    parser.add_argument('--dataset', '--datasets', dest='datasets', action='append',
                       help='Dataset name (can be repeated). Format: name or name:url')
    parser.add_argument('--models', dest='models', action='append',
                       help='Model name to verify/download (can be repeated)')
    parser.add_argument('--output', default='data',
                       help='Output directory for datasets (default: data)')
    parser.add_argument('--manifest', default='configs/datasets_manifest.json',
                       help='Path to datasets manifest JSON')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip download if dataset already exists')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if exists')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify expected files after download (default: True)')
    parser.add_argument('--no-verify', action='store_false', dest='verify',
                       help='Skip verification of expected files')
    parser.add_argument('--sample-percent', type=float,
                       help='Create a sample of this percentage (e.g., 2 for 2%%)')
    parser.add_argument('--include-ground-truth', action='store_true', default=True,
                       help='Include ground truth files in sample (default: True)')
    parser.add_argument('--no-ground-truth', action='store_false', dest='include_ground_truth',
                       help='Do not prioritize ground truth files in sample')
    
    args = parser.parse_args()
    
    if not args.datasets and not args.models:
        parser.print_help()
        sys.exit(1)
    
    downloader = DatasetDownloader(
        manifest_path=args.manifest,
        output_dir=args.output,
        skip_existing=args.skip_existing,
        force=args.force,
        verify=args.verify
    )
    
    success = True
    
    # Download datasets
    if args.datasets:
        for dataset_spec in args.datasets:
            # Parse name:url format
            if ':' in dataset_spec and '://' in dataset_spec:
                parts = dataset_spec.split(':', 1)
                dataset_name = parts[0]
                dataset_url = ':'.join(parts[1:])  # Rejoin in case of http://
            else:
                dataset_name = dataset_spec
                dataset_url = None
            
            if not downloader.download_dataset(dataset_name, dataset_url):
                success = False
            
            # Create sample if requested
            if args.sample_percent:
                if not downloader.create_sample(
                    dataset_name,
                    args.sample_percent,
                    args.include_ground_truth
                ):
                    success = False
    
    # Verify models
    if args.models:
        for model_name in args.models:
            if not downloader.verify_model(model_name):
                success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
