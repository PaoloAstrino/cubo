#!/usr/bin/env python3
"""
CUBO Performance Test Orchestrator

Interactive CLI menu system for running performance tests with dataset/config selection.
Features:
- Colored terminal output with progress bars
- Interactive dataset/config selection
- Test mode selection (retrieval-only, full, ingestion-only)
- Automatic dataset preparation
- Result visualization

Usage:
    python scripts/test_orchestrator.py
    python scripts/test_orchestrator.py --auto --dataset smoke --mode retrieval-only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cubo.utils.logger import logger

# ANSI color codes
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
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'


class TestOrchestrator:
    """Interactive test orchestrator with rich terminal UI."""

    def __init__(self):
        self.manifest_path = Path("configs/datasets_manifest.json")
        self.config_path = Path("configs/benchmark_config.json")
        self.manifest = self._load_manifest()
        self.configs = self._load_configs()

    def _load_manifest(self) -> Dict:
        """Load datasets manifest."""
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: Manifest path: {self.manifest_path}")
        print(f"DEBUG: Manifest absolute path: {self.manifest_path.absolute()}")
        print(f"DEBUG: Manifest exists: {self.manifest_path.exists()}")
        if not self.manifest_path.exists():
            print("DEBUG: Manifest file not found, returning empty dict")
            return {"datasets": {}, "models": {}}
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"DEBUG: Loaded manifest with datasets: {list(data.get('datasets', {}).keys())}")
        return data

    def _load_configs(self) -> Dict:
        """Load benchmark configs."""
        if not self.config_path.exists():
            return {"configs": []}
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def print_banner(self):
        """Print CUBO banner."""
        banner = f"""
{Colors.BG_BLUE}{Colors.BOLD}                                                           {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}   ██████╗██╗   ██╗██████╗  ██████╗                      {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}  ██╔════╝██║   ██║██╔══██╗██╔═══██╗                     {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}  ██║     ██║   ██║██████╔╝██║   ██║                     {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}  ██║     ██║   ██║██╔══██╗██║   ██║                     {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}  ╚██████╗╚██████╔╝██████╔╝╚██████╔╝                     {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}   ╚═════╝ ╚═════╝ ╚═════╝  ╚═════╝                      {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}                                                           {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}     Performance Test Orchestrator v1.0                   {Colors.ENDC}
{Colors.BG_BLUE}{Colors.BOLD}                                                           {Colors.ENDC}
"""
        print(banner)

    def print_section(self, title: str):
        """Print section header."""
        width = 60
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * width}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{title.center(width)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * width}{Colors.ENDC}\n")

    def print_option(self, number: int, text: str, details: str = ""):
        """Print menu option."""
        print(f"{Colors.OKCYAN}{Colors.BOLD}[{number}]{Colors.ENDC} {Colors.OKBLUE}{text}{Colors.ENDC}")
        if details:
            print(f"    {Colors.WARNING}{details}{Colors.ENDC}")

    def print_navigation(self):
        """Print navigation options."""
        print(f"\n{Colors.OKCYAN}[B]{Colors.ENDC} {Colors.OKBLUE}Go back to previous menu{Colors.ENDC}")
        print(f"{Colors.OKCYAN}[M]{Colors.ENDC} {Colors.OKBLUE}Return to main menu{Colors.ENDC}")
        print(f"{Colors.OKCYAN}[0]{Colors.ENDC} {Colors.OKBLUE}Cancel/Exit{Colors.ENDC}")

    def get_input(self, prompt: str, valid_options: Optional[List[str]] = None, allow_navigation: bool = True) -> str:
        """Get user input with validation and navigation support."""
        nav_options = ['b', 'm', '0'] if allow_navigation else []
        
        while True:
            user_input = input(f"{Colors.OKGREEN}{prompt}{Colors.ENDC} ").strip().lower()
            
            # If no valid options specified, accept any input (for free-form input like numbers)
            if valid_options is None:
                if allow_navigation and user_input in nav_options:
                    return user_input.upper()  # Return uppercase for navigation
                return user_input
            
            # If valid options specified, check against them
            all_valid = valid_options + nav_options
            if user_input in all_valid:
                return user_input
            if allow_navigation and user_input in nav_options:
                return user_input.upper()  # Return uppercase for navigation
            print(f"{Colors.FAIL}Invalid option. Please choose from: {', '.join(all_valid)}{Colors.ENDC}")

    def select_datasets(self) -> Tuple[List[Tuple[str, str]], bool]:
        """Interactive dataset selection with navigation."""
        while True:
            self.print_section("SELECT DATASETS")
            
            datasets = self.manifest.get('datasets', {})
            if not datasets:
                print(f"{Colors.FAIL}No datasets found in manifest!{Colors.ENDC}")
                return [], False

            print(f"{Colors.OKBLUE}Available datasets:{Colors.ENDC}\n")
            dataset_list = list(datasets.items())
            
            for idx, (name, info) in enumerate(dataset_list, 1):
                desc = info.get('description', 'No description')
                size = info.get('size_gb', 'Unknown')
                status = self._check_dataset_status(name)
                self.print_option(idx, f"{name} ({size} GB)", f"{desc} - {status}")
            
            print(f"\n{Colors.OKCYAN}[A]{Colors.ENDC} {Colors.OKBLUE}All 3 main datasets (ultradomain, ragbench, beir){Colors.ENDC}")
            print(f"{Colors.OKCYAN}[S]{Colors.ENDC} {Colors.OKBLUE}Select specific datasets{Colors.ENDC}")
            self.print_navigation()
            
            selection = self.get_input("\nChoice (A/S/B/M/0):", ['a', 's']).upper()
            
            if selection == '0':
                return [], False
            elif selection == 'B':
                return [], True  # Go back
            elif selection == 'M':
                return [], True  # Main menu
            
            if selection == 'A':
                # Select the 3 main benchmark datasets
                main_datasets = ['ultradomain', 'ragbench', 'beir']
                selected = []
                for name in main_datasets:
                    if name in datasets:
                        selected.append((name, f"data/{name}"))
                if not selected:
                    print(f"{Colors.FAIL}Main datasets not found in manifest!{Colors.ENDC}")
                    continue
                print(f"{Colors.OKGREEN}Selected: {', '.join(main_datasets)}{Colors.ENDC}")
                return selected, False
            
            if selection == 'S':
                # Manual selection
                selected, went_back = self._select_specific_datasets(dataset_list)
                if went_back:
                    continue
                return selected, False
        
        return [], False

    def _select_specific_datasets(self, dataset_list: List[Tuple[str, Dict]]) -> Tuple[List[Tuple[str, str]], bool]:
        """Select specific datasets with navigation."""
        while True:
            print(f"\n{Colors.OKBLUE}Enter dataset numbers (comma-separated):{Colors.ENDC}")
            self.print_navigation()
            
            numbers = self.get_input("Numbers:")
            
            if numbers == '0':
                return [], True  # Cancel
            elif numbers == 'B':
                return [], True  # Go back
            elif numbers == 'M':
                return [], True  # Main menu
            
            selected = []
            try:
                indices = [int(x.strip()) for x in numbers.split(',')]
                for idx in indices:
                    if 1 <= idx <= len(dataset_list):
                        name = dataset_list[idx - 1][0]
                        selected.append((name, f"data/{name}"))
                    else:
                        print(f"{Colors.FAIL}Invalid number: {idx}{Colors.ENDC}")
                        selected = []
                        break
            except ValueError:
                print(f"{Colors.FAIL}Invalid input format{Colors.ENDC}")
                continue
            
            if selected:
                return selected, False

    def _check_dataset_status(self, dataset_name: str) -> str:
        """Check if dataset is downloaded."""
        dataset_dir = Path("data") / dataset_name
        marker = dataset_dir / ".download_complete"
        
        if marker.exists():
            return f"{Colors.OKGREEN}✓ Downloaded{Colors.ENDC}"
        elif dataset_dir.exists():
            return f"{Colors.WARNING}⚠ Incomplete{Colors.ENDC}"
        else:
            return f"{Colors.FAIL}✗ Not found{Colors.ENDC}"

    def select_configs(self) -> List[str]:
        """Interactive config selection."""
        self.print_section("SELECT RETRIEVAL CONFIGS")
        
        configs = self.configs.get('configs', [])
        if not configs:
            print(f"{Colors.FAIL}No configs found!{Colors.ENDC}")
            return []

        print(f"{Colors.OKBLUE}Available configs:{Colors.ENDC}\n")
        
        for idx, config in enumerate(configs, 1):
            name = config.get('name', f'config_{idx}')
            updates = config.get('config_updates', {})
            backend = updates.get('vector_store_backend', 'N/A')
            self.print_option(idx, name, f"Backend: {backend}")
        
        print(f"\n{Colors.OKCYAN}[A]{Colors.ENDC} {Colors.OKBLUE}Select all{Colors.ENDC}")
        print(f"{Colors.OKCYAN}[0]{Colors.ENDC} {Colors.OKBLUE}Cancel{Colors.ENDC}")
        
        selection = self.get_input("\nSelect configs (comma-separated numbers or 'A' for all):")
        
        if selection == '0':
            return []
        
        if selection.upper() == 'A':
            return [config.get('name') for config in configs]
        
        selected = []
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            for idx in indices:
                if 1 <= idx <= len(configs):
                    selected.append(configs[idx - 1].get('name'))
        except ValueError:
            print(f"{Colors.FAIL}Invalid input{Colors.ENDC}")
            return []
        
        return selected

    def select_test_mode(self) -> str:
        """Select test mode."""
        self.print_section("SELECT TEST MODE")
        
        modes = [
            ("retrieval-only", "Test retrieval performance (fast, no generation)"),
            ("full", "Full RAG pipeline (retrieval + generation)"),
            ("ingestion-only", "Test document ingestion throughput")
        ]
        
        for idx, (mode, desc) in enumerate(modes, 1):
            self.print_option(idx, mode, desc)
        
        selection = self.get_input("\nSelect mode (1-3):", valid_options=['1', '2', '3'])
        return modes[int(selection) - 1][0]

    def configure_test_params(self) -> Dict:
        """Configure test parameters."""
        self.print_section("CONFIGURE TEST PARAMETERS")
        
        print(f"{Colors.OKBLUE}K values for Recall@K (comma-separated, default: 3,5,10):{Colors.ENDC}")
        k_values = self.get_input("K values:") or "3,5,10"
        
        print(f"\n{Colors.OKBLUE}Question limits (leave empty for all):{Colors.ENDC}")
        easy_limit = self.get_input("Easy questions limit:") or None
        medium_limit = self.get_input("Medium questions limit:") or None
        hard_limit = self.get_input("Hard questions limit:") or None
        
        print(f"\n{Colors.OKBLUE}Output directory (default: results/benchmark_runs):{Colors.ENDC}")
        output_dir = self.get_input("Output dir:") or "results/benchmark_runs"
        
        print(f"\n{Colors.OKBLUE}Max retries (default: 3):{Colors.ENDC}")
        max_retries = self.get_input("Max retries:") or "3"
        
        print(f"\n{Colors.OKBLUE}Skip existing runs? (y/n, default: n):{Colors.ENDC}")
        skip_existing = self.get_input("Skip existing:").lower() in ['y', 'yes']
        
        return {
            'k_values': k_values,
            'easy_limit': int(easy_limit) if easy_limit else None,
            'medium_limit': int(medium_limit) if medium_limit else None,
            'hard_limit': int(hard_limit) if hard_limit else None,
            'output_dir': output_dir,
            'max_retries': int(max_retries),
            'skip_existing': skip_existing
        }

    def configure_dataset_download(self, dataset_name: str) -> Dict:
        """Configure download options for a dataset."""
        print(f"\n{Colors.HEADER}Configure download for {dataset_name}:{Colors.ENDC}")
        
        dataset_info = self.manifest.get('datasets', {}).get(dataset_name, {})
        size_gb = dataset_info.get('size_gb', 'Unknown')
        
        print(f"{Colors.WARNING}Dataset size: {size_gb} GB{Colors.ENDC}")
        
        # Ask about sample creation
        create_sample = self.get_input(
            f"\nCreate a sample subset? (y/n, recommended for large datasets):",
            valid_options=['y', 'n', 'yes', 'no']
        )
        
        sample_percent = None
        include_ground_truth = True
        
        if create_sample.lower() in ['y', 'yes']:
            print(f"{Colors.OKBLUE}Sample percentage (default: 2):{Colors.ENDC}")
            sample_input = self.get_input("Percent:") or "2"
            try:
                sample_percent = int(float(sample_input))
            except ValueError:
                sample_percent = 2
            
            include_gt = self.get_input(
                "\nInclude ground truth files in sample? (y/n, default: y):",
                valid_options=['y', 'n', 'yes', 'no', '']
            )
            include_ground_truth = include_gt.lower() != 'n'
        
        # Ask about model verification
        verify_models = self.get_input(
            "\nVerify embedding models after download? (y/n, default: y):",
            valid_options=['y', 'n', 'yes', 'no', '']
        )
        verify = verify_models.lower() != 'n'
        
        return {
            'sample_percent': sample_percent,
            'include_ground_truth': include_ground_truth,
            'verify': verify
        }

    def prepare_datasets(self, datasets: List[Tuple[str, str]]) -> bool:
        """Ensure datasets are downloaded with configuration options."""
        self.print_section("DATASET PREPARATION")
        
        missing_datasets = []
        for name, path in datasets:
            dataset_dir = Path(path)
            marker = dataset_dir / ".download_complete"
            if not marker.exists():
                missing_datasets.append(name)
        
        if not missing_datasets:
            print(f"{Colors.OKGREEN}✓ All datasets are ready!{Colors.ENDC}")
            return True
        
        print(f"{Colors.WARNING}Missing datasets: {', '.join(missing_datasets)}{Colors.ENDC}")
        download = self.get_input("\nDownload missing datasets? (y/n):", valid_options=['y', 'n', 'yes', 'no'])
        
        if download.lower() not in ['y', 'yes']:
            return False
        
        # Download each missing dataset with configuration
        for dataset_name in missing_datasets:
            print(f"\n{Colors.HEADER}{Colors.BOLD}=== {dataset_name.upper()} ==={Colors.ENDC}")
            
            # Get download configuration
            config = self.configure_dataset_download(dataset_name)
            
            # Build download command
            cmd = [
                sys.executable,
                'scripts/download_and_prepare.py',
                '--dataset', dataset_name,
                '--skip-existing'
            ]
            
            if config['sample_percent'] is not None:
                cmd.extend(['--sample-percent', str(config['sample_percent'])])
                if config['include_ground_truth']:
                    cmd.append('--include-ground-truth')
                else:
                    cmd.append('--no-ground-truth')
            
            if config['verify']:
                cmd.append('--verify')
            else:
                cmd.append('--no-verify')
            
            print(f"\n{Colors.OKCYAN}Running: {' '.join(cmd)}{Colors.ENDC}\n")
            
            # Confirm before downloading
            confirm = self.get_input(f"Start download for {dataset_name}? (y/n):", 
                                    valid_options=['y', 'n', 'yes', 'no'])
            if confirm.lower() not in ['y', 'yes']:
                print(f"{Colors.WARNING}Skipping {dataset_name}{Colors.ENDC}")
                continue
            
            # Run download
            result = subprocess.run(cmd, cwd=Path.cwd())
            
            if result.returncode != 0:
                print(f"{Colors.FAIL}Failed to download {dataset_name}{Colors.ENDC}")
                retry = self.get_input("\nContinue with other datasets? (y/n):",
                                      valid_options=['y', 'n', 'yes', 'no'])
                if retry.lower() not in ['y', 'yes']:
                    return False
            else:
                print(f"{Colors.OKGREEN}✓ {dataset_name} downloaded successfully!{Colors.ENDC}")
        
        return True

    def run_benchmark(self, datasets: List[Tuple[str, str]], configs: List[str],
                     mode: str, params: Dict) -> bool:
        """Run benchmark with selected parameters."""
        self.print_section("RUNNING BENCHMARK")
        
        # Build command
        cmd = [
            sys.executable,
            'scripts/benchmark_runner.py',
            '--mode', mode,
            '--configs', str(self.config_path),
            '--k-values', params['k_values'],
            '--output-dir', params['output_dir'],
            '--max-retries', str(params['max_retries'])
        ]
        
        # Add datasets
        for name, path in datasets:
            cmd.extend(['--datasets', f"{path}:{name}"])
        
        # Add skip-existing flag
        if params.get('skip_existing'):
            cmd.append('--skip-existing')
        
        # Add question limits (these would need to be added to datasets dict)
        # For now, we'll note this is a limitation
        
        print(f"\n{Colors.OKBLUE}Running command:{Colors.ENDC}")
        print(f"{Colors.WARNING}{' '.join(cmd)}{Colors.ENDC}\n")
        
        confirm = self.get_input("Start benchmark? (y/n):", valid_options=['y', 'n', 'yes', 'no'])
        if confirm.lower() not in ['y', 'yes']:
            return False
        
        # Run benchmark
        result = subprocess.run(cmd, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"\n{Colors.BG_GREEN}{Colors.BOLD} BENCHMARK COMPLETED SUCCESSFULLY {Colors.ENDC}")
            return True
        else:
            print(f"\n{Colors.FAIL}Benchmark failed with exit code {result.returncode}{Colors.ENDC}")
            return False

    def view_results(self, output_dir: str):
        """Offer to generate plots from results."""
        self.print_section("RESULTS")
        
        results_path = Path(output_dir)
        if not results_path.exists():
            print(f"{Colors.FAIL}Results directory not found: {results_path}{Colors.ENDC}")
            return
        
        summary_csv = results_path / "summary.csv"
        if not summary_csv.exists():
            print(f"{Colors.FAIL}No summary.csv found in results{Colors.ENDC}")
            return
        
        print(f"{Colors.OKGREEN}✓ Results saved to: {results_path}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✓ Summary CSV: {summary_csv}{Colors.ENDC}")
        
        generate = self.get_input("\nGenerate plots? (y/n):", valid_options=['y', 'n', 'yes', 'no'])
        if generate.lower() not in ['y', 'yes']:
            return
        
        plot_dir = results_path.parent / "plots"
        cmd = [
            sys.executable,
            'scripts/plot_results.py',
            '--results-dir', str(results_path),
            '--output-dir', str(plot_dir)
        ]
        
        print(f"\n{Colors.OKCYAN}Generating plots...{Colors.ENDC}")
        result = subprocess.run(cmd, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✓ Plots saved to: {plot_dir}{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}Plot generation failed{Colors.ENDC}")

    def main_menu(self) -> bool:
        """Display main menu and handle user choices with navigation."""
        while True:
            self.print_section("CUBO TEST ORCHESTRATOR")
            print(f"{Colors.OKBLUE}Welcome to the CUBO Performance Testing Suite{Colors.ENDC}\n")
            
            self.print_option(1, "Run Performance Tests", "Execute retrieval benchmarks with metrics")
            self.print_option(2, "Download & Prepare Datasets", "Download and prepare benchmark datasets")
            self.print_option(3, "View System Status", "Check system health and configuration")
            self.print_option(4, "Run Diagnostics", "Execute system diagnostics and validation")
            
            print(f"\n{Colors.OKCYAN}[0]{Colors.ENDC} {Colors.OKBLUE}Exit{Colors.ENDC}")
            
            choice = self.get_input("\nChoice (0-4):", ['0', '1', '2', '3', '4'], allow_navigation=False)
            
            if choice == '0':
                print(f"{Colors.OKGREEN}Goodbye!{Colors.ENDC}")
                return False
            
            if choice == '1':
                if not self.run_tests_menu():
                    continue  # User went back or cancelled
            elif choice == '2':
                if not self.dataset_menu():
                    continue  # User went back or cancelled
            elif choice == '3':
                self.view_system_status()
            elif choice == '4':
                self.run_diagnostics()
        
        return True

    def run_tests_menu(self) -> bool:
        """Run tests menu with navigation."""
        while True:
            self.print_section("RUN PERFORMANCE TESTS")
            
            self.print_option(1, "Quick Test", "Run basic retrieval test on available data")
            self.print_option(2, "Full Benchmark", "Run comprehensive benchmark suite")
            self.print_option(3, "Custom Test", "Configure custom test parameters")
            
            self.print_navigation()
            
            choice = self.get_input("\nChoice (1-3/B/M/0):", ['1', '2', '3']).upper()
            
            if choice == '0':
                return False
            elif choice == 'B':
                return True  # Go back to main menu
            elif choice == 'M':
                return False  # Return to main menu
            
            if choice == '1':
                if not self.quick_test_menu():
                    continue
            elif choice == '2':
                if not self.full_benchmark_menu():
                    continue
            elif choice == '3':
                if not self.custom_test_menu():
                    continue
        
        return True

    def dataset_menu(self) -> bool:
        """Dataset management menu with navigation."""
        while True:
            self.print_section("DATASET MANAGEMENT")
            
            self.print_option(1, "Download Datasets", "Download and prepare benchmark datasets")
            self.print_option(2, "Verify Datasets", "Check dataset integrity and status")
            self.print_option(3, "Clean Datasets", "Remove downloaded datasets")
            
            self.print_navigation()
            
            choice = self.get_input("\nChoice (1-3/B/M/0):", ['1', '2', '3']).upper()
            
            if choice == '0':
                return False
            elif choice == 'B':
                return True  # Go back to main menu
            elif choice == 'M':
                return False  # Return to main menu
            
            if choice == '1':
                datasets, went_back = self.select_datasets()
                if went_back:
                    continue
                if datasets:
                    self.download_datasets(datasets)
            elif choice == '2':
                self.verify_datasets()
            elif choice == '3':
                self.clean_datasets()
        
        return True

    def quick_test_menu(self) -> bool:
        """Quick test configuration with navigation."""
        while True:
            self.print_section("QUICK TEST CONFIGURATION")
            
            self.print_option(1, "Retrieval Only", "Test retrieval performance only")
            self.print_option(2, "Full Pipeline", "Test complete RAG pipeline")
            
            self.print_navigation()
            
            choice = self.get_input("\nChoice (1-2/B/M/0):", ['1', '2']).upper()
            
            if choice == '0':
                return False
            elif choice == 'B':
                return True  # Go back to run tests menu
            elif choice == 'M':
                return False  # Return to main menu
            
            if choice == '1':
                self.run_quick_test(retrieval_only=True)
            elif choice == '2':
                self.run_quick_test(retrieval_only=False)
            
            return True  # After running test, go back to run tests menu

    def full_benchmark_menu(self) -> bool:
        """Full benchmark configuration with navigation."""
        while True:
            self.print_section("FULL BENCHMARK CONFIGURATION")
            
            self.print_option(1, "Standard Benchmark", "Run standard benchmark suite")
            self.print_option(2, "Extended Benchmark", "Run extended benchmark with additional metrics")
            
            self.print_navigation()
            
            choice = self.get_input("\nChoice (1-2/B/M/0):", ['1', '2']).upper()
            
            if choice == '0':
                return False
            elif choice == 'B':
                return True  # Go back to run tests menu
            elif choice == 'M':
                return False  # Return to main menu
            
            if choice == '1':
                self.run_full_benchmark(extended=False)
            elif choice == '2':
                self.run_full_benchmark(extended=True)
            
            return True  # After running benchmark, go back to run tests menu

    def custom_test_menu(self) -> bool:
        """Custom test configuration with navigation."""
        while True:
            self.print_section("CUSTOM TEST CONFIGURATION")
            
            self.print_option(1, "Configure Parameters", "Set custom test parameters")
            self.print_option(2, "Load Preset", "Load predefined test configuration")
            
            self.print_navigation()
            
            choice = self.get_input("\nChoice (1-2/B/M/0):", ['1', '2']).upper()
            
            if choice == '0':
                return False
            elif choice == 'B':
                return True  # Go back to run tests menu
            elif choice == 'M':
                return False  # Return to main menu
            
            if choice == '1':
                self.configure_custom_test()
            elif choice == '2':
                self.load_test_preset()
            
            return True  # After configuration, go back to run tests menu

    def run_interactive(self):
        """Run interactive menu system."""
        self.print_banner()
        
        try:
            self.main_menu()
            print(f"\n{Colors.HEADER}{Colors.BOLD}Thank you for using CUBO Test Orchestrator!{Colors.ENDC}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Interrupted by user. Exiting.{Colors.ENDC}\n")
        except Exception as e:
            print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}\n")

    # Placeholder methods for menu system - implement as needed
    def run_quick_test(self, retrieval_only: bool):
        """Run quick test."""
        print(f"{Colors.OKCYAN}Running quick test (retrieval_only={retrieval_only})...{Colors.ENDC}")
        print(f"{Colors.WARNING}Quick test implementation pending{Colors.ENDC}")

    def run_full_benchmark(self, extended: bool):
        """Run full benchmark."""
        print(f"{Colors.OKCYAN}Running full benchmark (extended={extended})...{Colors.ENDC}")
        print(f"{Colors.WARNING}Full benchmark implementation pending{Colors.ENDC}")

    def configure_custom_test(self):
        """Configure custom test."""
        print(f"{Colors.OKCYAN}Custom test configuration...{Colors.ENDC}")
        print(f"{Colors.WARNING}Custom test configuration pending{Colors.ENDC}")

    def load_test_preset(self):
        """Load test preset."""
        print(f"{Colors.OKCYAN}Loading test preset...{Colors.ENDC}")
        print(f"{Colors.WARNING}Test preset loading pending{Colors.ENDC}")

    def download_datasets(self, datasets: List[Tuple[str, str]]):
        """Download datasets."""
        print(f"{Colors.OKCYAN}Downloading datasets: {[name for name, _ in datasets]}{Colors.ENDC}")
        self.prepare_datasets(datasets)

    def verify_datasets(self):
        """Verify datasets."""
        print(f"{Colors.OKCYAN}Verifying datasets...{Colors.ENDC}")
        datasets = self.manifest.get('datasets', {})
        for name in datasets:
            status = self._check_dataset_status(name)
            print(f"  {name}: {status}")

    def clean_datasets(self):
        """Clean datasets."""
        print(f"{Colors.WARNING}Dataset cleaning not implemented yet{Colors.ENDC}")

    def view_system_status(self):
        """View system status."""
        self.print_section("SYSTEM STATUS")
        print(f"{Colors.OKGREEN}System status check pending{Colors.ENDC}")

    def run_diagnostics(self):
        """Run diagnostics."""
        self.print_section("DIAGNOSTICS")
        print(f"{Colors.OKGREEN}Running diagnostics...{Colors.ENDC}")
        print(f"{Colors.WARNING}Diagnostics implementation pending{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="CUBO Performance Test Orchestrator - Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--auto', action='store_true',
                       help='Skip interactive mode and run with provided args')
    parser.add_argument('--dataset', '--datasets', dest='datasets', action='append',
                       help='Dataset name(s) for auto mode')
    parser.add_argument('--mode', choices=['retrieval-only', 'full', 'ingestion-only'],
                       default='retrieval-only', help='Test mode for auto mode')
    parser.add_argument('--configs', help='Config file path for auto mode')
    parser.add_argument('--k-values', default='3,5,10', help='K values for auto mode')
    parser.add_argument('--output', default='results/benchmark_runs',
                       help='Output directory for auto mode')
    
    args = parser.parse_args()
    
    orchestrator = TestOrchestrator()
    
    if args.auto:
        # Auto mode - run directly with provided args
        if not args.datasets:
            print(f"{Colors.FAIL}--dataset required in auto mode{Colors.ENDC}")
            sys.exit(1)
        
        datasets = [(d, f"data/{d}") for d in args.datasets]
        configs = ['hybrid_default']  # Default config
        params = {
            'k_values': args.k_values,
            'easy_limit': None,
            'medium_limit': None,
            'hard_limit': None,
            'output_dir': args.output,
            'max_retries': 3,
            'skip_existing': True
        }
        
        if orchestrator.prepare_datasets(datasets):
            orchestrator.run_benchmark(datasets, configs, args.mode, params)
    else:
        # Interactive mode
        orchestrator.run_interactive()


if __name__ == '__main__':
    main()
