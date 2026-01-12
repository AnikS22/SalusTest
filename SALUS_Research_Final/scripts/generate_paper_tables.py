"""
Generate Paper-Ready Tables from Experiment Results

Reads experiment results and generates LaTeX and Markdown tables
formatted for paper submission.

Usage:
    python scripts/generate_paper_tables.py --results_dir results --output_dir results/paper_tables
"""

import sys
from pathlib import Path
import argparse
import json
import csv
from typing import Dict, List


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_csv(path: Path) -> List[Dict]:
    """Load CSV file."""
    if path.exists():
        with open(path) as f:
            return list(csv.DictReader(f))
    return []


def format_percentage(value, decimals=1):
    """Format as percentage."""
    try:
        return f"{float(value)*100:.{decimals}f}\\%"
    except:
        return "N/A"


def format_number(value, decimals=3):
    """Format as number."""
    try:
        return f"{float(value):.{decimals}f}"
    except:
        return "N/A"


def generate_method_comparison_table(results: List[Dict], output_dir: Path):
    """Generate Table 1: Method Comparison."""

    # Markdown version
    md_path = output_dir / "table1_method_comparison.md"
    with open(md_path, 'w') as f:
        f.write("# Table 1: Failure Prediction Performance Comparison\n\n")
        f.write("| Method | AUROC | Recall@90%Pr | Precision | False Alarm Rate | Latency (ms) |\n")
        f.write("|--------|-------|--------------|-----------|------------------|-------------|\n")

        for result in results:
            method = result['method']
            auroc = format_number(result['auroc'], 3)
            recall = format_percentage(result['recall'], 1)
            precision = format_percentage(result['precision'], 1)
            far = format_percentage(result['false_alarm_rate'], 2)
            latency = format_number(result.get('inference_time_ms', 0), 2) if result.get('inference_time_ms') else "N/A"

            f.write(f"| {method} | {auroc} | {recall} | {precision} | {far} | {latency} |\n")

    # LaTeX version
    tex_path = output_dir / "table1_method_comparison.tex"
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Failure Prediction Performance Comparison}\n")
        f.write("\\label{tab:method_comparison}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & AUROC & Recall@90\\%Pr & Precision & FAR & Latency (ms) \\\\\n")
        f.write("\\midrule\n")

        for result in results:
            method = result['method'].replace('(ours)', '\\textbf{(ours)}')
            auroc = format_number(result['auroc'], 3)
            recall = format_percentage(result['recall'], 1)
            precision = format_percentage(result['precision'], 1)
            far = format_percentage(result['false_alarm_rate'], 2)
            latency = format_number(result.get('inference_time_ms', 0), 2) if result.get('inference_time_ms') else "---"

            # Bold SALUS row
            if 'SALUS' in result['method']:
                f.write(f"\\textbf{{{method}}} & \\textbf{{{auroc}}} & \\textbf{{{recall}}} & \\textbf{{{precision}}} & \\textbf{{{far}}} & \\textbf{{{latency}}} \\\\\n")
            else:
                f.write(f"{method} & {auroc} & {recall} & {precision} & {far} & {latency} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ Generated Table 1: Method Comparison")
    print(f"  Markdown: {md_path}")
    print(f"  LaTeX: {tex_path}")


def generate_ablation_table(results: List[Dict], output_dir: Path):
    """Generate Table 2: Ablation Study."""

    # Markdown version
    md_path = output_dir / "table2_ablation.md"
    with open(md_path, 'w') as f:
        f.write("# Table 2: Signal Group Ablation Study\n\n")
        f.write("| Signal Group | AUROC | AUROC Drop | Importance |\n")
        f.write("|--------------|-------|------------|------------|\n")

        for result in results:
            ablation = result['ablation']
            auroc = format_number(result['auroc'], 3)
            drop = format_number(result.get('auroc_drop', 0), 3)
            drop_pct = format_percentage(result.get('auroc_drop_pct', 0) / 100, 1)

            # Determine importance
            drop_val = float(result.get('auroc_drop', 0))
            if drop_val < -0.1:
                importance = "**HIGH**"
            elif drop_val < -0.05:
                importance = "Medium"
            else:
                importance = "Low"

            f.write(f"| {ablation} | {auroc} | {drop} ({drop_pct}) | {importance} |\n")

    # LaTeX version
    tex_path = output_dir / "table2_ablation.tex"
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Signal Group Ablation Study}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Signal Group & AUROC & AUROC Drop & Importance \\\\\n")
        f.write("\\midrule\n")

        for result in results:
            ablation = result['ablation'].replace('_', '\\_')
            auroc = format_number(result['auroc'], 3)
            drop = format_number(result.get('auroc_drop', 0), 3)

            # Determine importance
            drop_val = float(result.get('auroc_drop', 0))
            if drop_val < -0.1:
                importance = "\\textbf{HIGH}"
            elif drop_val < -0.05:
                importance = "Medium"
            else:
                importance = "Low"

            # Bold full model row
            if ablation == 'full':
                f.write(f"\\textbf{{{ablation}}} & \\textbf{{{auroc}}} & --- & Baseline \\\\\n")
            else:
                f.write(f"{ablation} & {auroc} & {drop} & {importance} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ Generated Table 2: Ablation Study")
    print(f"  Markdown: {md_path}")
    print(f"  LaTeX: {tex_path}")


def generate_summary_stats(results_dir: Path, output_dir: Path):
    """Generate summary statistics file."""
    summary_path = output_dir / "summary_statistics.txt"

    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SALUS Experiment Summary Statistics\n")
        f.write("="*70 + "\n\n")

        # Method comparison
        comparison_file = results_dir / "method_comparison_detailed.json"
        if comparison_file.exists():
            with open(comparison_file) as fp:
                comparison_data = json.load(fp)

            salus_result = next((r for r in comparison_data if 'SALUS' in r['method']), None)
            if salus_result:
                f.write("SALUS Performance:\n")
                f.write(f"  AUROC: {salus_result['auroc']:.4f}\n")
                f.write(f"  Recall: {salus_result['recall']:.1%}\n")
                f.write(f"  Precision: {salus_result['precision']:.1%}\n")
                f.write(f"  False Alarm Rate: {salus_result['false_alarm_rate']:.2%}\n")
                if 'inference_time_ms' in salus_result:
                    f.write(f"  Inference Latency: {salus_result['inference_time_ms']:.2f} ms\n")
                f.write("\n")

                # Improvements
                if 'auroc_improvement_pct' in salus_result:
                    f.write("Improvements over best baseline:\n")
                    f.write(f"  AUROC: +{salus_result['auroc_improvement_pct']:.1f}%\n")
                    f.write(f"  Recall: +{salus_result['recall_improvement_pct']:.1f}%\n")
                    f.write(f"  False Alarm Reduction: {salus_result['far_reduction']:.1f}× fewer\n")
                    f.write("\n")

        # Ablation results
        ablation_file = results_dir / "ablation_results.csv"
        if ablation_file.exists():
            ablation_data = load_csv(ablation_file)

            f.write("Signal Group Importance:\n")
            for result in ablation_data:
                if result['ablation'].startswith('no_'):
                    group = result['ablation'].replace('no_', '')
                    drop = float(result.get('auroc_drop', 0))
                    f.write(f"  {group}: {drop:.4f} AUROC drop\n")
            f.write("\n")

        f.write("="*70 + "\n")
        f.write("End of Summary\n")
        f.write("="*70 + "\n")

    print(f"✓ Generated summary statistics: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Paper Tables')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/paper_tables',
                       help='Output directory for generated tables')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Generating Paper-Ready Tables")
    print("="*70)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Table 1: Method Comparison
    comparison_file = results_dir / "method_comparison_detailed.json"
    if comparison_file.exists():
        with open(comparison_file) as f:
            comparison_data = json.load(f)
        generate_method_comparison_table(comparison_data, output_dir)
    else:
        print(f"⚠ Method comparison results not found: {comparison_file}")

    # Table 2: Ablation Study
    ablation_file = results_dir / "ablation_results.csv"
    if ablation_file.exists():
        ablation_data = load_csv(ablation_file)
        generate_ablation_table(ablation_data, output_dir)
    else:
        print(f"⚠ Ablation results not found: {ablation_file}")

    # Summary statistics
    generate_summary_stats(results_dir, output_dir)

    print(f"\n{'='*70}")
    print("Table Generation Complete!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  {output_dir}/")
    for file in output_dir.glob("*"):
        print(f"    - {file.name}")
    print()


if __name__ == '__main__':
    main()
