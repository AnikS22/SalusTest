#!/usr/bin/env python3
"""Fill paper with actual data we have"""
import json
import re

# Load actual results
with open('../results/salus_results_massive.json') as f:
    salus = json.load(f)

with open('../results/baseline_results_massive.json') as f:
    baseline = json.load(f)

# Read paper
with open('salus_paper.tex', 'r') as f:
    paper = f.read()

# Replace placeholders with actual data
replacements = {
    # SALUS performance
    r'\\textcolor\{red\}\{\[XX\.X\%\]\} recall': f'{salus["recall"]*100:.1f}\\% recall',
    r'\\textcolor\{red\}\{\[XX\.X\%\]\} precision': f'{salus["precision"]*100:.1f}\\% precision',
    r'\\textcolor\{red\}\{\[X\.X\]\}ms inference latency': '\\textcolor{red}{[TBD]}ms inference latency',
    r'\\textcolor\{red\}\{\[XX\.X×\]\} reduction in false alarms': '\\textcolor{red}{[TBD×]} reduction in false alarms',
    r'\\textcolor\{red\}\{\[XX\.X\%\]\} improvement over baselines': f'{((salus["auroc"]/baseline["auroc"]-1)*100):.1f}\\% AUROC improvement',
    r'\\textcolor\{red\}\{\[XX\%\]\} of predictive performance': '\\textcolor{red}{[TBD\\%]} of predictive performance',
}

for pattern, replacement in replacements.items():
    paper = re.sub(pattern, replacement, paper)

# Write back
with open('salus_paper.tex', 'w') as f:
    f.write(paper)

print("✓ Paper updated with actual data")
print(f"  - Recall: {salus['recall']*100:.1f}%")
print(f"  - Precision: {salus['precision']*100:.1f}%")
print(f"  - AUROC: {salus['auroc']:.4f}")
print(f"  - AUROC improvement: {((salus['auroc']/baseline['auroc']-1)*100):.1f}%")
