# Analysis Directory

This directory contains analysis scripts and visualization tools for the NEAT framework.

## Contents

- statistical_tests.py - Statistical significance testing
- generate_plots.py - Visualization generation
- performance_analysis/ - Performance analysis tools
- convergence_analysis/ - Convergence analysis scripts

## Usage

To generate analysis visualizations:

```bash
python analysis/generate_plots.py --input results/ --output figures/
```

To run statistical tests:

```bash
python analysis/statistical_tests.py --data results/performance_metrics/
```

## Output

Analysis results and generated plots will be saved to the figures/ directory.
