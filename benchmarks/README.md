# Benchmarks Directory

This directory contains benchmark scripts and comparison implementations for the NEAT framework.

## Contents

- `run_comparison.py` - Main benchmark script for comparing NEAT against baseline models
- `baseline_models/` - Implementation of comparison models (GPT-4, Gemini, etc.)
- `evaluation_metrics/` - Performance evaluation and statistical analysis

## Usage

To run benchmarks against all baseline models:

```bash
python benchmarks/run_comparison.py --models all --dataset math_reasoning
```

## Benchmark Results

Detailed benchmark results and analysis can be found in the `results/` directory.
