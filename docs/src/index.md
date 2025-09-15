```@meta
CurrentModule = RealLabelNormalization
```

# RealLabelNormalization.jl

A Julia package for robust normalization of real-valued labels, commonly used in regression tasks. This package provides various normalization methods with built-in outlier handling and NaN support.

## Features

- **Multiple normalization methods**: Min-max and Z-score normalization
- **Flexible normalization modes**: Global or column-wise normalization
- **Robust outlier handling**: Configurable quantile-based clipping
- **NaN handling**: Preserves NaN values while computing statistics on valid data
- **Consistent train/test normalization**: Save statistics from training data and apply to test data

## Quick Start

```julia
using RealLabelNormalization

# Basic min-max normalization to `[-1, 1]` with outlier clipping
labels = [1.0, 5.0, 3.0, 8.0, 2.0, 100.0]  # 100.0 is an outlier
normalized = normalize_labels(labels)

# Z-score normalization
normalized = normalize_labels(labels; method=:zscore)

# Multi-target normalization (matrix input)
# Each row is a sample, each column is a different target variable
labels_matrix = [1.0 10.0; 5.0 20.0; 3.0 15.0; 8.0 25.0]
normalized = normalize_labels(labels_matrix; mode=:columnwise)
```

## Installation

```julia
using Pkg
Pkg.add("RealLabelNormalization")
```

## API Reference

```@index
```

```@autodocs
Modules = [RealLabelNormalization]
```
