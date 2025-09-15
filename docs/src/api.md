# API Reference

## Main Functions

```@docs
normalize_labels
```

```@docs
compute_normalization_stats
```

```@docs
apply_normalization
```

```@docs
denormalize_labels
```

## Function Index

```@index
```

## Internal Implementation Details

The package is organized into several internal modules for different aspects of label normalization:

- **Clipping**: Handles outlier detection and clipping based on quantiles
- **Methods**: Implements different normalization algorithms (min-max, z-score)
- **Statistics**: Computes and stores normalization statistics
- **Core**: Main API functions that orchestrate the normalization process

For details on the internal implementation, please refer to the source code in the package repository.
