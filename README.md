# TPU Variant Provider for WheelNext

[![Wheel Variant](https://img.shields.io/badge/Wheel_Variant_Plugin-TPU-green)](https://wheelnext.dev)

A variant provider plugin for the Wheel Variant proposed standard that enables automatic detection of Google TPU devices to select TPU-optimized Python packages.

## Overview

The TPU Variant Provider Plugin is part of the work conducted under the [WheelNext](https://wheelnext.dev) initiative to "Re-invent the Wheel" and modernize Python package distribution for hardware-accelerated computing.

This package provides the logic to automatically detect Google TPU hardware (v3, v4, v5p, v5e, v6e, TPU7x) attached through PCI, allowing compatible package installers (such as variant-enabled versions of `uv`) to fetch the TPU-optimized wheel.

## Installation

This package is designed to be automatically resolved and installed by variant-aware installers when necessary. **You should not manually `pip install` this package directly.**

## How It Works

### Variant Property Format

Variant Properties emitted by this plugin follow a three-tuple structure:

```
namespace :: feature :: value
```

For this plugin:
- `tpu :: present :: true` - A Google TPU device was detected.
- `tpu :: present :: false` - No Google TPU device was detected.

### Detection Override

This plugin supports the `TPU_VARIANT_PROVIDER_FORCE_PRESENT` environment variable to manually overwrite TPU detection (useful for testing or debugging without physical hardware):

```bash
# Force TPU Present
export TPU_VARIANT_PROVIDER_FORCE_PRESENT=true

# Force TPU Absent
export TPU_VARIANT_PROVIDER_FORCE_PRESENT=false
```

---

## Configuring Your Project (Usage with Wheel Variants)

To use this variant provider in a Python project, define the variant configuration in your `pyproject.toml`:

```toml
[variant.default-priorities]
namespace = ["tpu"]

[variant.providers.tpu]
requires = ["tpu-variant-provider>=0.1.0"]
enable-if = "platform_system == 'Linux'"
plugin-api = "tpu_variant_provider.plugin:TpuVariantPlugin"
```

You can then declare dependencies that conditionally install packages based on the presence of a TPU:

```toml
[project]
name = "my-tpu-enabled-package"
version = "1.0.0"
dependencies = [
    "jax-tpu ; 'tpu::present::true' in variant_properties",
    "jax-cpu ; 'tpu::present::false' in variant_properties",
]
```

---

## Running Tests

Tests are managed and run using `tox`.

### Running Tests with `uv`

To run the test suite across the default Python 3.12 environment:

```bash
uvx --with tox-uv tox -e py312
```

### Running Linting & Style Checks

- **Black Code Formatting Check:**
  ```bash
  uvx --with tox-uv tox -e format
  ```

- **Pyrefly Static Type Checker:**
  ```bash
  uvx --with tox-uv tox -e typecheck
  ```

---

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
