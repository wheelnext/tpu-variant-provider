# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import pathlib
from typing import Final

_GOOGLE_PCI_VENDOR_ID: Final[str] = "0x1ae0"

# TPU PCI device IDs mapping to their corresponding generation.
# Adapted from hardware_utils.py in JAX.
_TPU_PCI_DEVICE_IDS: Final[set[str]] = {
    "0x0027",  # TPU v3
    "0x0056",  # plc
    "0x005e",  # TPU v4
    "0x0062",  # TPU v5p
    "0x0063",  # TPU v5e
    "0x006f",  # TPU v6e
    "0x0076",  # TPU7x
}


def is_tpu_present() -> bool:
    """Checks whether a physical TPU is present on the current device.

    Supports the TPU_VARIANT_PROVIDER_FORCE_PRESENT environment variable
    override:
    - If set to "true", "1", or "yes" (case-insensitive), returns True.
    - If set to "false", "0", or "no" (case-insensitive), returns False.
    - Otherwise, falls back to scanning PCI devices for Google TPU hardware.

    Returns:
        bool: True if a TPU is present, False otherwise.
    """
    force_present = os.environ.get("TPU_VARIANT_PROVIDER_FORCE_PRESENT")
    if force_present is not None:
        return force_present.casefold() in ("true", "1", "yes")

    # Standard Google TPU PCI device scan
    for vendor_path in glob.glob("/sys/bus/pci/devices/*/vendor"):
        try:
            vendor_path_obj = pathlib.Path(vendor_path)
            vendor_id = vendor_path_obj.read_text().strip()
            if vendor_id != _GOOGLE_PCI_VENDOR_ID:
                continue

            device_path_obj = vendor_path_obj.parent / "device"
            device_id = device_path_obj.read_text().strip()
            if device_id in _TPU_PCI_DEVICE_IDS:
                return True
        except (OSError, PermissionError):
            # Safely ignore permission errors or file reading issues
            continue

    return False
