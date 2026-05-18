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

import os
from unittest.mock import MagicMock, patch

import pytest

from tpu_variant_provider import plugin


def _clear_plugin_caches() -> None:
    """Clear caches on cached classmethods to avoid cross-test contamination."""
    cls = plugin.TpuVariantPlugin
    for name in ("get_supported_configs", "get_all_configs"):
        method = getattr(cls, name, None)
        func = getattr(method, "__func__", None)
        clear = getattr(func, "cache_clear", None)
        if callable(clear):
            clear()


def _create_mock_path(path_str: str, vendor_id: str, device_id: str) -> MagicMock:
    mock_inst = MagicMock()

    if path_str.endswith("vendor"):
        mock_inst.read_text.return_value = f"{vendor_id}\n"

    parent_mock = MagicMock()
    mock_inst.parent = parent_mock

    def truediv_side_effect(name: str) -> MagicMock:
        child_mock = MagicMock()
        child_path = os.path.join(os.path.dirname(path_str), name)
        if child_path.endswith("device"):
            child_mock.read_text.return_value = f"{device_id}\n"
        return child_mock

    parent_mock.__truediv__.side_effect = truediv_side_effect
    return mock_inst


def _reset_env() -> None:
    os.environ.pop("TPU_VARIANT_PROVIDER_FORCE_PRESENT", None)


def setup_function() -> None:
    """Per-test setup: reset env and clear caches."""
    _reset_env()
    _clear_plugin_caches()


def teardown_function() -> None:
    """Per-test teardown: reset env and clear caches."""
    _reset_env()
    _clear_plugin_caches()


def test_static_metadata() -> None:
    assert plugin.TpuVariantPlugin.namespace == "tpu"
    assert plugin.TpuVariantPlugin.is_build_plugin is False


def test_get_all_configs() -> None:
    configs = plugin.TpuVariantPlugin.get_all_configs()
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.name == "present"
    assert cfg.values == ["true", "false"]
    assert cfg.multi_value is False


@pytest.mark.parametrize(
    "is_present, expected_value",
    [
        (True, "true"),
        (False, "false"),
    ],
)
@patch("tpu_variant_provider.plugin.is_tpu_present")
def test_get_supported_configs(mock_is_present, is_present, expected_value) -> None:
    mock_is_present.return_value = is_present
    _clear_plugin_caches()

    configs = plugin.TpuVariantPlugin.get_supported_configs()
    assert len(configs) == 1
    assert configs[0].name == "present"
    assert configs[0].values == [expected_value]


def test_env_override(subtests) -> None:
    test_cases = [
        ("true", "true"),
        ("1", "true"),
        ("YES", "true"),
        ("false", "false"),
        ("0", "false"),
        ("NO", "false"),
    ]

    for env_val, expected in test_cases:
        with subtests.test(env_val=env_val):
            os.environ["TPU_VARIANT_PROVIDER_FORCE_PRESENT"] = env_val
            _clear_plugin_caches()
            configs = plugin.TpuVariantPlugin.get_supported_configs()
            assert configs[0].values == [expected]


@pytest.mark.parametrize(
    "vendor_id, device_id, expected_present",
    [
        ("0x1ae0", "0x005e", "true"),  # Valid Google TPU v4
        ("0x8086", "0x005e", "false"),  # Wrong vendor (Intel)
        ("0x1ae0", "0x9999", "false"),  # Wrong device (Google but not TPU)
    ],
)
@patch("tpu_variant_provider.detect.glob.glob")
@patch("tpu_variant_provider.detect.pathlib.Path")
def test_pci_detection(
    mock_path, mock_glob, vendor_id, device_id, expected_present
) -> None:
    mock_glob.return_value = ["/sys/bus/pci/devices/0000:00:04.0/vendor"]
    mock_path.side_effect = lambda p: _create_mock_path(str(p), vendor_id, device_id)
    _clear_plugin_caches()

    configs = plugin.TpuVariantPlugin.get_supported_configs()
    assert configs[0].values == [expected_present]
