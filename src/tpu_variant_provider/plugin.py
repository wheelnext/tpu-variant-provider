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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Final, Protocol, TypeAlias, runtime_checkable

from tpu_variant_provider.detect import is_tpu_present

VariantNamespace: Final[TypeAlias] = str
VariantFeatureName: Final[TypeAlias] = str
VariantFeatureValue: Final[TypeAlias] = str


@runtime_checkable
class VariantPropertyType(Protocol):
    """A protocol for variant properties."""

    @property
    def namespace(self) -> VariantNamespace:
        """Namespace (from plugin)."""
        raise NotImplementedError

    @property
    def feature(self) -> VariantFeatureName:
        """Feature name (within the namespace)."""
        raise NotImplementedError

    @property
    def value(self) -> VariantFeatureValue:
        """Feature value."""
        raise NotImplementedError


@dataclass(frozen=True)
class VariantFeatureConfig:
    name: str

    # Acceptable values in priority order
    values: Sequence[str]
    multi_value: bool = False


class TpuVariantPlugin:
    """A wheelnext variant plugin for detecting TPUs."""

    namespace: Final[str] = "tpu"
    is_build_plugin: Final[bool] = False

    @classmethod
    @cache
    def get_supported_configs(cls) -> Sequence[VariantFeatureConfig]:
        """Returns the supported features detected on the current system.

        Returns:
            Sequence[VariantFeatureConfig]: Sequence containing the detected
            "present" status.
        """
        present = "true" if is_tpu_present() else "false"
        return [
            VariantFeatureConfig(
                name="present",
                values=[present],
            )
        ]

    @classmethod
    @cache
    def get_all_configs(cls) -> Sequence[VariantFeatureConfig]:
        """Returns all possible configurations ever produced by this plugin.

        Returns:
            Sequence[VariantFeatureConfig]: Sequence containing the "present"
            feature schema.
        """
        return [
            VariantFeatureConfig(
                name="present",
                values=["true", "false"],
            )
        ]
