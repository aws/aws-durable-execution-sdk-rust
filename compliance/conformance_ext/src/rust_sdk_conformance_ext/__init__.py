# SPDX-License-Identifier: Apache-2.0
"""Conformance extension registering Rust-SDK-driven requirement suites.

The language-agnostic conformance runner ships a fixed core requirement
tree. Suites that originate from this SDK's own issue tracker —
multi-page-history replay (issue #5), non-determinism detection
(issue #6), and combinator task-ownership coverage (issue #7) — are
contributed through the runner's public extension entry point instead, so
``--suite history``, ``--suite nondeterminism``, and ``--suite combinator``
become valid the moment this package is installed alongside the runner:

    pip install ./conformance_ext

The requirement YAML files live under ``requirements/`` inside this
package and follow the same schema as the runner's core requirements.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path

from aws_durable_execution_conformance_tests.extensions import RequirementSuite

_REQUIREMENTS_ROOT = Path(str(resources.files("rust_sdk_conformance_ext"))) / "requirements"


class RustSdkConformanceExtension:
    """Registers the Rust SDK's history, nondeterminism, and combinator suites."""

    name = "rust-sdk-conformance-ext"
    requires_core = ">=1.0,<2"

    def requirement_suites(self) -> Sequence[RequirementSuite]:
        """Return the suites this extension contributes."""
        return (
            RequirementSuite(name="history", root=_REQUIREMENTS_ROOT / "history"),
            RequirementSuite(name="nondeterminism", root=_REQUIREMENTS_ROOT / "nondeterminism"),
            RequirementSuite(name="combinator", root=_REQUIREMENTS_ROOT / "combinator"),
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """No extension-owned CLI options."""

    def validate_configuration(self, args: argparse.Namespace) -> None:
        """No extension-owned configuration to validate."""

    def deployment_parameters(self, args: argparse.Namespace) -> Mapping[str, str]:
        """No additional SAM parameters beyond the template's own."""
        return {}

    def deployment_secrets(self, args: argparse.Namespace) -> Mapping[str, str]:
        """No secret SAM parameters."""
        return {}


extension = RustSdkConformanceExtension()
