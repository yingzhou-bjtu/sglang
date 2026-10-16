import ast
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from email.parser import Parser
from pathlib import Path
from unittest import mock

from wheel.wheelfile import WheelFile

ROOT = Path(__file__).resolve().parents[3]


class MusaWheelReleaseTest(unittest.TestCase):
    def test_aot_standard_preserves_legacy_build_override(self):
        source = ROOT / "python/sglang/kernels/aot/setup_musa.py"
        assignments = [
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "mcc_flags"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        expression = compile(ast.Expression(assignments[0].value), str(source), "eval")
        for standard in (None, "c++17", "c++20"):
            with self.subTest(standard=standard), mock.patch.dict(os.environ):
                os.environ.pop("SGLANG_MUSA_CXX_STANDARD", None)
                if standard is not None:
                    os.environ["SGLANG_MUSA_CXX_STANDARD"] = standard
                flags = eval(
                    expression,
                    {
                        "os": os,
                        "operator_namespace": "sgl_kernel",
                        "mtgpu_target": "mp_31",
                    },
                )
                self.assertEqual(
                    [flag for flag in flags if flag.startswith("-std=")],
                    ["-std=" + (standard or "c++20")],
                )

    def test_index_isolates_backend_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / "python/sglang/kernels/aot/dist"
            dist.mkdir(parents=True)
            packages = {
                "musa43": "sglang_kernel-0.4.9+musa43",
                "musa53": "sglang_kernel-0.4.9+musa53",
                "musa530": "sglang_kernel-0.4.9+musa530",
                "rocm720": "sglang_kernel-0.4.9+rocm720",
                "rocm1000": "sglang_kernel-0.4.9+rocm1000",
                "cu130": "sglang_kernel-0.4.9+cu130",
                "xpu": "sglang_kernel_xpu-0.4.9",
            }
            for package in packages.values():
                (dist / f"{package}-cp310-abi3-manylinux2014_x86_64.whl").write_bytes(
                    b"index-test-fixture"
                )
            for backend, version in (
                ("musa", "43"),
                ("musa", "53"),
                ("rocm", "720"),
                ("rocm", "1000"),
                ("cuda", "130"),
                ("xpu", ""),
            ):
                with self.subTest(backend=backend, version=version):
                    command = [
                        sys.executable,
                        str(ROOT / "scripts/update_kernel_whl_index.py"),
                        f"--{backend}",
                    ]
                    if version:
                        command.append(version)
                    subprocess.run(command, cwd=root, check=True, capture_output=True)
                    folder = ("cu" if backend == "cuda" else backend) + version
                    package_dir = (
                        "sglang-kernel-xpu" if backend == "xpu" else "sglang-kernel"
                    )
                    index = root / "sgl-whl" / folder / package_dir / "index.html"
                    content = index.read_text()
                    self.assertEqual(content.count("<a href="), 1)
                    self.assertIn(packages[folder], content)
                    digest = hashlib.sha256(b"index-test-fixture").hexdigest()
                    self.assertIn(f"#sha256={digest}", content)
                    tag = "v0.4.9+xpu" if backend == "xpu" else "v0.4.9"
                    self.assertIn(f"/releases/download/{tag}/", content)

    def test_rename_keeps_filename_and_metadata_consistent(self):
        for suffix in ("43", "53"):
            with (
                self.subTest(suffix=suffix),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                wheel = root / "sglang_kernel-0.4.9-cp310-cp310-linux_x86_64.whl"
                info = "sglang_kernel-0.4.9.dist-info"
                with WheelFile(wheel, "w") as archive:
                    archive.writestr(
                        f"{info}/METADATA",
                        b"Metadata-Version: 2.1\nName: sglang-kernel\nVersion: 0.4.9\n",
                    )
                    archive.writestr(
                        f"{info}/WHEEL",
                        b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
                        b"Tag: cp310-cp310-linux_x86_64\n",
                    )
                subprocess.run(
                    [
                        "bash",
                        str(ROOT / "scripts/ci/musa/rename_wheels_musa.sh"),
                        suffix,
                        str(root),
                    ],
                    env={**os.environ, "PYTHON": sys.executable},
                    check=True,
                    capture_output=True,
                )
                version = f"0.4.9+musa{suffix}"
                wheels = list(root.glob("*.whl"))
                self.assertEqual(len(wheels), 1)
                self.assertEqual(
                    wheels[0].name,
                    f"sglang_kernel-{version}-cp310-cp310-manylinux2014_x86_64.whl",
                )
                with WheelFile(wheels[0]) as archive:
                    info = f"sglang_kernel-{version}.dist-info"
                    metadata = Parser().parsestr(
                        archive.read(f"{info}/METADATA").decode()
                    )
                    self.assertEqual(metadata["Version"], version)
                    self.assertIn(
                        b"Tag: cp310-cp310-manylinux2014_x86_64",
                        archive.read(f"{info}/WHEEL"),
                    )


if __name__ == "__main__":
    unittest.main()
