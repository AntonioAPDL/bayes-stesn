#!/usr/bin/env python3
"""Build and verify a release in an isolated, pinned virtual environment.

The installed suite runs from a temporary directory against the built wheel,
with package-origin assertions before and after collection.  This catches
source-tree imports even when benchmark contract tests inspect sdist files.
Nothing is uploaded or published.  Receipts, JUnit XML and logs are retained
under results/validation; the machine-readable manifest is results/release_manifest.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import traceback
from urllib.parse import urlparse
import venv
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYPI = "https://pypi.org/simple"
PYTORCH_CPU = "https://download.pytorch.org/whl/cpu"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def source_hashes(root):
    paths = []
    for directory in ("src","tests","benchmarks","tools","configs","docs","examples",".github"):
        if (root/directory).exists():
            paths.extend(p for p in (root/directory).rglob("*") if p.is_file()
                         and p.suffix in {".py",".json",".npz",".yaml",".yml",".md"}
                         and "__pycache__" not in p.parts)
    paths.extend(p for p in (root/"report").rglob("*") if p.is_file()
                 and p.suffix in {".py", ".md"} and "__pycache__" not in p.parts)
    paths.extend(p for p in root.glob("requirements-*.lock"))
    paths.extend(root/name for name in ("pyproject.toml","MANIFEST.in","README.md","CONTRIBUTING.md",".gitignore","LICENSE")
                 if (root/name).exists())
    return {str(path.relative_to(root)):sha256(path) for path in sorted(set(paths))}


def locked_versions(*paths):
    """Resolve the repository's exact version locks, including relative -r."""
    pins = {}
    def read(path):
        for line in Path(path).read_text().splitlines():
            line = line.split("#",1)[0].strip()
            if not line:
                continue
            if line.startswith("-r "):
                read(Path(path).parent/line[3:].strip())
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)",line)
            if match is None:
                raise ValueError(f"Unsupported nonexact lock entry in {path}: {line}")
            name, version = match.groups()
            name = re.sub(r"[-_.]+","-",name).lower()
            if name in pins and pins[name] != version:
                raise ValueError(f"Conflicting locked versions for {name}")
            pins[name] = version
    for path in paths:
        read(path)
    return pins


def neural_install_commands(python, work, constraints, validation, pins):
    """Resolve support packages on PyPI; fetch only Torch from its CPU index."""
    support = work/"neural_support.requirements.txt"
    support.write_text("".join(f"{name}=={version}\n" for name,version in sorted(pins.items())
                               if name != "torch"))
    command = [python,"-I","-m","pip","--isolated","install"]
    return [
        ("neural_support_dependencies",command+[
            "--index-url",PYPI,"--only-binary=:all:","--constraint",constraints,
            "--report",validation/"neural_support_install_report.json","-r",support]),
        ("neural_dependencies",command+[
            "--index-url",PYTORCH_CPU,"--force-reinstall","--no-deps","--only-binary=:all:",
            "--constraint",constraints,"--report",validation/"neural_install_report.json",
            f"torch=={pins['torch']}"]),
    ]


def validation_inventory(root, manifest):
    """Seal only receipts and execution artifacts produced by this gate."""
    validation = root/"results/validation"
    paths = {root/step["log"] for step in manifest["steps"]}
    paths.update(root/entry["file"] for entry in manifest["dependency_install_receipts"])
    paths.update((validation/"fresh_environment.json", validation/"installed_roundtrip.json"))
    for prefix in ("source", "installed"):
        if manifest.get(prefix+"_tests", {}).get("skipped_by_explicit_flag"):
            continue
        paths.update((validation/f"{prefix}_pytest_receipt.json", validation/f"{prefix}_pytest.xml"))
    if manifest.get("venv_created_this_run"):
        paths.add(validation/"venv_creation.json")
    for key in ("cli_smoke", "benchmark_smoke"):
        directory = root/manifest[key]["artifact_directory"]
        paths.update(path for path in directory.rglob("*") if path.is_file())
    inventory = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Invalid current release evidence path: {path}")
        inventory.append({"file":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":sha256(path)})
    return inventory


def junit_summary(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag=="testsuite" else list(root.iter("testsuite"))
    result = {key:sum(int(s.get(key,"0")) for s in suites) for key in
              ("tests","failures","errors","skipped")}
    result["passed"] = result["tests"]-result["failures"]-result["errors"]-result["skipped"]
    result["seconds"] = sum(float(s.get("time","0")) for s in suites)
    return result


HARNESS = r'''
def main():
    import importlib.metadata as metadata
    import json
    from pathlib import Path
    import sys
    mode, testroot, expected, receipt, junit, feature_scope = sys.argv[1:]
    testroot, expected = Path(testroot), Path(expected).resolve()
    if mode == "source":
        # multiprocessing.spawn copies the parent's sys.path, including this
        # source origin, before importing the guarded harness in its child.
        sys.path.insert(0, str(testroot / "src"))
    import bayes_stesn
    origin = Path(bayes_stesn.__file__).resolve()
    if not origin.is_relative_to(expected):
        raise RuntimeError(f"Wrong package origin before tests: {origin}")
    import pytest
    class Warnings:
        def __init__(self):
            self.records = []
            self.optional_deselected = []
            self.skipped = []
        def pytest_collection_modifyitems(self, config, items):
            if feature_scope == "neural":
                return
            selected, optional = [], []
            for item in items:
                cls = getattr(item, "cls", None)
                reason = str(getattr(cls, "__unittest_skip_why__", ""))
                legacy_optional = getattr(cls, "__unittest_skip__", False) and "torch" in reason.lower()
                if item.get_closest_marker("neural") is not None or legacy_optional:
                    optional.append(item)
                    self.optional_deselected.append(item.nodeid)
                else:
                    selected.append(item)
            items[:] = selected
            if optional:
                config.hook.pytest_deselected(items=optional)
        def pytest_runtest_logreport(self, report):
            if report.skipped:
                self.skipped.append(report.nodeid)
        def pytest_warning_recorded(self, warning_message, when, nodeid, location):
            self.records.append({"category": warning_message.category.__name__,
                "message": str(warning_message.message), "when": when, "nodeid": nodeid,
                "filename": warning_message.filename, "lineno": warning_message.lineno})
    plugin = Warnings()
    code = int(pytest.main(["-q", "--import-mode=importlib", str(testroot / "tests"),
                           "--rootdir", str(testroot), "--junitxml", junit], plugins=[plugin]))
    origins = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.copy().items()
               if (name == "bayes_stesn" or name.startswith("bayes_stesn.")) and getattr(module, "__file__", None)}
    wrong = {name: path for name, path in origins.items() if not Path(path).is_relative_to(expected)}
    result = {"pytest_exit_code": code, "package_origin": str(origin), "module_origins": origins,
              "source_imports_detected": wrong, "warnings": plugin.records,
              "warning_count": len(plugin.records), "package_version": metadata.version("bayes-stesn"),
              "feature_scope": feature_scope, "optional_deselected": plugin.optional_deselected,
              "unexpected_skipped": plugin.skipped}
    Path(receipt).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if wrong:
        raise RuntimeError(f"Package source contamination: {wrong}")
    if plugin.skipped:
        raise RuntimeError(f"Selected tests must not be skipped: {plugin.skipped}")
    if plugin.records:
        raise RuntimeError(f"Selected tests emitted {len(plugin.records)} unexpected pytest warning(s); see {receipt}")
    return code

if __name__ == "__main__":
    raise SystemExit(main())
'''


ENVIRONMENT_PROBE = r'''
import importlib.metadata as m
import importlib.util
import json, os, platform, site, sys
from pathlib import Path
import bayes_stesn
packages = {d.metadata["Name"]: d.version for d in m.distributions() if d.metadata["Name"]}
cfg = Path(sys.prefix) / "pyvenv.cfg"
result = {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
          "base_prefix": sys.base_prefix, "platform": platform.platform(),
          "user_site_enabled": site.ENABLE_USER_SITE, "sys_path": sys.path,
          "pyvenv_cfg": cfg.read_text(), "packages": dict(sorted(packages.items())),
          "package_origin": str(Path(bayes_stesn.__file__).resolve()),
          "blas_threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
assert sys.prefix != sys.base_prefix
assert "include-system-site-packages = false" in result["pyvenv_cfg"].lower()
assert Path(result["package_origin"]).is_relative_to(Path(sys.prefix))
assert site.ENABLE_USER_SITE is False
external = {d.metadata["Name"]:str(Path(d.locate_file("")).resolve()) for d in m.distributions()
            if not Path(d.locate_file("")).resolve().is_relative_to(Path(sys.prefix))}
assert not external, f"External distributions leaked into the isolated runtime: {external}"
result["external_distribution_locations"] = external
if sys.argv[2] == "neural":
    assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3,12)
    import torch
    assert "+cpu" in torch.__version__ and torch.version.cuda is None
    assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix))
    result["neural"] = {"torch": torch.__version__, "cpu_only": True, "cuda_build": torch.version.cuda}
else:
    assert importlib.util.find_spec("torch") is None, "Core release gate must use an environment without PyTorch"
    result["neural"] = {"installed": False, "optional_tests": "explicitly deselected"}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"isolated": True, "package_origin": result["package_origin"], "packages": len(packages)}))
'''


ROUNDTRIP = r'''
import json
from pathlib import Path
import sys
import numpy as np
from bayes_stesn import FittedModel, PredictiveSample, simulate_predictive
root, receipt = map(Path,sys.argv[1:])
model = FittedModel.load(root / "model.npz")
original = PredictiveSample.load(root / "prediction.npz")
with np.load(root / "evaluation_truth.npz",allow_pickle=False) as data:
    coordinates = data["coordinates"]
repeated = simulate_predictive(model,target_coordinates=coordinates,horizon=10,paths=16,seed=407)
np.testing.assert_array_equal(original.observation,repeated.observation)
np.testing.assert_array_equal(original.latent,repeated.latent)
model.save(root / "model_roundtrip.npz")
loaded = FittedModel.load(root / "model_roundtrip.npz")
third = simulate_predictive(loaded,target_coordinates=coordinates,horizon=10,paths=16,seed=407)
np.testing.assert_array_equal(repeated.observation,third.observation)
repeated.save(root / "prediction_roundtrip.npz")
saved = PredictiveSample.load(root / "prediction_roundtrip.npz")
np.testing.assert_array_equal(repeated.observation,saved.observation)
numeric_only = True
for name in ("model.npz","prediction.npz","model_roundtrip.npz","prediction_roundtrip.npz"):
    with np.load(root / name,allow_pickle=False) as values:
        numeric_only &= all(values[key].dtype.kind != "O" for key in values.files)
assert numeric_only
result = {"passed":True,"observation_bitwise_equal":True,"latent_bitwise_equal":True,
          "safe_numeric_npz_only":True,"shape":list(original.observation.shape),
          "posterior_identity":model.posterior_identity,
          "sampler_initialization":model.sampler.initialize,
          "rhs_tradeoff_scale":model.sampler.rhs_tradeoff_scale}
receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result))
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-venv",type=Path,default=Path("/tmp/bstesn-release-venv"))
    parser.add_argument("--reuse-venv",action="store_true",help="Reuse only an already isolated environment; requirements are revalidated")
    parser.add_argument("--output",type=Path,default=ROOT/"results/release_manifest.json")
    parser.add_argument("--skip-source-tests",action="store_true",help="Use only when rerunning an already documented source gate")
    parser.add_argument("--with-neural",action="store_true",
                        help="Validate every optional neural test using the CPython 3.12 CPU version lock")
    args = parser.parse_args(argv)
    validation = ROOT/"results/validation"
    validation.mkdir(parents=True,exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="bstesn-release-work-"))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH",None)
    environment.pop("PYTHONHOME",None)
    for name in list(environment):
        if name.startswith("PIP_"):
            environment.pop(name)
    environment.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTHONNOUSERSITE="1",
                       PIP_DISABLE_PIP_VERSION_CHECK="1",PIP_CONFIG_FILE=os.devnull)
    started = time.perf_counter()
    manifest = {"schema_version":2,"started_utc":datetime.now(timezone.utc).isoformat(),
                "passed":False,"repository_published":False,"public_repository_url":None,
                "github_ci_executed":False,"github_ci_configured":bool(list((ROOT/".github/workflows").glob("*.y*ml"))),
                "source_hashes":source_hashes(ROOT),"steps":[],"public_package_index":PYPI,
                "with_neural":args.with_neural,"neural_package_index":PYTORCH_CPU if args.with_neural else None,
                "neural_support_package_index":PYPI if args.with_neural else None,
                "test_policy":"All optional tests; zero skips or pytest warnings" if args.with_neural else
                              "Core tests; declared neural markers and optional-PyTorch classes deselected; zero skips or pytest warnings",
                "work_directory":str(work),"fresh_venv":str(args.fresh_venv),
                "claim_scope":"Local tests, built artifacts and isolated installation; no public publication or remote CI claim."}
    harness = work/"pytest_harness.py"
    harness.write_text(HARNESS)

    def run(name,command,cwd=ROOT,timeout=600):
        log = validation/f"{name}.log"
        print(json.dumps({"step":name,"status":"running"}),flush=True)
        then = time.perf_counter()
        with log.open("w",encoding="utf-8") as stream:
            stream.write("COMMAND: "+json.dumps(list(map(str,command)))+"\n")
            stream.flush()
            completed = subprocess.run(list(map(str,command)),cwd=cwd,env=environment,
                stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=timeout)
        record = {"name":name,"exit_code":completed.returncode,"seconds":time.perf_counter()-then,
                  "log":str(log.relative_to(ROOT)),"log_sha256":sha256(log)}
        manifest["steps"].append(record)
        print(json.dumps({"step":name,"status":"passed" if completed.returncode==0 else "failed",
                          "seconds":round(record["seconds"],2)}),flush=True)
        if completed.returncode:
            raise RuntimeError(f"{name} failed; see {log}")

    try:
        if args.with_neural and (sys.implementation.name != "cpython" or sys.version_info[:2] != (3,12)):
            raise RuntimeError("--with-neural requires invoking this checker with CPython 3.12, matching requirements-neural-cpu.lock")
        run("build",[sys.executable,"-m","build","--no-isolation","--outdir",ROOT/"dist"])
        project = tomllib.loads((ROOT/"pyproject.toml").read_text())["project"]
        normalized = project["name"].replace("-","_")
        wheel, = (ROOT/"dist").glob(f"{normalized}-{project['version']}-*.whl")
        sdist, = (ROOT/"dist").glob(f"{normalized}-{project['version']}.tar.gz")
        manifest["artifacts"] = [{"file":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,
                                   "sha256":sha256(path)} for path in (wheel,sdist)]
        extraction = work/"sdist"
        extraction.mkdir()
        with tarfile.open(sdist) as archive:
            archive.extractall(extraction,filter="data")
        extracted, = (p for p in extraction.iterdir() if p.is_dir())
        required = ["requirements-core.lock","requirements-report.lock","requirements-dev.lock",
                    "docs/reproduction.md","tools/release_check.py","benchmarks/controlled_study.py",
                    "tests/fixtures/v3_reference.npz","requirements-neural-cpu.lock",
                    "benchmarks/profile_sparse_workflow.py","tests/test_sparse_workflow.py",
                    "src/bayes_stesn/benchmark/runner.py","CONTRIBUTING.md",
                    "docs/benchmark_validation.md","docs/benchmark_models.md",
                    "docs/benchmark_registry.yaml","configs/benchmark/smoke.yaml",
                    "docs/evidence/v02/generator_validation.json",
                    "docs/assets/v01_forecast_montage.png","tools/render_model_comparison.py",
                    "configs/benchmark/v03/studies/prepare.py",
                    "configs/benchmark/v03/studies/remaining_studies.yaml",
                    "tools/summarize_v03_learning_curves.py","tools/render_v03_study.py",
                    "tools/seal_v03_calibration.py","report/v03/README.md",
                    "configs/benchmark/v03/studies/stress_prepare.py",
                    "configs/benchmark/v03/studies/stress_recipe.yaml",
                    "configs/benchmark/v03/studies/stress_README.md",
                    "tools/summarize_v03_stress.py","tools/check_conditional_prior_v03.py",
                    "tools/audit_decoder_v03.py","tools/summarize_extensions_v03.py",
                    "docs/reproduction_v03.md",
                    "tests/test_v03_study_tools.py"]
        missing = [name for name in required if not (extracted/name).is_file()]
        if missing:
            raise RuntimeError(f"Source distribution is missing required reproducibility files: {missing}")
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            if "bayes_stesn/gaussian.py" not in members or not any(name.endswith("/METADATA") for name in members):
                raise RuntimeError("Wheel is missing package source or distribution metadata")
            for source in (ROOT/"src/bayes_stesn").rglob("*.py"):
                name = str(source.relative_to(ROOT/"src"))
                if name not in members or hashlib.sha256(archive.read(name)).hexdigest() != sha256(source):
                    raise RuntimeError(f"Wheel source missing or changed: {name}")
        manifest["archive_content_gate"] = {"passed":True,"required_sdist_files":required}
        path = args.fresh_venv.resolve()
        if path.exists():
            if not args.reuse_venv:
                raise FileExistsError(f"{path} exists; choose a new location or explicitly pass --reuse-venv")
            cfg = path/"pyvenv.cfg"
            if not cfg.is_file() or "include-system-site-packages = false" not in cfg.read_text().lower():
                raise RuntimeError("The supplied environment is not isolated from system packages")
            manifest["venv_created_this_run"] = False
        else:
            venv.EnvBuilder(with_pip=True,system_site_packages=False).create(path)
            manifest["venv_created_this_run"] = True
            write_json(validation/"venv_creation.json",{"path":str(path),
                "created_utc":datetime.now(timezone.utc).isoformat(),
                "system_site_packages":False,"builder_python":sys.executable})
        python = path/("Scripts/python.exe" if os.name=="nt" else "bin/python")
        executable = path/("Scripts/bstesn.exe" if os.name=="nt" else "bin/bstesn")
        locks = [ROOT/"requirements-dev.lock",ROOT/"requirements-report.lock"]
        if args.with_neural:
            locks.append(ROOT/"requirements-neural-cpu.lock")
        expected_versions = locked_versions(*locks)
        constraints = work/"locked_versions.constraints"
        constraints.write_text("".join(f"{name}=={version}\n" for name,version in sorted(expected_versions.items())))
        run("fresh_dependencies",[python,"-I","-m","pip","--isolated","install","--index-url",PYPI,
            "--only-binary=:all:","--constraint",constraints,"--report",validation/"fresh_install_report.json",
            "-r",ROOT/"requirements-dev.lock","-r",ROOT/"requirements-report.lock"],cwd=work)
        if args.with_neural:
            neural_pins = locked_versions(ROOT/"requirements-neural-cpu.lock")
            for name, command in neural_install_commands(python,work,constraints,validation,neural_pins):
                run(name,command,cwd=work,timeout=1200)
            neural_receipt = json.loads((validation/"neural_install_report.json").read_text())
            torch_urls = [item["download_info"]["url"] for item in neural_receipt.get("install", [])
                          if item.get("metadata", {}).get("name", "").lower() == "torch"]
            if len(torch_urls) != 1 or not (urlparse(torch_urls[0]).hostname or "").endswith(".pytorch.org"):
                raise RuntimeError("CPU Torch wheel receipt does not identify the official pytorch.org download host")
            manifest["torch_wheel_url"] = torch_urls[0]
        run("wheel_install",[python,"-I","-m","pip","--isolated","install","--no-deps","--force-reinstall",wheel],cwd=work)
        run("pip_check",[python,"-I","-m","pip","check"],cwd=work)
        probe = work/"environment_probe.py"
        probe.write_text(ENVIRONMENT_PROBE)
        feature_scope = "neural" if args.with_neural else "core"
        run("fresh_environment",[python,"-I",probe,validation/"fresh_environment.json",feature_scope],cwd=work)
        manifest["fresh_environment"] = json.loads((validation/"fresh_environment.json").read_text())
        installed_versions = {re.sub(r"[-_.]+","-",name).lower():version for name,version in
                              manifest["fresh_environment"]["packages"].items()}
        mismatches = {name:{"expected":version,"actual":installed_versions.get(name)}
                      for name,version in expected_versions.items() if installed_versions.get(name)!=version}
        manifest["locked_versions"] = expected_versions
        manifest["version_lock_mismatches"] = mismatches
        unlocked = sorted(set(installed_versions)-set(expected_versions)-{"bayes-stesn"})
        manifest["unlocked_distributions"] = unlocked
        if mismatches or unlocked:
            raise RuntimeError(f"Fresh environment differs from the version locks: mismatches={mismatches}; unlocked={unlocked}")
        manifest["dependency_install_receipts"] = [{"file":str(p.relative_to(ROOT)),"sha256":sha256(p)}
            for p in [validation/"fresh_install_report.json"]+
                      ([validation/"neural_support_install_report.json",validation/"neural_install_report.json"]
                       if args.with_neural else [])]
        if not args.skip_source_tests:
            run("source_pytest",[python,"-I",harness,"source",ROOT,ROOT/"src",
                validation/"source_pytest_receipt.json",validation/"source_pytest.xml",feature_scope],cwd=work)
            manifest["source_tests"] = {**junit_summary(validation/"source_pytest.xml"),
                **json.loads((validation/"source_pytest_receipt.json").read_text())}
        else:
            manifest["source_tests"] = {"skipped_by_explicit_flag":True}
        run("installed_pytest",[python,"-I",harness,"installed",extracted,path,
            validation/"installed_pytest_receipt.json",validation/"installed_pytest.xml",feature_scope],cwd=work)
        manifest["installed_tests"] = {**junit_summary(validation/"installed_pytest.xml"),
            **json.loads((validation/"installed_pytest_receipt.json").read_text())}
        smoke = validation/"installed_smoke"/work.name
        run("installed_cli_smoke",[executable,"demo","--smoke","--output",smoke],cwd=work)
        summary = json.loads((smoke/"summary.json").read_text())
        manifest["cli_smoke"] = {"passed":True,"smoke_only":summary["smoke_only"],
            "artifact_directory":str(smoke.relative_to(ROOT)),
            "diagnostic_gate_passed":summary["diagnostics"]["passed"],
            "scope":"Execution test; short-chain diagnostic failure is expected and is not convergence."}
        roundtrip = work/"prediction_roundtrip.py"
        roundtrip.write_text(ROUNDTRIP)
        run("installed_roundtrip",[python,"-I",roundtrip,smoke,validation/"installed_roundtrip.json"],cwd=work)
        manifest["safe_roundtrip"] = json.loads((validation/"installed_roundtrip.json").read_text())
        # Every invocation starts a new sealed benchmark. Reusing an old
        # output would either reject a changed build or resume stale jobs.
        benchmark_smoke = validation/"installed_benchmark_smoke"/work.name
        models = "persistence,seasonal_naive,basis_lgssm,separable_stgp,basis_krr,eqesn_rollout"
        if args.with_neural:
            models += ",dcrnn,basis_gru"
        run("installed_benchmark_smoke",[executable,"benchmark","run","--profile","smoke",
            "--models",models,"--scenarios","g1_gaussian_ar","--output",benchmark_smoke],cwd=work)
        benchmark_summary = json.loads((benchmark_smoke/"summary.json").read_text())
        if not benchmark_summary.get("source_stable"):
            raise RuntimeError("Installed benchmark smoke source fingerprint was not stable")
        manifest["benchmark_smoke"] = {"passed":True,"jobs":benchmark_summary["jobs"],
            "artifact_directory":str(benchmark_smoke.relative_to(ROOT)),
            "stage":benchmark_summary["stage"],"comparison_status":benchmark_summary["comparison_status"],
            "scope":"Execution-only benchmark check; no convergence or superiority claim."}
        final_hashes = source_hashes(ROOT)
        changed = sorted(key for key in set(final_hashes)|set(manifest["source_hashes"])
                         if final_hashes.get(key)!=manifest["source_hashes"].get(key))
        manifest["working_tree_changed_during_check"] = changed
        built_scientific = {str(p.relative_to(extracted)):sha256(p) for p in (extracted/"src/bayes_stesn").rglob("*.py")}
        drift = [name for name,digest in built_scientific.items() if not (ROOT/name).is_file() or sha256(ROOT/name)!=digest]
        manifest["built_scientific_source_hashes"] = built_scientific
        manifest["scientific_source_drift_after_build"] = drift
        manifest["github_ci_configured"] = bool(list((ROOT/".github/workflows").glob("*.y*ml")))
        manifest["validation_artifacts"] = validation_inventory(ROOT,manifest)
        manifest["passed"] = not drift and not changed and all(s["exit_code"]==0 for s in manifest["steps"])
        if drift or changed:
            raise RuntimeError("Source changed during release validation; freeze the source and rerun the release gate")
    except Exception as error:
        manifest["passed"] = False
        manifest["error"] = {"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        print(json.dumps({"release_check":"failed","error":str(error)}),flush=True)
    finally:
        manifest["elapsed_seconds"] = time.perf_counter()-started
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output,manifest)
        print(json.dumps({"manifest":str(args.output),"passed":manifest["passed"],
                          "elapsed_seconds":round(manifest["elapsed_seconds"],2)}),flush=True)
    return 0 if manifest["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
