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
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import traceback
import venv
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYPI = "https://pypi.org/simple"


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
    for directory in ("src","tests","benchmarks","tools","configs","docs"):
        if (root/directory).exists():
            paths.extend(p for p in (root/directory).rglob("*") if p.is_file()
                         and p.suffix in {".py",".json",".npz",".yaml",".yml",".md"}
                         and "__pycache__" not in p.parts)
    paths.extend(p for p in root.glob("requirements-*.lock"))
    paths.extend(root/name for name in ("pyproject.toml","MANIFEST.in","README.md","LICENSE")
                 if (root/name).exists())
    return {str(path.relative_to(root)):sha256(path) for path in sorted(set(paths))}


def junit_summary(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag=="testsuite" else list(root.iter("testsuite"))
    result = {key:sum(int(s.get(key,"0")) for s in suites) for key in
              ("tests","failures","errors","skipped")}
    result["passed"] = result["tests"]-result["failures"]-result["errors"]-result["skipped"]
    result["seconds"] = sum(float(s.get("time","0")) for s in suites)
    return result


HARNESS = r'''
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
mode, testroot, expected, receipt, junit = sys.argv[1:]
testroot, expected = Path(testroot), Path(expected).resolve()
if mode == "source":
    sys.path.insert(0, str(testroot / "src"))
import bayes_stesn
origin = Path(bayes_stesn.__file__).resolve()
if not origin.is_relative_to(expected):
    raise RuntimeError(f"Wrong package origin before tests: {origin}")
import pytest
class Warnings:
    def __init__(self):
        self.records = []
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
          "warning_count": len(plugin.records), "package_version": metadata.version("bayes-stesn")}
Path(receipt).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
if wrong:
    raise RuntimeError(f"Package source contamination: {wrong}")
raise SystemExit(code)
'''


ENVIRONMENT_PROBE = r'''
import importlib.metadata as m
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
    args = parser.parse_args(argv)
    validation = ROOT/"results/validation"
    validation.mkdir(parents=True,exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="bstesn-release-work-"))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH",None)
    environment.pop("PYTHONHOME",None)
    environment.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTHONNOUSERSITE="1",
                       PIP_DISABLE_PIP_VERSION_CHECK="1")
    started = time.perf_counter()
    manifest = {"schema_version":1,"started_utc":datetime.now(timezone.utc).isoformat(),
                "passed":False,"repository_published":False,"public_repository_url":None,
                "github_ci_executed":False,"github_ci_configured":bool(list((ROOT/".github/workflows").glob("*.y*ml"))),
                "source_hashes":source_hashes(ROOT),"steps":[],"public_package_index":PYPI,
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
        if not args.skip_source_tests:
            run("source_pytest",[sys.executable,"-I",harness,"source",ROOT,ROOT/"src",
                validation/"source_pytest_receipt.json",validation/"source_pytest.xml"])
            manifest["source_tests"] = {**junit_summary(validation/"source_pytest.xml"),
                **json.loads((validation/"source_pytest_receipt.json").read_text())}
        else:
            manifest["source_tests"] = {"skipped_by_explicit_flag":True}
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
                    "tests/fixtures/v3_reference.npz"]
        missing = [name for name in required if not (extracted/name).is_file()]
        if missing:
            raise RuntimeError(f"Source distribution is missing required reproducibility files: {missing}")
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            if "bayes_stesn/gaussian.py" not in members or not any(name.endswith("/METADATA") for name in members):
                raise RuntimeError("Wheel is missing package source or distribution metadata")
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
        run("fresh_dependencies",[python,"-I","-m","pip","install","--index-url",PYPI,
            "--only-binary=:all:","--report",validation/"fresh_install_report.json",
            "-r",ROOT/"requirements-dev.lock","-r",ROOT/"requirements-report.lock"],cwd=work)
        run("wheel_install",[python,"-I","-m","pip","install","--no-deps","--force-reinstall",wheel],cwd=work)
        run("pip_check",[python,"-I","-m","pip","check"],cwd=work)
        probe = work/"environment_probe.py"
        probe.write_text(ENVIRONMENT_PROBE)
        run("fresh_environment",[python,"-I",probe,validation/"fresh_environment.json"],cwd=work)
        manifest["fresh_environment"] = json.loads((validation/"fresh_environment.json").read_text())
        manifest["dependency_install_receipts"] = [{"file":str(p.relative_to(ROOT)),"sha256":sha256(p)}
            for p in (validation/"fresh_install_report.json",validation/"public_registry_install_report.json") if p.is_file()]
        run("installed_pytest",[python,"-I",harness,"installed",extracted,path,
            validation/"installed_pytest_receipt.json",validation/"installed_pytest.xml"],cwd=work)
        manifest["installed_tests"] = {**junit_summary(validation/"installed_pytest.xml"),
            **json.loads((validation/"installed_pytest_receipt.json").read_text())}
        smoke = validation/"installed_smoke"
        run("installed_cli_smoke",[executable,"demo","--smoke","--output",smoke],cwd=work)
        summary = json.loads((smoke/"summary.json").read_text())
        manifest["cli_smoke"] = {"passed":True,"smoke_only":summary["smoke_only"],
            "diagnostic_gate_passed":summary["diagnostics"]["passed"],
            "scope":"Execution test; short-chain diagnostic failure is expected and is not convergence."}
        roundtrip = work/"prediction_roundtrip.py"
        roundtrip.write_text(ROUNDTRIP)
        run("installed_roundtrip",[python,"-I",roundtrip,smoke,validation/"installed_roundtrip.json"],cwd=work)
        manifest["safe_roundtrip"] = json.loads((validation/"installed_roundtrip.json").read_text())
        final_hashes = source_hashes(ROOT)
        changed = sorted(key for key in set(final_hashes)|set(manifest["source_hashes"])
                         if final_hashes.get(key)!=manifest["source_hashes"].get(key))
        manifest["working_tree_changed_during_check"] = changed
        built_scientific = {str(p.relative_to(extracted)):sha256(p) for p in (extracted/"src/bayes_stesn").glob("*.py")}
        drift = [name for name,digest in built_scientific.items() if not (ROOT/name).is_file() or sha256(ROOT/name)!=digest]
        manifest["built_scientific_source_hashes"] = built_scientific
        manifest["scientific_source_drift_after_build"] = drift
        manifest["github_ci_configured"] = bool(list((ROOT/".github/workflows").glob("*.y*ml")))
        manifest["passed"] = not drift and all(s["exit_code"]==0 for s in manifest["steps"])
        if drift:
            raise RuntimeError("Scientific source changed after build; rebuild and rerun the release gate")
    except Exception as error:
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
