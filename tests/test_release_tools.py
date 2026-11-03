import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

REPO=Path(__file__).resolve().parents[1]


def _load_tool(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'tools'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release=_load_tool('release_check')
_previous_release=sys.modules.get('release_check')
sys.modules['release_check']=release
try:
    package=_load_tool('package_release')
finally:
    if _previous_release is None:
        del sys.modules['release_check']
    else:
        sys.modules['release_check']=_previous_release


def content_hash(value):
    """Independent canonical-JSON reference for these JSON-only manifests."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
                                    separators=(',',':')).encode()).hexdigest()


def fixture_package(root):
    """Keep subprocess origins and metadata independent of the real install."""
    (root/'src/bayes_stesn').mkdir(parents=True)
    (root/'src/bayes_stesn/__init__.py').write_text('')
    metadata=root/'src/bayes_stesn-0.0.0.dist-info'
    metadata.mkdir()
    (metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: bayes-stesn\nVersion: 0.0.0\n')
    (root/'tests').mkdir()


class ToolChecks(unittest.TestCase):
    def test_neural_install_plan_separates_support_and_cpu_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            pins={'torch':'2.15.0+cpu','setuptools':'84.1.0','filelock':'4.2.4'}
            commands=release.neural_install_commands(Path(sys.executable),work,
                work/'constraints.txt',work,pins)
            support_name,support_command=commands[0]
            torch_name,torch_command=commands[1]
            self.assertEqual(support_name,'neural_support_dependencies')
            self.assertEqual(support_command[support_command.index('--index-url')+1],release.PYPI)
            support=Path(support_command[support_command.index('-r')+1])
            self.assertEqual(support.read_text().splitlines(),['filelock==4.2.4','setuptools==84.1.0'])
            self.assertNotIn('--force-reinstall',support_command)
            self.assertEqual(torch_name,'neural_dependencies')
            self.assertEqual(torch_command[torch_command.index('--index-url')+1],release.PYTORCH_CPU)
            self.assertIn('--no-deps',torch_command)
            self.assertIn('--force-reinstall',torch_command)
            self.assertEqual(torch_command[-1],'torch==2.15.0+cpu')
            self.assertNotIn('-r',torch_command)
            self.assertEqual(Path(support_command[support_command.index('--report')+1]).name,
                             'neural_support_install_report.json')
            self.assertEqual(Path(torch_command[torch_command.index('--report')+1]).name,
                             'neural_install_report.json')
            self.assertEqual(pins['torch'],'2.15.0+cpu')

    def test_source_hashes_cover_shipped_scripts_without_generated_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            wanted=('CONTRIBUTING.md','MANIFEST.in','.gitignore','examples/quickstart.py',
                    'report/render_report.py','docs/evidence/receipt.json')
            for name in (*wanted,'report/benchmark_pilot/summary.json','docs/assets/example.png'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
            hashes=release.source_hashes(root)
            self.assertEqual(set(hashes),set(wanted))
            (root/'examples/quickstart.py').write_text('changed')
            self.assertNotEqual(hashes['examples/quickstart.py'],release.source_hashes(root)['examples/quickstart.py'])

    def test_benchmark_archive_and_tampering_gates(self):
        previous=package.ROOT
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'project';root.mkdir()
            package.ROOT=root
            try:
                source=root/'src/bayes_stesn/__init__.py';source.parent.mkdir(parents=True)
                source.write_text('VERSION = "test"\n')
                (root/'README.md').write_text('Temporary release fixture\n')
                validation=root/'results/validation';validation.mkdir(parents=True)
                (validation/'old_receipt.json').write_text('{"old":true}')
                for args in (['init','-q'],['config','user.name','Release test'],
                             ['config','user.email','release-test@example.invalid'],['add','.'],
                             ['commit','-qm','Temporary fixture source']):
                    subprocess.run(['git',*args],cwd=root,check=True)
                self.assertIn('results/validation/old_receipt.json',
                    subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines())
                run=root/'results/final_run';report=root/'report/benchmark_final'
                run.mkdir(parents=True);report.mkdir(parents=True)
                manifest={'schema':'benchmark-run-3.0','protocol':{'name':'smoke'},
                          'protocol_identity':'fixture-protocol','source_fingerprint':package.source_fingerprint(),
                          'stage':'execution_test'}
                manifest['identity']=content_hash(manifest)
                summary={'manifest_identity':manifest['identity'],'source_stable':True,
                         'final_source_fingerprint':manifest['source_fingerprint'],'jobs':1,
                         'comparison_status':'execution only'}
                (run/'manifest.json').write_text(json.dumps(manifest))
                (run/'summary.json').write_text(json.dumps(summary))
                (run/'resolved_protocol.yaml').write_text('name: smoke\n')
                job=run/'g1/rep_000/A/persistence';job.mkdir(parents=True)
                (job/'receipt.json').write_text(json.dumps({'manifest_identity':manifest['identity']}))
                (report/'run_manifest.json').write_text(json.dumps(manifest))
                (report/'summary.json').write_text(json.dumps(summary))
                (report/'report.html').write_text('<html>fixture</html>')
                (report/'benchmark_report.pdf').write_bytes(b'%PDF fixture for checksum tests only')
                (report/'figure.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
                inventory=[{'path':p.name,'bytes':p.stat().st_size,'sha256':package.digest(p)} for p in sorted(report.iterdir())]
                report_manifest={'run_identity':manifest['identity'],'render_only':True,'files':inventory}
                (report/'report_manifest.json').write_text(json.dumps(report_manifest))
                info,files=package.verify_benchmark(run,report)
                self.assertEqual(info['jobs'],1);self.assertIn(report/'figure.svg',files)
                svg=(report/'figure.svg').read_text();(report/'figure.svg').write_text(svg+'corrupt')
                with self.assertRaisesRegex(ValueError,'payload'):
                    package.verify_benchmark(run,report)
                (report/'figure.svg').write_text(svg)
                report_manifest['files'][0]['path']='../private'
                (report/'report_manifest.json').write_text(json.dumps(report_manifest))
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    package.verify_benchmark(run,report)
                report_manifest['files']=inventory
                # inventory was aliased by the deliberate mutation above.
                inventory[0]['path']='benchmark_report.pdf'
                (report/'report_manifest.json').write_text(json.dumps(report_manifest))
                (run/'resolved_protocol.yaml').write_text('name: altered\n')
                with self.assertRaisesRegex(ValueError,'protocol'):
                    package.verify_benchmark(run,report)
                (run/'resolved_protocol.yaml').write_text('name: smoke\n')
                stale=root/'results/integration_old';stale.mkdir();(stale/'junk.json').write_text('{}')
                renders=report/'render_interim';renders.mkdir();(renders/'page.png').write_bytes(b'qa')
                self.assertFalse(package.include(renders/'page.png'))
                self.assertFalse(package.include(stale/'junk.json'))
                current=validation/'current_receipt.json';current.write_text('{"current":true}')
                validation_entries=[{'file':str(current.relative_to(root)),'bytes':current.stat().st_size,'sha256':package.digest(current)}]
                evidence={'passed':True,'source_hashes':release.source_hashes(root),'artifacts':[],
                          'validation_artifacts':validation_entries}
                extra_run=root/'results/detail_run';extra_report=root/'report/benchmark_detail'
                shutil.copytree(run,extra_run);shutil.copytree(report,extra_report)
                shutil.rmtree(extra_report/'render_interim')
                extra_manifest=dict(manifest,protocol={'name':'detail'},protocol_identity='fixture-detail')
                extra_manifest.pop('identity');extra_manifest['identity']=content_hash(extra_manifest)
                extra_summary=dict(summary,manifest_identity=extra_manifest['identity'])
                (extra_run/'manifest.json').write_text(json.dumps(extra_manifest))
                (extra_run/'summary.json').write_text(json.dumps(extra_summary))
                (extra_run/'resolved_protocol.yaml').write_text('name: detail\n')
                (extra_run/'g1/rep_000/A/persistence/receipt.json').write_text(json.dumps({'manifest_identity':extra_manifest['identity']}))
                (extra_report/'run_manifest.json').write_text(json.dumps(extra_manifest))
                (extra_report/'summary.json').write_text(json.dumps(extra_summary))
                extra_inventory=[{'path':p.name,'bytes':p.stat().st_size,'sha256':package.digest(p)}
                                 for p in sorted(extra_report.iterdir()) if p.name!='report_manifest.json']
                (extra_report/'report_manifest.json').write_text(json.dumps({'run_identity':extra_manifest['identity'],
                    'render_only':True,'files':extra_inventory}))
                supplemental=root/'results/calibration_A';supplemental.mkdir()
                (supplemental/'calibration.json').write_text('{"blocks":25,"independent_replications":false}')
                with self.assertRaisesRegex(ValueError,'budget'):
                    package.supplemental_inventory(supplemental,1)
                with self.assertRaisesRegex(ValueError,'cap'):
                    package.supplemental_inventory(supplemental,10000,remaining_files=0)
                staging=supplemental/'.model.npz.bstesn-host-123-random.npz'
                staging.write_bytes(b'partial')
                self.assertFalse(package.include(staging))
                with self.assertRaisesRegex(ValueError,'excluded'):
                    package.supplemental_inventory(supplemental,10000)
                staging.unlink()
                link=supplemental/'untrusted_link';link.symlink_to(current)
                with self.assertRaisesRegex(ValueError,'symlinks'):
                    package.supplemental_inventory(supplemental,10000)
                link.unlink()
                alias=root/'results/alias';alias.symlink_to(supplemental,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'symlink'):
                    package.project_directory(alias)
                alias.unlink()
                old=current.read_text();current.write_text('tampered')
                with self.assertRaisesRegex(ValueError,'changed'):
                    package.verified_inventory(validation_entries)
                current.write_text(old)
                (root/'results/release_manifest.json').write_text(json.dumps(evidence))
                output=Path(temp)/'downloads'
                package.main(['--benchmark-run',str(run),'--benchmark-report',str(report),
                              '--additional-benchmark',str(extra_run),str(extra_report),
                              '--supplemental-evidence',str(supplemental),'--output',str(output)])
                with zipfile.ZipFile(output/'bayes_stesn_implementation.zip') as archive:
                    names=archive.namelist()
                    self.assertIn('bayes-stesn/report/benchmark_final/figure.svg',names)
                    self.assertIn('bayes-stesn/report/benchmark_detail/figure.svg',names)
                    self.assertIn('bayes-stesn/results/calibration_A/calibration.json',names)
                    self.assertIn('bayes-stesn/results/validation/current_receipt.json',names)
                    self.assertNotIn('bayes-stesn/results/validation/old_receipt.json',names)
                    sealed=json.loads(archive.read('bayes-stesn/release_artifacts.json'))
                    self.assertEqual(len(sealed['additional_benchmarks']),1)
                    self.assertNotEqual(sealed['benchmark']['identity'],sealed['additional_benchmarks'][0]['identity'])
                    self.assertEqual(len(sealed['supplemental_evidence']),1)
                    self.assertFalse(any('integration_old' in name or 'render_interim' in name for name in names))
                with self.assertRaisesRegex(FileNotFoundError,'controlled_study'):
                    package.main(['--output',str(output)])
                legacy=root/'report/controlled_study';legacy.mkdir()
                protocol={'protocol_sha256':'fixture-legacy'}
                for name in ('summary.json','protocol.json'):
                    (legacy/name).write_text(json.dumps(protocol))
                (root/'report/report_manifest.json').write_text('{}')
                (root/'report/bayes_stesn_implementation_report.pdf').write_bytes(b'%PDF fixture')
                (root/'report/bayes_stesn_results.html').write_text('<html>legacy fixture</html>')
                package.main(['--output',str(output/'legacy')])
                with zipfile.ZipFile(output/'legacy/bayes_stesn_implementation.zip') as archive:
                    self.assertIn('bayes-stesn/results/validation/old_receipt.json',archive.namelist())
                extra=root/'src/bayes_stesn/untracked.py';extra.write_text('x=1\n')
                with self.assertRaisesRegex(ValueError,'Source changed'):
                    package.main(['--benchmark-run',str(run),'--benchmark-report',str(report),'--output',str(output)])
                evidence['source_hashes']=release.source_hashes(root)
                (root/'results/release_manifest.json').write_text(json.dumps(evidence))
                with self.assertRaisesRegex(ValueError,'not committed'):
                    package.main(['--benchmark-run',str(run),'--benchmark-report',str(report),'--output',str(output)])
            finally:
                package.ROOT=previous

    def test_current_validation_inventory_excludes_previous_gate_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'results/validation';directory.mkdir(parents=True)
            fixed=('fresh_environment.json','installed_roundtrip.json','installed_pytest_receipt.json',
                   'installed_pytest.xml','current.log','fresh_install_report.json',
                   'neural_support_install_report.json','neural_install_report.json')
            for name in fixed:
                (directory/name).write_text('{}')
            for name in ('source_pytest.xml','source_pytest_receipt.json','public_registry_install_report.json','venv_creation.json'):
                (directory/name).write_text('stale')
            old=directory/'installed_benchmark_smoke/previous';old.mkdir(parents=True)
            (old/'manifest.json').write_text('old run')
            current=directory/'installed_benchmark_smoke/current';current.mkdir(parents=True)
            (current/'manifest.json').write_text('new run')
            cli=directory/'installed_smoke';cli.mkdir();(cli/'model.npz').write_bytes(b'current')
            manifest={'steps':[{'log':'results/validation/current.log'}],
                      'dependency_install_receipts':[{'file':'results/validation/'+name} for name in
                          ('fresh_install_report.json','neural_support_install_report.json','neural_install_report.json')],
                      'source_tests':{'skipped_by_explicit_flag':True},'installed_tests':{},
                      'venv_created_this_run':False,
                      'cli_smoke':{'artifact_directory':str(cli.relative_to(root))},
                      'benchmark_smoke':{'artifact_directory':str(current.relative_to(root))}}
            inventory=release.validation_inventory(root,manifest)
            names={entry['file'] for entry in inventory}
            self.assertEqual(len(names),len(fixed)+2)
            self.assertIn('results/validation/installed_benchmark_smoke/current/manifest.json',names)
            self.assertIn('results/validation/neural_support_install_report.json',names)
            self.assertIn('results/validation/neural_install_report.json',names)
            self.assertFalse(any('previous' in name or 'source_pytest' in name or 'venv_creation' in name for name in names))

    def test_harness_explicit_optional_deselection_and_zero_skip_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture_package(root)
            (root/'tests/test_sample.py').write_text('''import unittest
def test_present():
    assert True
@unittest.skip("Optional PyTorch not installed")
class TestOptional(unittest.TestCase):
    def test_optional(self):
        pass
''')
            harness=root/'harness.py';harness.write_text(release.HARNESS)
            env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONNOUSERSITE':'1'}
            def execute(scope):
                return subprocess.run([sys.executable,'-I',str(harness),'source',str(root),str(root/'src'),
                    str(root/'receipt.json'),str(root/'tests.xml'),scope],cwd=root,env=env,capture_output=True,text=True)
            result=execute('core');self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            receipt=json.loads((root/'receipt.json').read_text())
            self.assertEqual(receipt['package_version'],'0.0.0')
            self.assertEqual(len(receipt['optional_deselected']),1)
            self.assertEqual(release.junit_summary(root/'tests.xml')['skipped'],0)
            result=execute('neural');self.assertNotEqual(result.returncode,0)
            self.assertEqual(len(json.loads((root/'receipt.json').read_text())['unexpected_skipped']),1)
            with (root/'tests/test_sample.py').open('a') as stream:
                stream.write('\n@unittest.skip("unexpected skip")\nclass TestUnexpected(unittest.TestCase):\n    def test_skipped(self):\n        pass\n')
            result=execute('core');self.assertNotEqual(result.returncode,0)

    def test_harness_deselects_neural_marker_only_in_core_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture_package(root)
            (root/'pytest.ini').write_text('[pytest]\nmarkers =\n    neural: optional CPU neural fixture\n')
            (root/'tests/test_marked.py').write_text('''from pathlib import Path
import pytest
def test_core():
    pass
@pytest.mark.neural
def test_optional():
    Path(__file__).with_name("neural_executed.txt").write_text("executed")
''')
            harness=root/'harness.py';harness.write_text(release.HARNESS)
            env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONNOUSERSITE':'1'}
            for scope,selected,deselected in (('core',1,1),('neural',2,0)):
                with self.subTest(scope=scope):
                    result=subprocess.run([sys.executable,'-I',str(harness),'source',str(root),str(root/'src'),
                        str(root/'receipt.json'),str(root/'tests.xml'),scope],cwd=root,env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    receipt=json.loads((root/'receipt.json').read_text())
                    self.assertEqual(len(receipt['optional_deselected']),deselected)
                    self.assertEqual(receipt['unexpected_skipped'],[])
                    self.assertEqual(receipt['warning_count'],0)
                    self.assertEqual(release.junit_summary(root/'tests.xml')['passed'],selected)
                    self.assertEqual((root/'tests/neural_executed.txt').exists(),scope=='neural')

    def test_harness_rejects_pytest_warnings_even_when_tests_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture_package(root)
            (root/'tests/test_warning.py').write_text('import warnings\ndef test_warning():\n    warnings.warn("release warning fixture", UserWarning)\n')
            harness=root/'harness.py';harness.write_text(release.HARNESS)
            env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONNOUSERSITE':'1'}
            for scope in ('core','neural'):
                with self.subTest(scope=scope):
                    result=subprocess.run([sys.executable,'-I',str(harness),'source',str(root),str(root/'src'),
                        str(root/'receipt.json'),str(root/'tests.xml'),scope],cwd=root,env=env,capture_output=True,text=True)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('unexpected pytest warning',result.stderr)
                    receipt=json.loads((root/'receipt.json').read_text())
                    self.assertEqual(receipt['package_version'],'0.0.0')
                    self.assertEqual(receipt['pytest_exit_code'],0)
                    self.assertEqual(receipt['warning_count'],1)
                    self.assertEqual(receipt['warnings'][0]['message'],'release warning fixture')
                    self.assertEqual(release.junit_summary(root/'tests.xml')['passed'],1)

    def test_harness_spawn_runs_worker_once_with_inherited_source_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture_package(root)
            (root/'src/bayes_stesn/worker.py').write_text('''import json
import os
from pathlib import Path
import bayes_stesn
def record_child(target):
    Path(target).write_text(json.dumps({"pid":os.getpid(),"origin":bayes_stesn.__file__}))
''')
            (root/'tests/test_spawn.py').write_text('''import json
import multiprocessing
import os
from pathlib import Path
from bayes_stesn.worker import record_child
def test_spawn():
    root=Path(__file__).resolve().parents[1]
    with (root/"suite_executions.txt").open("a") as stream:
        stream.write(multiprocessing.current_process().name+"\\n")
    assert multiprocessing.current_process().name=="MainProcess"
    child=multiprocessing.get_context("spawn").Process(target=record_child,args=(str(root/"child.json"),))
    child.start()
    try:
        child.join(10)
        assert child.exitcode==0
    finally:
        if child.is_alive():
            child.terminate();child.join()
        child.close()
    result=json.loads((root/"child.json").read_text())
    assert result["pid"]!=os.getpid()
    assert Path(result["origin"]).resolve()==root/"src/bayes_stesn/__init__.py"
''')
            harness=root/'harness.py';harness.write_text(release.HARNESS)
            env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONNOUSERSITE':'1'}
            result=subprocess.run([sys.executable,'-I',str(harness),'source',str(root),str(root/'src'),
                str(root/'receipt.json'),str(root/'tests.xml'),'core'],cwd=root,env=env,
                capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual((root/'suite_executions.txt').read_text().splitlines(),['MainProcess'])
            receipt=json.loads((root/'receipt.json').read_text())
            self.assertEqual(receipt['source_imports_detected'],{})
            self.assertEqual(receipt['warning_count'],0)
            self.assertEqual(release.junit_summary(root/'tests.xml')['passed'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
