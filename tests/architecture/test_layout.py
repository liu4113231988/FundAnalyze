"""Package boundaries and backward-compatible research entry points."""
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

from market_research import config as C
from market_research.cli.registry import COMMANDS, LEGACY_COMMANDS


def legacy_entries():
    return [C.ROOT/name if (C.ROOT/name).is_file() else C.ROOT/'compat/entrypoints'/name
            for name in LEGACY_COMMANDS]


class LayoutTests(unittest.TestCase):
    def test_test_directories_do_not_shadow_packaging_dependency(self):
        import packaging.version
        self.assertFalse(Path(packaging.__file__).resolve().is_relative_to(C.ROOT/'tests'))

    def test_compatibility_implementations_are_centralized(self):
        self.assertEqual({p.name for p in (C.ROOT/'src').glob('*.py')},{'__init__.py'})
        self.assertEqual({p.name for p in (C.ROOT/'webapp').glob('*.py')},{'__init__.py'})
        for path in ('compat/src/models.py','compat/webapp/app.py','compat/webapp/api/tasks.py'):
            self.assertTrue((C.ROOT/path).is_file(),path)
        from importlib.resources import files
        self.assertTrue((files('market_research.api')/'static/index.html').is_file())

    def test_old_import_and_pickle_class_paths_are_aliases(self):
        from market_research.modeling.catalog import ModelSpec
        from market_research.modeling.engine import ModelZoo
        self.assertIs(importlib.import_module('src.config'),C)
        self.assertIs(importlib.import_module('src.models').ModelZoo,ModelZoo)
        self.assertIs(pickle.loads(b'csrc.models\nModelSpec\n.'),ModelSpec)

    def test_root_scripts_only_forward_main(self):
        entries=legacy_entries()
        self.assertEqual([p.name for p in C.ROOT.glob('run*.py')],['run.py'])
        self.assertEqual(len(entries),14)
        for path in entries:
            tree=ast.parse(path.read_text(encoding='utf-8'))
            self.assertFalse(any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in tree.body),path.name)
            self.assertLessEqual(len(path.read_text(encoding='utf-8').splitlines()),8,path.name)

    def test_core_does_not_import_http_or_legacy_modules(self):
        root=C.ROOT/'src/market_research'
        for folder in ('services','market','funds','modeling','data','tasks'):
            for path in (root/folder).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    modules=[node.module or ''] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
                    self.assertFalse(any(m=='src' or m.startswith(('src.','webapp','market_research.api')) for m in modules),str(path))

    def test_research_commands_import_without_creating_results(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'output') as directory:
            env={**os.environ,'FIL_HOME':directory,'FIL_DATA_DIR':str(Path(directory)/'data')}
            code="import importlib; from market_research.cli.registry import COMMANDS, LEGACY_COMMANDS; [importlib.import_module('market_research.'+m) for m in COMMANDS.values()]; print('imported')"
            result=subprocess.run([sys.executable,'-c',code],cwd=directory,env=env,capture_output=True,timeout=40)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
            self.assertFalse(any(Path(directory).rglob('*.csv')))

    def test_all_legacy_entries_support_help_without_running_research(self):
        paths=legacy_entries()
        def check(path):
            result=subprocess.run([sys.executable,str(path),'--help'],cwd=C.ROOT,capture_output=True,timeout=50)
            return path.name,result
        with ThreadPoolExecutor(max_workers=4) as pool:
            for name,result in pool.map(check,paths):
                self.assertEqual(result.returncode,0,name+': '+result.stderr.decode(errors='replace'))
                self.assertIn(b'usage:',result.stdout.lower(),name)


if __name__=='__main__':unittest.main()
