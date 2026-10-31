"""Argument isolation, legacy routing and public CLI behavior."""
import contextlib
import importlib
import inspect
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from market_research.cli import main as cli
from market_research.cli.registry import COMMAND_SPECS, COMMANDS, LEGACY_COMMANDS


class CliTests(unittest.TestCase):
    def test_command_and_legacy_names_are_unique(self):
        self.assertEqual(len(COMMAND_SPECS),len(COMMANDS))
        self.assertEqual(len(COMMAND_SPECS),len(LEGACY_COMMANDS))

    def test_explicit_arguments_leave_process_arguments_untouched(self):
        original=sys.argv
        given=['experiments','110022','--window-years','2']
        captured=[]
        def command(argv,*,prog):
            self.assertIs(sys.argv,original)
            captured.append((list(argv),prog))
            argv.append('child mutation')
            return 7
        with patch.object(cli,'import_module',return_value=SimpleNamespace(main=command)):
            self.assertEqual(cli.main(given,prog='run.py'),7)
        self.assertEqual(captured,[(['110022','--window-years','2'],'run.py experiments')])
        self.assertEqual(given,['experiments','110022','--window-years','2'])
        self.assertIs(sys.argv,original)

    def test_root_script_preserves_implicit_analysis_and_accepts_commands(self):
        with patch.object(cli,'execute',return_value=0) as run:
            cli.root_main(['110022','--refresh'])
            self.assertEqual(run.call_args.args,('analyze',['110022','--refresh']))
            cli.root_main(['holdings','110022'])
            self.assertEqual(run.call_args.args,('holdings',['110022']))
            cli.root_main([])
            self.assertEqual(run.call_args.args,('analyze',[]))

    def test_unknown_command_and_help_do_not_load_research(self):
        with patch.object(cli,'import_module') as load,contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as invalid:
                cli.root_main(['not-a-command'])
            self.assertEqual(invalid.exception.code,2)
            with self.assertRaises(SystemExit) as help_result:
                cli.root_main(['--help'])
            self.assertEqual(help_result.exception.code,0)
            load.assert_not_called()

    def test_legacy_adapter_uses_registry_and_preserves_exit_code(self):
        with patch.object(cli,'execute',return_value=1) as run:
            self.assertEqual(cli.legacy_main('compat/entrypoints/run_selection.py',[]),1)
            self.assertEqual(run.call_args.args,('selection',[]))
            self.assertEqual(run.call_args.kwargs['prog'],'run_selection.py')
        with self.assertRaises(ValueError):
            cli.legacy_main('unknown.py',[])

    def test_every_command_accepts_explicit_arguments(self):
        for module in COMMANDS.values():
            command=importlib.import_module('market_research.'+module).main
            signature=inspect.signature(command)
            self.assertIn('argv',signature.parameters,module)
            self.assertIn('prog',signature.parameters,module)


if __name__=='__main__':unittest.main()
