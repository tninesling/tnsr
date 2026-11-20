"""Fail-closed tests for the explicit resident-desktop benchmark policy."""
import unittest
from unittest.mock import patch
import desktop_contexts as d


class DesktopIsolation(unittest.TestCase):
    def setUp(self):
        self.item = d.admit_descriptor(20, '/usr/libexec/xdg-desktop-portal-gnome',
                                      'C+G', 6, 123)
        self.allowed = {20: self.item}

    def test_compute_application_never_becomes_a_desktop_exception(self):
        for name, kind, memory in [('/usr/bin/python3', 'C', 6),
                                   ('xdg-desktop-portal-gnome', 'C', 6),
                                   ('snapd-desktop-integration', 'C+G', 128)]:
            with self.assertRaises(ValueError):
                d.admit_descriptor(20, name, kind, memory, 123)
        rows = d.classify_csv('20, desktop, 6\n30, python, 240\n', self.allowed,
                              identity=lambda pid: 123)
        self.assertEqual([x['pid'] for x in rows], [30])

    def test_strict_default_and_process_reuse_and_allocation_growth(self):
        self.assertEqual(len(d.classify_csv('20, desktop, 6', {})), 1)
        self.assertEqual(d.classify_csv('20, desktop, 6', self.allowed,
                                       identity=lambda pid: 123), [])
        self.assertEqual(len(d.classify_csv('20, desktop, 6', self.allowed,
                                            identity=lambda pid: 124)), 1)
        self.assertEqual(len(d.classify_csv('20, desktop, 7', self.allowed,
                                            identity=lambda pid: 123)), 1)
        self.assertEqual(d.classify_csv('30,worker,999', self.allowed, own_pid=30), [])

    def test_engine_activity_type_drift_and_growth_invalidate_idle_context(self):
        idle = '0 20 C+G - - - - - - 6 0 desktop'
        self.assertFalse(d.pmon_records(idle, self.allowed)[0]['active'])
        for column in range(3, 9):
            fields = idle.split(); fields[column] = '1'
            self.assertTrue(d.pmon_records(' '.join(fields), self.allowed)[0]['active'])
        for changed in [idle.replace('C+G', 'C'), idle.replace('6 0', '7 0')]:
            self.assertTrue(d.pmon_records(changed, self.allowed)[0]['active'])
        self.assertEqual(d.pmon_records('0 - - - - - - - - - - -', {}), [])
        with self.assertRaises(ValueError):
            d.pmon_records('unexpected format', self.allowed)

    def test_proc_stat_handles_spaces_in_comm(self):
        # Linux stat field22 follows comm+state, with 19 preceding tokens here.
        contents = '20 (desktop worker (helper)) ' + ' '.join(['S'] + ['0']*18 + ['123'])
        with patch.object(d.Path, 'read_text', return_value=contents):
            self.assertEqual(d.start_ticks(20), 123)


class CampaignAndCoverage(unittest.TestCase):
    def setUp(self):
        self.item = d.admit_descriptor(20, '/usr/libexec/xdg-desktop-portal-gnome',
                                      'C+G', 6, 123)
        self.allowed = {20: self.item}

    def test_campaign_set_cannot_be_rebased_between_workers(self):
        d.validate_current(self.allowed, self.allowed)
        for current in [{}, {30: dict(self.item, pid=30)},
                        {20: dict(self.item, process_start_ticks=124)},
                        {20: dict(self.item, baseline_mib=7)},
                        {20: dict(self.item, name='/other/xdg-desktop-portal-gnome')}]:
            with self.assertRaises(ValueError):
                d.validate_current(current, self.allowed)
        d.validate_current({20: dict(self.item, baseline_mib=5)}, self.allowed)
        # Lower allocation does not lower the original ceiling on the next worker.
        self.assertEqual(self.allowed[20]['baseline_mib'], 6)

    def test_unique_service_names_and_max_two(self):
        with self.assertRaises(ValueError):
            d.validate_allowed({20: self.item, 21: dict(self.item, pid=21)})
        other = d.admit_descriptor(21, 'snapd-desktop-integration', 'C+G', 6, 125)
        d.validate_allowed({20: self.item, 21: other})

    def test_no_samples_and_interrupted_coverage_reject(self):
        self.assertTrue(d.coverage_errors([], self.allowed, 1))
        sample = lambda t: dict(pid=20, elapsed_seconds=t)
        self.assertEqual(d.coverage_errors([sample(.2), sample(1), sample(2)], self.allowed, 2.2), [])
        for samples, duration in [([sample(.2)], 4),
                                   ([sample(.2), sample(4)], 4.2),
                                   ([sample(4)], 4.2)]:
            self.assertTrue(d.coverage_errors(samples, self.allowed, duration))

    def test_disappearance_rejected_even_without_runtime_samples(self):
        import tempfile
        import threading
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            m = object.__new__(d.Monitor)
            m.directory = d.Path(tmp)
            m.allowed, m.samples, m.errors, m.violations = self.allowed, [], [], []
            m.started, m.process, m.reader = None, None, None
            m.raw_stream = m.stderr_stream = None
            m.stopping = threading.Event()
            with patch.object(d, 'descriptors', return_value=({}, '<empty/>')):
                report = m.stop()
            self.assertTrue(any('set changed' in e for e in report['monitor_errors']))
            self.assertTrue(any('no utilization samples' in e for e in report['monitor_errors']))

    def test_monitor_exit_and_parse_errors_are_not_idle(self):
        import tempfile
        import threading
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            m = object.__new__(d.Monitor)
            m.directory = d.Path(tmp)
            m.allowed, m.samples, m.errors, m.violations = self.allowed, [], [], []
            m.started, m.reader = None, None
            m.process = Mock()
            m.process.poll.return_value = 2
            m.raw_stream = m.stderr_stream = None
            m.stopping = threading.Event()
            with patch.object(d, 'descriptors', return_value=(self.allowed, '<ok/>')):
                report = m.stop()
            self.assertTrue(any('exited early: 2' in e for e in report['monitor_errors']))
        with self.assertRaises(ValueError):
            d.classify_csv('20, N/A', self.allowed)
        with self.assertRaises(ValueError):
            d.classify_csv('20, desktop, -1', self.allowed)


class WorkerLifecycle(unittest.TestCase):
    """No subprocesses/GPU work: deterministic startup and reader failure injection."""
    class Process:
        def __init__(self, rows='', pid=777, worker=False):
            import io
            self.stdout = io.StringIO(rows)
            self.pid, self.worker = pid, worker
            self.returncode = 0 if worker else None
            self.terminated = self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    class DeferredReader:
        def __init__(self, target, **unused):
            self.target, self.ident = target, None

        def start(self):
            self.ident = 1

        def join(self, timeout=None):
            self.target()

        def is_alive(self):
            return False

    def execute_mock(self, popens, *, desktop=False, monitor=None):
        import json
        import tempfile
        import run
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            directory = d.Path(tmp) / 'worker'
            # A real results record permits proving successful lifecycle behavior.
            original_mkdir = d.Path.mkdir
            def mkdir(path, *args, **kwargs):
                original_mkdir(path, *args, **kwargs)
                if path == directory:
                    original_mkdir(path / 'artifacts')
                    (path / 'artifacts/results.json').write_text('{}')
            monitor = monitor or Mock()
            monitor.stop.return_value = dict(activity_violations=[], monitor_errors=[])
            with patch.object(run, 'gpu_snapshot', return_value={'compute_processes': ''}), \
                    patch.object(run.subprocess, 'Popen', side_effect=popens) as launch, \
                    patch.object(run.threading, 'Thread', self.DeferredReader), \
                    patch.object(d, 'Monitor', return_value=monitor), \
                    patch.object(d.Path, 'mkdir', mkdir):
                result = run.execute(['mock-worker'], directory, 1, desktop, {} if desktop else None)
                self.assertEqual(json.loads((directory / 'process.json').read_text())['status'], result['status'])
                self.assertTrue((directory / 'nvml-raw.csv').exists())
            return result, launch, monitor

    def test_sampler_launch_failure_records_contended_and_never_launches_worker(self):
        result, launch, monitor = self.execute_mock([OSError('sampler failed')], desktop=True)
        self.assertEqual(result['status'], 'contended')
        self.assertEqual(launch.call_count, 1)
        self.assertFalse(monitor.start.called)

    def test_desktop_start_failure_cleans_sampler_before_worker(self):
        from unittest.mock import Mock
        sampler, monitor = self.Process(), Mock()
        monitor.start.side_effect = OSError('pmon failed')
        result, launch, monitor = self.execute_mock([sampler], desktop=True, monitor=monitor)
        self.assertEqual(result['status'], 'contended')
        self.assertTrue(sampler.terminated)
        self.assertTrue(monitor.stop.called)
        self.assertEqual(launch.call_count, 1)

    def test_worker_launch_failure_closes_both_monitors(self):
        sampler = self.Process()
        result, _, monitor = self.execute_mock([sampler, OSError('worker failed')], desktop=True)
        self.assertEqual(result['status'], 'contended')
        self.assertTrue(sampler.terminated)
        self.assertTrue(monitor.stop.called)

    def test_sampler_parse_thread_failure_invalidates_successful_worker(self):
        sampler = self.Process('malformed\n')
        result, _, _ = self.execute_mock([sampler, self.Process(worker=True)])
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['status'], 'contended')
        self.assertTrue(any('NVML reader' in e for e in result['monitor_errors']))

    def test_sampler_early_exit_invalidates_successful_worker(self):
        sampler = self.Process('777, 10\n')
        # Exit is first observed during cleanup, not the pre-launch health check.
        calls = iter([None, 7])
        sampler.poll = lambda: next(calls)
        sampler.returncode = 7
        result, _, _ = self.execute_mock([sampler, self.Process(worker=True)])
        self.assertEqual(result['status'], 'contended')
        self.assertTrue(any('exited early: 7' in e for e in result['monitor_errors']))

    def test_healthy_sampler_passes_and_default_remains_strict(self):
        sampler = self.Process('777, 10\n')
        result, launch, _ = self.execute_mock([sampler, self.Process(worker=True)])
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(launch.call_args_list[0].args[0][0], 'nvidia-smi')
        self.assertEqual(result['nvml_samples'][0]['whole_process_memory_mib'], 10)
        import run
        with self.assertRaises(ValueError):
            run.execute([], d.Path('/unused'), 1, True)


class PmonReaderFailures(unittest.TestCase):
    def test_parse_failure_and_early_eof_keep_raw_stream_and_invalidate(self):
        import tempfile
        from unittest.mock import Mock
        allowed = {20: d.admit_descriptor(20, 'xdg-desktop-portal-gnome', 'C+G', 6, 123)}
        idle = '0 20 C+G - - - - - - 6 0 desktop\n'
        for raw in ['unexpected format\n', idle]:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmp:
                with patch.object(d, 'descriptors', return_value=(allowed, '<ok/>')), \
                        patch.object(d.subprocess, 'check_output', return_value=idle):
                    monitor = d.Monitor(d.Path(tmp), allowed)
                process = WorkerLifecycle.Process(raw)
                with patch.object(d.subprocess, 'Popen', return_value=process):
                    with self.assertRaises(ValueError):
                        monitor.start()
                with patch.object(d, 'descriptors', return_value=(allowed, '<ok/>')):
                    report = monitor.stop()
                self.assertTrue(report['monitor_errors'])
                self.assertEqual((d.Path(tmp) / 'desktop-context-pmon.txt').read_text(), raw)
                self.assertTrue((d.Path(tmp) / 'desktop-context-pmon-stderr.txt').exists())


class StopProcessErrors(unittest.TestCase):
    def test_failed_terminate_still_attempts_wait_and_kill(self):
        import subprocess
        from unittest.mock import Mock
        process = Mock()
        process.poll.return_value = None
        process.terminate.side_effect = OSError('injected terminate failure')
        process.wait.side_effect = [subprocess.TimeoutExpired('mock', 5), 0]
        errors = []
        d.stop_process(process, errors, 'mock-monitor', True)
        self.assertTrue(process.kill.called)
        self.assertEqual(process.wait.call_count, 2)
        self.assertTrue(any('terminate' in e for e in errors))

    def test_failed_cleanup_is_recorded_without_escaping(self):
        from unittest.mock import Mock
        process = Mock()
        process.poll.side_effect = OSError('injected poll failure')
        errors = []
        d.stop_process(process, errors, 'mock-monitor', True)
        self.assertTrue(any('cleanup' in e for e in errors))


if __name__ == '__main__':
    unittest.main()
