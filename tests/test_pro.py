"""Offline-only tests; webcam not needed."""
import json
import pytest
from gazeguard.alerts import AlertEngine
from gazeguard.analytics import aggregate, load_history, render_html, read_session, save_html
from gazeguard.cli import main, diagnostics
from gazeguard.config import Settings, atomic_json
from gazeguard.reporting import SessionRecorder


def test_alert_duration_and_cooldown():
    alerts = AlertEngine(after_seconds=2, cooldown_seconds=10)
    assert alerts.update('LOOKING_AWAY', 0) is None
    assert alerts.update('LOOKING_AWAY', 1.9) is None
    assert alerts.update('LOOKING_AWAY', 2).code == 'LOOKING_AWAY'
    assert alerts.update('LOOKING_AWAY', 3) is None
    assert alerts.update('ATTENTIVE', 4) is None
    assert alerts.update('LOOKING_AWAY', 5) is None
    assert alerts.update('LOOKING_AWAY', 7) is None
    assert alerts.update('LOOKING_AWAY', 12).code == 'LOOKING_AWAY'


def test_alert_absent_and_validation():
    alert = AlertEngine(1, 2)
    assert alert.update('ABSENT', 1) is None
    assert alert.update('ABSENT', 2).message == 'Return to the camera view'
    with pytest.raises(ValueError):
        AlertEngine(0, 1)


def test_weighted_history_and_html_escape(tmp_path):
    folder = tmp_path / 'sessions'
    folder.mkdir()
    a = SessionRecorder(0)
    a.record(0, 'ATTENTIVE')
    a.export(folder, 10)
    b = SessionRecorder(0)
    b.record(0, 'LOOKING_AWAY')
    b.export(folder, 30)
    sessions, issues = load_history(folder)
    assert len(sessions) == 2 and not issues
    stats = aggregate(sessions)
    assert stats['attention_percent'] == 25.0
    html = render_html(sessions, 80)
    assert 'Session attention trend' in html
    assert '<script' not in html
    out = save_html(tmp_path / 'dash.html', sessions)
    assert out.exists()
    with pytest.raises(ValueError):
        render_html(sessions, -1)


def test_reject_malformed_history(tmp_path):
    folder = tmp_path / 'sessions'
    folder.mkdir()
    (folder / 'session_bad.json').write_text('wrong')
    (folder / 'other.json').write_text('{}')
    sessions, warnings = load_history(folder)
    assert sessions == [] and len(warnings) == 1
    with pytest.raises(ValueError):
        load_history(folder, 0)


def test_cli_end_to_end(tmp_path, capsys):
    assert main(['init', '--data-dir', str(tmp_path)]) == 0
    assert (tmp_path/'settings.json').exists()
    assert main(['history', '--data-dir', str(tmp_path)]) == 0
    assert main(['report', '--data-dir', str(tmp_path)]) == 0
    assert (tmp_path/'analytics.html').exists()
    output = capsys.readouterr().out
    assert 'Report:' in output


def test_config_pro_validation():
    Settings(focus_target_percent=100).validate()
    with pytest.raises(ValueError):
        Settings(focus_target_percent=101).validate()
    with pytest.raises(ValueError):
        Settings(alert_after_seconds=-2).validate()


def test_doctor_without_camera(tmp_path):
    result = diagnostics(tmp_path)
    assert result['camera'] == 'not tested'
    assert 'python' in result
