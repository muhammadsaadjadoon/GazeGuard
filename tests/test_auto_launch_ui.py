"""Default launch needs no camera button or mandatory manual setup."""
from pathlib import Path
from gazeguard.webapp import Engine, ASSETS

def test_browser_autostart_copy_and_recovery():
    html=(ASSETS/'index.html').read_text(encoding='utf-8')
    js=(ASSETS/'app.js').read_text(encoding='utf-8')
    assert 'Initializing live camera' in html
    assert 'id="start-center"' not in html
    assert 'id="start"' not in html
    assert 'id="retry-camera"' in html
    assert "command('start')" in js
    assert "state.running=!!d.camera||d.status==='LOADING'||d.status==='CONNECTING'" in js

def test_engine_does_not_require_setup(tmp_path):
    e=Engine(tmp_path)
    assert e.snapshot()['mode']=='AUTOMATIC'
    assert e.wizard is None
    assert e.snapshot()['status']=='CONNECTING'
