"""Training checks use synthetic numbers only; they do NOT measure webcam accuracy."""
from pathlib import Path
import csv
import numpy as np
import pytest
from gazeguard.trained_gaze import FEATURE_NAMES, load_model, save_model
from training.train_landmarks import read_rows, train_model


def fake_data(path):
    rng=np.random.default_rng(312)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['participant_id','session_id','label',*FEATURE_NAMES]);writer.writeheader()
        for person in ['p01','p02','p03']:
            for label, center in [('lens',-0.05),('screen',0.12),('left',-.20),('right',.24)]:
                for i in range(65):
                    x=np.array([center,0,0,0,center,0,center,0])+rng.normal(0,.007,8)
                    writer.writerow(dict(participant_id=person,session_id='s01',label=label,
                                         **dict(zip(FEATURE_NAMES,x))))


def test_end_to_end_synthetic_training(tmp_path):
    csvfile=tmp_path/'data.csv'
    fake_data(csvfile)
    X,y,sessions,people=read_rows(csvfile)
    model=train_model(X,y,people)
    assert model.validation['fp']==0
    assert model.validation['recall']>=.35
    output=tmp_path/'model.json'
    save_model(output,model)
    loaded=load_model(output)
    assert loaded.classify(np.array([-.05,0,0,0,-.05,0,-.05,0]))
    assert not loaded.classify(np.array([.12,0,0,0,.12,0,.12,0]))


def test_insufficient_groups_is_rejected(tmp_path):
    f=tmp_path/'data.csv'
    fake_data(f)
    x,y,sessions,people=read_rows(f)
    with pytest.raises(ValueError,match='two independent sessions'):
        train_model(x,y,np.array(['same']*len(y)))


def test_model_rejects_bad_format(tmp_path):
    p=tmp_path/'bad.json'
    p.write_text('{"format":"bad"}')
    with pytest.raises(ValueError): load_model(p)
