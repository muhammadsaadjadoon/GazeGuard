"""Train a small binary lens-gaze classifier on participant/session-separated data.

Input rows: participant_id,session_id,label and eight numeric landmark columns.
Lens label = 1; screen/other labels = 0. No camera images required.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from gazeguard.trained_gaze import FEATURE_NAMES, TrainedGazeModel, save_model, sigmoid

ALLOWED = {'lens', 'screen', 'left', 'right', 'up', 'down', 'away'}


def read_rows(path: Path):
    xs, ys, sessions, people = [], [], [], []
    with Path(path).open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        required = {'participant_id','session_id','label', *FEATURE_NAMES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('CSV must include: ' + ', '.join(sorted(required)))
        for row in reader:
            label = row['label'].strip().lower()
            if label not in ALLOWED:
                raise ValueError('Unknown class label: '+label)
            x = np.array([float(row[name]) for name in FEATURE_NAMES], dtype=float)
            if not np.all(np.isfinite(x)):
                continue
            person = row['participant_id'].strip()
            session = row['session_id'].strip()
            if not person or not session:
                raise ValueError('participant_id and session_id required')
            xs.append(x)
            ys.append(int(label=='lens'))
            sessions.append(person + '/' + session)
            people.append(person)
    if not xs:
        raise ValueError('Dataset is empty')
    return np.asarray(xs),np.asarray(ys),np.asarray(sessions),np.asarray(people)


def split_groups(groups, labels, seed=41):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('Collect at least two independent sessions (preferably participants) before training')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    for group in unique:
        test = groups == group
        if len(np.unique(labels[test])) == 2 and len(np.unique(labels[~test])) == 2:
            return ~test,test,str(group)
    raise ValueError('No group split contains both lens and non-lens samples in train and validation')


def train_model(x, y, groups, *, epochs=900, learning_rate=.07):
    train, valid, held_out = split_groups(groups,y)
    if min(np.bincount(y[train],minlength=2)) < 15 or min(np.bincount(y[valid],minlength=2)) < 10:
        raise ValueError('Need >=15 training and >=10 validation samples per class, from separate groups')
    mu=x[train].mean(axis=0)
    sd=np.maximum(x[train].std(axis=0),.01)
    centered = np.clip((x[train]-mu)/sd, -12, 12)
    a = np.concatenate([centered, centered**2], axis=1)
    w=np.zeros(a.shape[1]); bias=0.
    class_count=np.bincount(y[train],minlength=2)
    sample_weights=np.where(y[train]==1, .5/class_count[1], .5/class_count[0])*len(y[train])
    for _ in range(epochs):
        p=sigmoid(a@w+bias)
        residual=(p-y[train])*sample_weights
        w-=learning_rate*(a.T@residual/len(a)+.012*w)
        bias-=learning_rate*residual.mean()
    val_norm=np.clip((x[valid]-mu)/sd,-12,12)
    validation_prob=sigmoid(np.concatenate([val_norm,val_norm**2],axis=1)@w+bias)
    neg=validation_prob[y[valid]==0]
    pos=validation_prob[y[valid]==1]
    # Choose a boundary that does not falsely claim lens contact for observed
    # non-lens VALIDATION samples; this is NOT a population guarantee.
    threshold=max(.50, float(np.max(neg))+.005)
    if threshold >= 1:
        raise ValueError('Model is not separable on validation data; collect better examples')
    recall=float(np.mean(pos >= threshold))
    if recall < .35:
        raise ValueError(f'Model failed held-out lens recall ({recall:.1%}); collect better data before deployment')
    pred=validation_prob>=threshold
    tp=int(np.sum(pred & (y[valid]==1))); fp=int(np.sum(pred & (y[valid]==0)))
    tn=int(np.sum(~pred & (y[valid]==0))); fn=int(np.sum(~pred & (y[valid]==1)))
    report=dict(split='held_out_session_or_participant', held_out_group=held_out,
                train_rows=int(train.sum()),validation_rows=int(valid.sum()),
                tp=tp,fp=fp,tn=tn,fn=fn,
                recall=round(tp/max(tp+fn,1),4),
                false_active_rate=round(fp/max(fp+tn,1),4),
                caution='Held-out group metrics are not a real-world accuracy guarantee')
    return TrainedGazeModel(mu,sd,w,bias,threshold,report)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Train webcam landmark lens/non-lens classifier')
    parser.add_argument('--csv',type=Path,default=Path('datasets/private/landmarks.csv'))
    parser.add_argument('--output',type=Path,default=Path('models/trained_gaze.json'))
    args=parser.parse_args(argv)
    x,y,sessions,people=read_rows(args.csv)
    # Participant-independent validation whenever multiple distinct participants exist.
    groups=people if len(np.unique(people))>=2 else sessions
    model=train_model(x,y,groups)
    save_model(args.output,model)
    print(json.dumps(dict(model=str(args.output),threshold=model.threshold,**model.validation),indent=2))
    print('Model trained and saved. Evaluate on a NEW webcam/device before enabling live use.')


if __name__=='__main__':
    main()
