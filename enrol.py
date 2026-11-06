"""Build CAM++ or TDNN templates using the query's multiwindow aggregation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

from identify import ROOT, SpeakerIdentifier, checksum, load_windows, normalize

MVP_MEMBERS = ['member01', 'member02', 'member03', 'member04', 'member05']


def read_enrollment(manifest, members, minimum):
    if minimum < 1 or not members or len(set(members)) != len(members) or 'Unknown' in members:
        raise ValueError('Require distinct member IDs and a positive recording minimum')
    with Path(manifest).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'path', 'recording_id', 'speaker_id', 'split'}
    if not rows or any(not required <= row.keys() for row in rows):
        raise ValueError('Enrollment CSV needs path,recording_id,speaker_id,split')
    ids, resolved, counts = set(), [], Counter()
    for row in rows:
        if row['split'] != 'enrollment' or row['speaker_id'] not in members:
            raise ValueError('Every enrollment row must belong to a declared member and enrollment split')
        if not row['recording_id'].strip() or row['recording_id'] in ids:
            raise ValueError('Enrollment recording IDs must be nonempty and unique')
        ids.add(row['recording_id'])
        path = Path(row['path'])
        if not path.is_absolute():
            path = Path(manifest).parent/path
        path = path.resolve()
        if {'session02','session03','heldout'} & {part.lower() for part in path.parts}:
            raise ValueError('Development/held-out recordings cannot be used for enrollment')
        resolved.append({**row, 'path': path})
        counts[row['speaker_id']] += 1
    missing = {member: minimum-counts[member] for member in members if counts[member] < minimum}
    if missing:
        raise ValueError(f'Missing enrollment recordings: {missing}')
    return resolved


def load_reference(model_type):
    if model_type == 'tdnn':
        from ml.tdnn.identify_frozen import FrozenTDNN

        release_path = ROOT/'ml/tdnn/frozen_libri251_fp32.json'
        release = json.loads(release_path.read_text(encoding='utf-8'))
        return FrozenTDNN(release['conditions_id']).identifier
    config_path = ROOT/'ml/campplus/frozen_threshold.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    database_path = config_path.parent/config['template_path']
    if checksum(database_path) != config['binding']['template_db_sha256']:
        raise ValueError('Reference CAM++ template database changed')
    identifier = SpeakerIdentifier(model_type, database_path)
    if identifier.model_hash != config['binding']['model_sha256']:
        raise ValueError('Reference CAM++ model differs from its configuration')
    return identifier


def build_templates(manifest, output, members, minimum, conditions_id, model_type='campplus'):
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists; use a new database filename')
    if output.suffix.lower() != '.json':
        raise ValueError('Template output must use the .json extension')
    if not conditions_id.strip():
        raise ValueError('A recording/preprocessing conditions ID is required')
    manifest_hash = checksum(Path(manifest))
    rows = read_enrollment(manifest, members, minimum)
    # Validate all input audio and PCM duplicates before loading the model.
    seen_pcm, provenance = set(), []
    for row in rows:
        load_windows(row['path'], 'multiwindow')
        with wave.open(str(row['path']), 'rb') as reader:
            pcm = reader.readframes(reader.getnframes())
        digest = hashlib.sha256(pcm).hexdigest()
        if digest in seen_pcm:
            raise ValueError('Duplicate PCM in enrollment; use independent recordings')
        seen_pcm.add(digest)
        provenance.append(dict(recording_id=row['recording_id'], speaker_id=row['speaker_id'],
            audio_sha256=checksum(row['path']), pcm_sha256=digest))
    identifier = load_reference(model_type)
    model_hash = identifier.model_hash
    per_member = {member: [] for member in members}
    for row, source in zip(rows, provenance):
        windows, _ = load_windows(row['path'], 'multiwindow')
        vectors = identifier.extract_windows(windows)
        dimension = 192 if model_type == 'campplus' else 64
        if vectors.shape != (len(windows), dimension):
            raise ValueError(f'Expected one {dimension}-D embedding per window')
        recording = normalize(normalize(vectors).mean(axis=0))
        per_member[row['speaker_id']].append(recording)
        if checksum(row['path']) != source['audio_sha256']:
            raise ValueError('Enrollment recording changed during extraction')
    if checksum(identifier.model_path) != model_hash:
        raise ValueError('Model changed during enrollment')
    if checksum(Path(manifest)) != manifest_hash:
        raise ValueError('Enrollment manifest changed during extraction')
    database = dict(model_type=model_type, model_sha256=model_hash, model_path=identifier.model_path.relative_to(ROOT).as_posix(),
        strategy='multiwindow', members={m:normalize(np.stack(v).mean(axis=0)).tolist() for m,v in per_member.items()},
        enrollment_conditions_id=conditions_id, enrollment_manifest_sha256=manifest_hash,
        enrollment_recordings=provenance,
        completeness='five_member_templates' if set(members) == set(MVP_MEMBERS) else 'partial_diagnostic_templates',
        threshold_requires_recalibration=True)
    if model_type == 'tdnn':
        database['inference_branch'] = 'fp32'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(database, indent=2, allow_nan=False), encoding='utf-8')
    return dict(output=str(output), model_type=model_type, members=members, recordings=len(rows),
        template_db_sha256=checksum(output), completeness=database['completeness'], threshold_requires_recalibration=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['campplus','tdnn'], required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditions-id',required=True)
    p.add_argument('--members',nargs='+',default=MVP_MEMBERS)
    p.add_argument('--min-recordings',type=int,default=10)
    a=p.parse_args()
    print(json.dumps(build_templates(a.manifest,a.output,a.members,a.min_recordings,a.conditions_id,a.model),indent=2))
