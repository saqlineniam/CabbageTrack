"""
Read-only download of every image in the Roboflow project together with its ORIGINAL
(un-augmented) annotation. Nothing in the Roboflow project is created or modified.

Output: data/detector/cabbage/roboflow/images/<id>.jpg, data/detector/cabbage/roboflow/ann/<id>.json, data/detector/cabbage/roboflow/manifest.csv

Usage (the key is read from the environment, never stored):
  set ROBOFLOW_API_KEY=<your key>
  python detector/fetch_roboflow.py --project drone-shot --out data/detector/cabbage/roboflow
  python detector/fetch_roboflow.py --project cabbage-young --include-unassigned --out data/detector/young_cabbage/roboflow
  python detector/fetch_roboflow.py --project cauliflower-annotation-for-cabba --include-unassigned --out data/detector/cauliflower/roboflow
The Zenodo record already contains these downloads, so this step is only needed to re-fetch from Roboflow.
"""
import argparse
import json
import os
import time

import pandas as pd
import requests
from roboflow import Roboflow

WS, PROJ = 'niam-q7thy', 'drone-shot'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default=os.environ.get('ROBOFLOW_API_KEY'), help='default: environment variable ROBOFLOW_API_KEY')
    ap.add_argument('--out', default='data/detector/cabbage/roboflow')
    ap.add_argument('--project', default=PROJ)
    ap.add_argument('--include-unassigned', action='store_true',
                    help='also fetch annotated images not yet added to a train/valid/test split')
    a = ap.parse_args()
    if not a.key:
        ap.error('set ROBOFLOW_API_KEY (or pass --key)')
    proj = a.project
    os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'ann'), exist_ok=True)
    p = Roboflow(api_key=a.key).workspace(WS).project(proj)

    items, off = [], 0
    while True:
        page = p.search(offset=off, limit=100, fields=['id', 'name', 'split', 'created', 'labels', 'tags'])
        if not page:
            break
        items += page
        off += len(page)
        if len(page) < 100:
            break
    print(f'{len(items)} images in project')
    print('splits:', pd.Series([str(it.get('split')) for it in items]).value_counts().to_dict())
    if not a.include_unassigned:
        items = [it for it in items if it.get('split') in ('train', 'valid', 'test')]
        print(f'{len(items)} images assigned to the dataset (train/valid/test); unassigned uploads are skipped')

    rows = []
    for k, it in enumerate(items):
        iid = it['id']
        jp = os.path.join(a.out, 'ann', f'{iid}.json')
        ip = os.path.join(a.out, 'images', f'{iid}.jpg')
        if not os.path.exists(jp):
            for attempt in range(3):
                try:
                    r = requests.get(f'https://api.roboflow.com/{WS}/{proj}/images/{iid}', params={'api_key': a.key}, timeout=60)
                    r.raise_for_status()
                    break
                except Exception:
                    time.sleep(2)
            img = r.json().get('image', {})
            json.dump(img, open(jp, 'w'))
        img = json.load(open(jp))
        if not os.path.exists(ip):
            rr = requests.get(img['urls']['original'], timeout=120)
            rr.raise_for_status()
            open(ip, 'wb').write(rr.content)
        ann = img.get('annotation') or {}
        boxes = ann.get('boxes', []) if isinstance(ann, dict) else []
        counts = {}
        for b in boxes:
            counts[b.get('label')] = counts.get(b.get('label'), 0) + 1
        rows.append({'id': iid, 'name': img.get('name'), 'roboflow_split': img.get('split'), 'created': it.get('created'),
                     'width': ann.get('width'), 'height': ann.get('height'), 'n_objects': len(boxes),
                     'n_polygons': sum(1 for b in boxes if b.get('type') == 'polygon'),
                     **{f'n_{c}': n for c, n in counts.items()}})
        if (k + 1) % 50 == 0:
            print(f'  {k + 1}/{len(items)}', flush=True)
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(os.path.join(a.out, 'manifest.csv'), index=False)
    print(df.drop(columns=['id', 'name', 'created']).describe().round(1).to_string())


if __name__ == '__main__':
    main()
