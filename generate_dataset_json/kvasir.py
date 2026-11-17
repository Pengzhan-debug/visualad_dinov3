import os
import json
import pandas as pd


class ClinicDBSolver(object):
    CLSNAMES = [
        'colon',
    ]

    def __init__(self, root='data/Kvasir'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        img_dir = f'{self.root}/images'
        mask_dir = f'{self.root}/masks'

        # Kvasir is a flat all-anomalous dataset: images/ and masks/ share stems.
        img_names = sorted(os.listdir(img_dir))
        mask_by_stem = {os.path.splitext(n)[0]: n for n in sorted(os.listdir(mask_dir))}

        cls_info = []
        missing = []
        for img_name in img_names:
            stem = os.path.splitext(img_name)[0]
            mask_name = mask_by_stem.get(stem)
            if mask_name is None:
                missing.append(img_name)
                continue
            cls_info.append(dict(
                img_path=f'images/{img_name}',
                mask_path=f'masks/{mask_name}',
                cls_name=self.CLSNAMES[0],
                specie_name='',
                anomaly=1,
            ))

        info['test'][self.CLSNAMES[0]] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('images:', len(img_names), 'paired:', len(cls_info), 'missing_masks:', len(missing))
        if missing:
            print('missing mask examples:', missing[:5])



if __name__ == '__main__':
    runner = ClinicDBSolver(root='/media/lenovo/sda7/pz/data/Kvasir')
    runner.run()
