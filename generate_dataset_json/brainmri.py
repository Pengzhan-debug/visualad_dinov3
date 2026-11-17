import os
import json


class BrainMRISolver(object):
    CLSNAMES = ['brainmri']

    def __init__(self, root='data/brain_mri'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        base_dir = f'{self.root}/brainmri'  # contains no/ and yes/

        cls_info = []
        for specie in ['no', 'yes']:
            specie_dir = f'{base_dir}/{specie}'
            is_abnormal = specie != 'no'
            img_names = sorted(os.listdir(specie_dir))
            for img_name in img_names:
                cls_info.append(dict(
                    img_path=f'brainmri/{specie}/{img_name}',
                    mask_path='',
                    cls_name=self.CLSNAMES[0],
                    specie_name=specie,
                    anomaly=1 if is_abnormal else 0,
                ))

        info['test'][self.CLSNAMES[0]] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        n_normal = sum(1 for e in cls_info if e['anomaly'] == 0)
        print('total:', len(cls_info), 'normal:', n_normal, 'anomaly:', len(cls_info) - n_normal)


if __name__ == '__main__':
    runner = BrainMRISolver(root='/media/lenovo/sda7/pz/data/brain_mri')
    runner.run()
