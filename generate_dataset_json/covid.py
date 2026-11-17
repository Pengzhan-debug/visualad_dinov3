import os
import json


class CovidSolver(object):
    # COVID-19 Radiography: Normal/images + test/images are flat splits.
    # Used as a classification-style set (no masks), Normal -> 0, test -> 1.
    CLSNAMES = ['chest']

    def __init__(self, root='data/covid_19'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        splits = [('Normal', 0), ('test', 1)]

        cls_info = []
        for split, anomaly in splits:
            img_dir = f'{self.root}/{split}/images'
            img_names = sorted(os.listdir(img_dir))
            for img_name in img_names:
                cls_info.append(dict(
                    img_path=f'{split}/images/{img_name}',
                    mask_path='',
                    cls_name=self.CLSNAMES[0],
                    specie_name=split,
                    anomaly=anomaly,
                ))
            print(split, 'images:', len(img_names))

        info['test'][self.CLSNAMES[0]] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        n_normal = sum(1 for e in cls_info if e['anomaly'] == 0)
        print('total:', len(cls_info), 'normal:', n_normal, 'anomaly:', len(cls_info) - n_normal)


if __name__ == '__main__':
    runner = CovidSolver(root='/media/lenovo/sda7/pz/data/covid_19')
    runner.run()
