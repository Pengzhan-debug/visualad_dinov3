import os
import json


class Br35Solver(object):
    # br35h has only tumor images (filenames start with 'y'), no masks.
    CLSNAMES = ['tumor']

    def __init__(self, root='data/br35h'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        for phase in ['train', 'test']:
            phase_dir = f'{self.root}/{phase}'
            img_names = sorted(os.listdir(phase_dir))
            cls_info = []
            for img_name in img_names:
                cls_info.append(dict(
                    img_path=f'{phase}/{img_name}',
                    mask_path='',
                    cls_name=self.CLSNAMES[0],
                    specie_name=self.CLSNAMES[0],
                    anomaly=1,
                ))
            info[phase][self.CLSNAMES[0]] = cls_info
            print(phase, 'images:', len(cls_info))

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == '__main__':
    runner = Br35Solver(root='/media/lenovo/sda7/pz/data/br35h')
    runner.run()
