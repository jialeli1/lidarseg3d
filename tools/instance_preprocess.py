import argparse
from det3d.datasets.semantickitti import SemanticKITTIDataset


if __name__ == '__main__':
    # instance preprocessing
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-d', '--data_path', default='data')
    # parser.add_argument('-o', '--out_path', default='data')
    # args = parser.parse_args()
    
    data_root = "data/SemanticKITTI/dataset/sequences" #"data/nuScenes"
    out_path = 'data/SemanticKITTI/dataset'
    instance_pkl = "data/SemanticKITTI/dataset/instance_path.pkl"
    train_seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']


    train_pt_dataset = SemanticKITTIDataset(
        info_path=None,
        root_path=data_root,
        sequences=train_seq,
        test_mode=False,
    )
    train_pt_dataset.save_instance(out_path)

    print('==> Instance preprocessing finished.')