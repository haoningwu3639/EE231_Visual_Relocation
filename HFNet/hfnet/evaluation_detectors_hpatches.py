
from hfnet.datasets.hpatches import Hpatches
from hfnet.evaluation.loaders import sift_loader, export_loader, fast_loader, harris_loader
from hfnet.evaluation.keypoint_detectors import evaluate
from hfnet.utils import tools

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


data_config = {'make_pairs': True, 'shuffle': True}
dataset = Hpatches(**data_config)


all_configs = {
    'hfnet': {
        'experiment': 'hfnet/hpatches',
        'predictor': export_loader,
    },
    'hfnet_ours': {
        'experiment': 'hfnet_ours/predictions_hpatches',
        'predictor': export_loader,
    },
    'sift': {
        'predictor': sift_loader,
        'do_nms': False,
        'nms_thresh': 8,
    },
    'fast': {
        'predictor': fast_loader,
        'do_nms': True,
        'nms_thresh': 8,
    },
    'harris': {
        'predictor': harris_loader,
        'do_nms': True,
        'nms_thresh': 8,
    },
    'superpoint': {
        'experiment': 'super_point_pytorch/hpatches',
        'predictor': export_loader,
        'do_nms': True,
        'nms_thresh': 8,
        'remove_borders': 4,
    },
    'lfnet': {
        'experiment': 'lfnet/hpatches_kpts-500',
        'predictor': export_loader,
    },
}
eval_config = {'correct_match_thresh': 3, 'num_features': 500}



methods = ['hfnet', 'hfnet_ours', 'sift', 'harris', 'fast', 'superpoint', 'lfnet']
configs = {m: all_configs[m] for m in methods}
for method, config in configs.items():
    config = tools.dict_update(config, eval_config)
    data_iter = dataset.get_test_set()
    metrics, _, _, _ = evaluate(data_iter, config, is_2d=True)
    
    print('> {}'.format(method))
    for k, v in metrics.items():
        print('{:<25} {:.3f}'.format(k, v))
    print(config)




