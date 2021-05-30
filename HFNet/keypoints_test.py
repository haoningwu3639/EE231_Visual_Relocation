import numpy as np
import cv2
from hfnet.datasets.hpatches import Hpatches
from hfnet.evaluation.loaders import sift_loader, export_loader, fast_loader, harris_loader
from hfnet.evaluation.utils.keypoints import keypoints_warp_2D
from hfnet.evaluation.keypoint_detectors import compute_correctness

# from utils import plot_images

data_config = {'make_pairs': True, 'shuffle': True, 'hard': True}
dataset = Hpatches(**data_config)

all_configs = {
    'sift': {
        'predictor': sift_loader,
    },
    'fast': {
        'predictor': fast_loader,
        'do_nms': True,
        'nms_thresh': 4,
    },
    'harris': {
        'predictor': harris_loader,
        'do_nms': True,
        'nms_thresh': 4,
    },
    'superpoint': {
        'experiment': 'super_point_pytorch/hpatches',
        'predictor': export_loader,
        'do_nms': True,
        'nms_thresh': 4,
        'remove_borders': 4,
    },
    'lfnet': {
        'experiment': 'lfnet/hpatches_kpts-500',
        'predictor': export_loader,
    },
    'hfnet': {
        'experiment': 'hfnet-shared_weights-unc_aug-photo_glm-bdd_lrsteps/hpatches',
        'predictor': export_loader,
        'do_nms': True,
        'nms_thresh': 4,
        'remove_borders': 4,
    }
}
eval_config = {'num_features': 500}
configs = {c: {**all_configs[c], **eval_config} for c in ['sift', 'superpoint', 'hfnet']}

def compute_matching_score_single(kpts_w, kpts, matches, vis_w, dist_thresh=3):
    vis_matched = vis_w[matches[:, 0]]
    match_dist = np.linalg.norm(kpts_w[matches[:, 0]] - kpts[matches[:, 1]], axis=-1)
    correct_matches = ((match_dist < dist_thresh)*vis_matched).sum()
    match_score = correct_matches / np.maximum(np.sum(vis_w), 1.0)
    assert vis_matched.sum() == vis_w.sum()
    return match_score, vis_matched.sum(), correct_matches, (match_dist < dist_thresh)

data_iter = dataset.get_test_set()
for _, data in zip(range(1), data_iter):
    for method, config in configs.items():
        pred1 = config['predictor'](data['image'], data['name'], **config)
        pred2 = config['predictor'](data['image2'], data['name2'], **config)
        
        all_matches, matches_dist = matching(
            pred1['descriptors'], pred2['descriptors'],
            do_ratio_test=True, cross_check=False)
              
        kpts1_w, vis1 = keypoints_warp_2D(
            pred1['keypoints'], np.linalg.inv(data['homography']), data['image2'].shape[1::-1])
        kpts2_w, vis2 = keypoints_warp_2D(
            pred2['keypoints'], data['homography'], data['image'].shape[1::-1])
        
        ms, vis_m, cm, good = compute_matching_score_single(kpts1_w, pred2['keypoints'], all_matches, vis1)
        matches = all_matches[good]
        
        # plot_matches(
        #     data['image'].astype(np.uint8), pred1['keypoints'],
        #     data['image2'], pred2['keypoints'], matches, color=(0, 0.67, 0),
        #     dpi=150, ylabel=method, thickness=1.,
        #     title=[f'Score: {ms:.3f}, #Correct: {cm}, #Visible: {vis_m}, #Keypoints: {len(kpts1_w)}'])
        cv2.imwrite('image.png', data['image'].astype(np.uint8))
        cv2.imwrite('image2.png', data['image2'].astype(np.uint8))
        