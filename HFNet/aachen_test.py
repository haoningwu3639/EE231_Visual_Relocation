import cv2
import numpy as np
from pathlib import Path
import yaml

from hfnet.datasets.aachen import Aachen
from hfnet.evaluation.localization import Localization
from hfnet.evaluation.utils.db_management import read_query_list
from hfnet.evaluation.loaders import export_loader
from hfnet.settings import DATA_PATH, EXPER_PATH

from notebooks.utils import plot_matches, draw_matches

config_global = {
    'db_name': 'globaldb_hfnet.pkl',
    'experiment': 'hfnet/aachen',
    'predictor': export_loader, 
    'has_keypoints': False, 
    'has_descriptors': False, 
    'pca_dim': 1024,
    'num_prior': 10,
}

config_local = {
    'db_name': 'localdb_hfnet.pkl',
    'experiment': 'hfnet/aachen',
    'predictor': export_loader,
    'has_keypoints': True,
    'has_descriptors': True,
#    'do_nms': True,
#    'nms_thresh': 4,
    'num_features': 2000,
    'ratio_thresh': 0.9,
}
model = 'hfnet_model'

config_pose = {
    'reproj_error': 10,
    'min_inliers': 12,
}
config = {'global': config_global, 'local': config_local, 'pose': config_pose}
loc = Localization('aachen', model, config)
queries = read_query_list(Path(loc.base_path, 'night_time_queries_with_intrinsics.txt'))

np.random.RandomState(0).shuffle(queries)
query_dataset = Aachen(**{'resize_max': 960,
                          'image_names': [q.name for q in queries]})
def get_image(name):
    path = Path(DATA_PATH, query_dataset.dataset_folder, name)
    return cv2.imread(path.as_posix())[..., ::-1]

query_iter = query_dataset.get_test_set()

for i, query_info, query_data in zip(range(20), queries, query_iter):
    results, debug = loc.localize(query_info, query_data, debug=True)
    s = f'{i} {"Success" if results.success else "Failure"}, inliers {results.num_inliers:^4}, ' \
        + f'ratio {results.inlier_ratio:.3f}, landmarks {len(debug["matching"]["lm_frames"]):>4}, ' \
        + f'spl {debug["index_success"]:>2}, places {[len(p) for p in debug["places"]]:}, ' \
        + f'pos {[f"{n:.1f}" for n in results.T[:3, 3]]}'
    print(s)
    
    sorted_frames, counts = np.unique(
        [debug['matching']['lm_frames'][m2] for m1, m2 in debug['matches'][debug['inliers']]],
        return_counts=True)
    best_id = sorted_frames[np.argmax(counts)]

    query_image = get_image(query_info.name)
    best_image = get_image(loc.images[best_id].name)
    best_matches_inliers = [(m1, debug['matching']['lm_indices'][m2]) 
                            for m1, m2 in debug['matches'][debug['inliers']] 
                            if debug['matching']['lm_frames'][m2] == best_id]
    best_matches_outliers = [(m1, debug['matching']['lm_indices'][m2])
                            for i, (m1, m2) in enumerate(debug['matches']) 
                            if debug['matching']['lm_frames'][m2] == best_id
                            and i not in debug['inliers']]

    # plot_matches(
    #     query_image, debug['query_item'].keypoints,
    #     best_image, loc.local_db[best_id].keypoints,
    #     np.array(best_matches_inliers), color=(0, 1., 0),
    #     dpi=100, ylabel=str(i), thickness=1.)

    temp = draw_matches(
        query_image, debug['query_item'].keypoints,
        best_image, loc.local_db[best_id].keypoints,
        np.array(best_matches_inliers), color=(0, 255, 0),
        thickness=1)
    cv2.imwrite("./aachen_localization_test/" + str(i) + ".png", cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))