import motmetrics as mm
import numpy as np

def compute_mot_metrics(pred_tracks, gt_tracks):
    acc = mm.MOTAccumulator(auto_id=True)

    # Collect all frame indices (union of preds and gts)
    all_frames = sorted(set(t for t, *_ in pred_tracks) | set(gt_tracks.keys()))

    for t in all_frames:
        # Predictions: (frame, id, x1, y1, x2, y2)
        preds = [(tid, x1, y1, x2, y2) for t_, tid, x1, y1, x2, y2 in pred_tracks if t_ == t]
        pred_ids = [p[0] for p in preds]
        pred_boxes = np.array([p[1:] for p in preds]) if preds else np.empty((0, 4))

        # Ground truth: (id, x1, y1, x2, y2)
        gts = gt_tracks.get(t, [])
        gt_ids = [g[0] for g in gts]
        gt_boxes = np.array([g[1:] for g in gts]) if gts else np.empty((0, 4))

        # IoU distance matrix
        dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

        # Update MOT accumulator
        acc.update(gt_ids, pred_ids, dists)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'mota', 'num_switches'], name='mot_eval')

    return {
        'mot_eval/idf1': summary.loc['mot_eval']['idf1'],
        'mot_eval/mota': summary.loc['mot_eval']['mota'],
        'mot_eval/id_switches': summary.loc['mot_eval']['num_switches'],
    }
