import os
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import rasterio
from data.utils_data.io import DATA_DIR


def generate_miou(hparams: str, path_truth: str, path_pred: str) -> list:
    #################################################################################################

    def calc_miou(cm_array):
        """
        Specifically, we calculate the per-patch confusion matrix and per-class
        Intersection over Union (IoU) without excluding pixels belonging to
        the 'other' class, even though they represent a marginal part of the test-set.
        However, when computing the mean IoU (mIoU), we do remove the IoU of the 'other'
        class due to its association with majority or lower quality level pixels or very
        underrepresented land cover.
        """
        m = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.diag(cm_array) / (
                cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array)
            )
        m = np.nansum(ious[:-1]) / (np.logical_not(np.isnan(ious[:-1]))).sum()
        return m.astype(float), ious[:-1]

    #################################################################################################

    patch_confusion_matrices = []

    for gt_path in tqdm(path_truth, desc=f"Metrics", unit="img"):
        gt_path = Path(gt_path)
        pred_path = Path(path_pred) / f"PRED_{gt_path.name}"
        channel = hparams["labels_configs"].get("label_channel_nomenclature", 1)
        with rasterio.open(os.path.join(DATA_DIR, gt_path), "r") as src_gt:
            target = src_gt.read(channel)
        with rasterio.open(pred_path, "r") as src_pred:
            pred = src_pred.read(1)

        target = torch.from_numpy(target)
        target = target.detach().cpu().numpy()

        patch_confusion_matrices.append(
            confusion_matrix(
                target.flatten(),
                pred.flatten(),
                labels=list(range(hparams["inputs"]["num_classes"])),
            )
        )
    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    mIou, ious = calc_miou(sum_confmat)
    return mIou, ious


def generate_mf1s(hparams, path_truth: str, path_pred: str) -> list:
    #################################################################################################
    def get_data_paths(path, filter):
        for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()

    def get_confusion_metrics(confusion_matrix):
        """Computes confusion metrics out of a confusion matrix (N classes)
        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]
        Returns
        -------
        metrics : dict
            a dictionary holding all computed metrics
        Notes
        -----
        Metrics are: 'percentages', 'precisions', 'recalls', 'f1s', 'mf1', 'oa'
        """
        tp = np.diag(confusion_matrix)
        tp_fn = np.sum(confusion_matrix, axis=0)
        tp_fp = np.sum(confusion_matrix, axis=1)

        has_no_rp = tp_fn == 0
        has_no_pp = tp_fp == 0

        tp_fn[has_no_rp] = 1
        tp_fp[has_no_pp] = 1

        percentages = tp_fn / np.sum(confusion_matrix)
        precisions = tp / tp_fp
        recalls = tp / tp_fn

        p_zero = precisions == 0
        precisions[p_zero] = 1

        f1s = 2 * (precisions * recalls) / (precisions + recalls)
        ious = tp / (tp_fn + tp_fp - tp)

        precisions[has_no_pp] *= 0.0
        precisions[p_zero] *= 0.0
        recalls[has_no_rp] *= 0.0

        f1s[p_zero] *= 0.0
        f1s[percentages == 0.0] = np.nan
        ious[percentages == 0.0] = np.nan

        mf1 = np.nanmean(f1s[:-1])
        miou = np.nanmean(ious[:-1])

        oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        metrics = {
            "percentages": percentages,
            "precisions": precisions,
            "recalls": recalls,
            "f1s": f1s,
            "mf1": mf1,
            "ious": ious,
            "miou": miou,
            "oa": oa,
        }
        return metrics

    patch_confusion_matrices = []

    for gt_path in tqdm(path_truth, desc=f"Metrics", unit="img"):
        gt_path = Path(gt_path)
        pred_path = Path(path_pred) / f"PRED_{gt_path.name}"
        channel = hparams["labels_configs"].get("label_channel_nomenclature", 1)
        with rasterio.open(os.path.join(DATA_DIR, gt_path), "r") as src_gt:
            target = src_gt.read(channel)
        with rasterio.open(pred_path, "r") as src_pred:
            pred = src_pred.read(1)

        target = torch.from_numpy(target)
        target = target.detach().cpu().numpy()

        patch_confusion_matrices.append(
            confusion_matrix(
                target.flatten(),
                pred.flatten(),
                labels=list(range(hparams["inputs"]["num_classes"])),
            )
        )
    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    metrics = get_confusion_metrics(sum_confmat)
    return metrics["mf1"], metrics["f1s"], metrics["oa"]
