from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection

from eval.knn.knn_imp.adapters import DatasetWithEnumeratedTargets
from eval.knn.knn_imp.loaders import SamplerType, make_data_loader
from eval.knn.knn_imp.helpers import MetricLogger
from model.model_utils import encode_images


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model, model_name, pool="cls"):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.pool = pool

    def forward(self, samples):
        output = encode_images(samples, encoder=self.model, encoder_name=self.model_name)

        if self.pool == "cls":
            output = output[:, 0]
        elif self.pool == "avg":
            output = output[:, 1:].mean(dim=1)

        return nn.functional.normalize(output, dim=-1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 50, header):
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank, global_size):
    tensor_all_ranks = torch.empty(
        global_size,
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())

    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, batch_size, num_workers, global_size, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=True
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, global_size, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, global_size, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")

    metric_logger = MetricLogger(delimiter="  ")

    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 100):
        # print(f"[rank {torch.distributed.get_rank()}] STARTING with this iteration, going onto next batch...")
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        features_rank = model(samples).float()

        # print(f"[rank {torch.distributed.get_rank()}] index: {index}, features: {features_rank.shape}")
        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            print(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index, global_size).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank, global_size).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank, global_size).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

        # print(f"[rank {torch.distributed.get_rank()}] DONE with this iteration, going onto next batch...")

    print(f"Features shape: {tuple(features.shape)}")
    print(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels
