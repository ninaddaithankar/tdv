# Adapted from https://github.com/facebookresearch/dino (Apache 2.0)

import gc

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms as pth_transforms

import utils
from imagenet_dataloader import ImageNetDataset


class ReturnIndexDataset(ImageNetDataset):
    """ImageNetDataset that returns (image, index) instead of (image, label)."""
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img, idx


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 100):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples, return_all_tokens=False).clone()

        # init storage feature matrix on rank 0
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # gather indices from all processes (async)
        y_all = torch.empty(dist.get_world_size(), index.size(0),
                            dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        index_reduce = dist.all_gather(y_l, index, async_op=True)
        index_reduce.wait()
        index_all = torch.cat(y_l)

        # gather features from all processes (async)
        feats_all = torch.empty(dist.get_world_size(), feats.size(0), feats.size(1),
                                dtype=feats.dtype, device=feats.device)
        output_l = list(feats_all.unbind(0))
        output_reduce = dist.all_gather(output_l, feats, async_op=True)
        output_reduce.wait()

        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())

    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels,
                   k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        features = test_features[idx: min(idx + imgs_per_chunk, num_test_images)]
        targets = test_labels[idx: min(idx + imgs_per_chunk, num_test_images)]
        batch_size = targets.shape[0]

        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        correct = predictions.eq(targets.data.view(-1, 1))
        top1 += correct.narrow(1, 0, 1).sum().item()
        top5 += correct.narrow(1, 0, min(5, k)).sum().item()
        total += targets.size(0)

    return top1 * 100.0 / total, top5 * 100.0 / total


def evaluate_knn(encoder, args):
    """
    Run online KNN evaluation on ImageNet.

    Args:
        encoder: backbone model (teacher_without_ddp.backbone). Called with
                 model(x, return_all_tokens=False) → CLS token [B, D].
        args:    training args. Must contain:
                   args.nb_knn           - list of k values, e.g. [10, 20]
                   args.knn_temperature  - softmax temperature (default 0.07)
                   args.knn_use_cuda     - store features on GPU (default True)
                   args.batch_size_per_gpu
                   args.num_workers

    Returns:
        dict mapping e.g. "knn_10_top1" → float, logged on rank-0 only.
    """
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = ReturnIndexDataset(hparams=None, split="train", transform=transform)
    dataset_val = ReturnIndexDataset(hparams=None, split="validation", transform=transform)

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Val: no sampler — all ranks process the full val set (standard dino pattern).
    # Rank-0 stores features; other ranks' results are discarded.
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"KNN data: {len(dataset_train)} train / {len(dataset_val)} val images.")

    was_training = encoder.training
    encoder.eval()

    print("Extracting train features...")
    train_features = extract_features(encoder, loader_train, args.knn_use_cuda)
    print("Extracting val features...")
    test_features = extract_features(encoder, loader_val, args.knn_use_cuda)

    encoder.train(was_training)

    results = {}
    if utils.get_rank() == 0:
        train_features = F.normalize(train_features, dim=1, p=2)
        test_features = F.normalize(test_features, dim=1, p=2)

        train_labels = torch.tensor(dataset_train.ds["label"]).long()
        test_labels = torch.tensor(dataset_val.ds["label"]).long()

        if args.knn_use_cuda:
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features ready. Running k-NN classification...")
        for k in args.nb_knn:
            top1, top5 = knn_classifier(
                train_features, train_labels,
                test_features, test_labels,
                k, args.knn_temperature,
            )
            print(f"  {k}-NN  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")
            results[f"knn_{k}_top1"] = top1
            results[f"knn_{k}_top5"] = top5

    if dist.is_initialized():
        dist.barrier()

    del train_features, test_features
    gc.collect()
    torch.cuda.empty_cache()

    return results
