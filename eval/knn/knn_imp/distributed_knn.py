from functools import partial

import torch
from torch.nn.functional import one_hot, softmax

from eval.knn.knn_imp.metrics import AccuracyAveraging, build_topk_accuracy_metric
from eval.knn.knn_imp.utils import ModelWithNormalize, evaluate, extract_features
from eval.knn.knn_imp.loaders import SamplerType, make_data_loader

def _model_device(model: torch.nn.Module) -> torch.device:
    # Try parameters
    for p in model.parameters(recurse=True):
        return p.device
    # Then buffers
    for b in model.buffers(recurse=True):
        return b.device
    # Fallback
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


class DistributedKnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, nb_knn, T, device, global_rank, global_size, num_classes=1000):
        super().__init__()

        self.global_rank = global_rank
        self.global_size = global_size

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T.to(self.device)
        self.candidates = train_labels.chunk(self.global_size)[self.global_rank].view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        torch.distributed.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]

        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)

        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def create_module_dict(*, module, n_per_class_list, n_tries, nb_knn, train_features, train_labels, samples_per_class):
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules[f"{samples_per_class}"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_train(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}

# def setup_rank_logging(rank):
#     log_filename = f"rank_{rank}_log.txt"
#     print(f"Now writing logs for rank {rank} to file {log_filename}")
#     sys.stdout = open(log_filename, "w")
#     sys.stderr = sys.stdout

def eval_knn(
    model,
    model_name,
    train_dataset,
    val_dataset,
    accuracy_averaging,
    nb_knn,
    pool,
    temperature,
    batch_size,
    num_workers,
    gather_on_cpu,
    global_rank,
    global_size,
    device='cpu',
    n_per_class_list=[-1],
    n_tries=1,
):
    model = ModelWithNormalize(model, model_name, pool)

    print("Extracting features for train set...")

    train_features, train_labels = extract_features(
        model, train_dataset, batch_size, num_workers, global_size=global_size, gather_on_cpu=gather_on_cpu
    )
    print(f"Train features created, shape {train_features.shape}.")

    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )
    num_classes = train_labels.max() + 1
    metric_collection = build_topk_accuracy_metric(accuracy_averaging, num_classes=num_classes)

    partial_module = partial(DistributedKnnModule, global_rank=global_rank, global_size=global_size, T=temperature, device=device, num_classes=num_classes)
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
        samples_per_class=train_dataset.n_samples_per_class,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},
            }
            metrics = {**metrics, **{(n_per_class, t, k): metric_collection.clone() for k in knn_try.nb_knn}}
    model_with_knn = torch.nn.Sequential(model, knn_module_dict)

    # ============ evaluation ... ============
    print("Start the k-NN classification.")
    _, results_dict = evaluate(model_with_knn, val_dataloader, postprocessors, metrics, device)

    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys
            }
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    return results_dict


def eval_knn_with_model(
    trainer,
    model,
    model_name,
    train_dataset,
    val_dataset,
    nb_knn=(10, 20, 100, 200),
    pool='cls',
    temperature=0.07,
    autocast_dtype=torch.float,
    accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
    gather_on_cpu=False,
    batch_size=256,
    num_workers=2,
    n_per_class_list=[200],
    n_tries=1,
):

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_knn = eval_knn(
            model=model,
            model_name=model_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            accuracy_averaging=accuracy_averaging,
            nb_knn=nb_knn,
            temperature=temperature,
            pool=pool,
            batch_size=batch_size,
            num_workers=num_workers,
            gather_on_cpu=gather_on_cpu,
            n_per_class_list=n_per_class_list,
            n_tries=n_tries,
            device=_model_device(model),
            global_rank=trainer.global_rank,
            global_size=trainer.world_size,
        )

    results_dict = {}
    if trainer.is_global_zero:
        print(results_dict_knn)
        for knn_ in results_dict_knn.keys():
            top1 = results_dict_knn[knn_]["top-1"].item() * 100.0
            top5 = results_dict_knn[knn_]["top-5"].item() * 100.0
            results_dict[f"knn_eval {knn_} Top 1"] = top1
            results_dict[f"knn_eval {knn_} Top 5"] = top5
            print(f"{knn_} classifier result: Top1: {top1:.2f} Top5: {top5:.2f}")

    trainer.strategy.barrier()

    return results_dict
