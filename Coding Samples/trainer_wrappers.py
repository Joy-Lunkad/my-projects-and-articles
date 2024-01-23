from __future__ import annotations

import argparse
import functools
import importlib
import os
import warnings
from glob import glob
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Text,
    Tuple,
    Union,
)

import dill
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import natsort
import numpy as np
import optax
import rich
import tensorflow as tf
import tensorflow_datasets as tfds
import tree
from jaxline import utils as jl_utils
from ml_collections import config_dict

import functions.utils as utils
import wandb
from datasets import all_datasets
from functions.imagenet_class_list import imagenet_classlist

warnings.simplefilter(action="ignore", category=FutureWarning)
from etils.etree import jax as etree

from functions import (
    memory_utils,
    tpu_utils,
    utils,
    memory_assets,
    meta_batching,
    wb_utils,
)

# TYPE CHECKING
from ml_collections.config_dict.config_dict import ConfigDict
from haiku import TransformedWithState
from jax.random import KeyArray
from optax import GradientTransformation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainers.train_aim import Args

Array = jax.Array
Batch = Mapping[Text, np.ndarray]


def build_config_with_args(args: Args):
    restore_path = (
        args.restore_path if args.restore_path else config_dict.placeholder(str)
    )

    get_config = importlib.import_module(f"configs.{args.config_file}").get_config

    if not args.local:
        zone = tpu_utils.get_tpu_vm_zone()
        print("zone", zone)
        if zone == "europe-west4-a":
            bucket = "eu-aim-exps"
        elif zone == "us-central1-f":
            bucket = "us-aim-exps"
        else:
            raise ValueError(f"Zone {zone} not supported")
        print("Bucket", bucket)
    elif args.local and args.real_data:
        zone = "us-central1-f"
        bucket = "us-aim-exps"
    else:
        zone = None
        bucket = None

    print("args.eval_batch_size = ", args.eval_batch_size)

    full_config = get_config(
        exp_no=args.exp_no,
        bucket=bucket,
        restore_path=restore_path,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        stable_memory_decay=args.stable_memory_decay,
        dropout=args.dropout,
        warmup_epochs=args.warmup_epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        bfloat16=args.bfloat16,
        scale_lr_with_bs=args.scale_lr_with_bs,
    )

    config = full_config.experiment_kwargs.config
    full_config.include_top = args.include_top
    full_config.restore_path_bucket = args.restore_path_bucket
    full_config.transfer_params = args.transfer_params
    full_config.keep_latest_checkpoint_only = args.keep_latest
    config.dataset_configs.train_config.augment_name = args.train_augment_name
    config.dataset_configs.train_config.augment_before_mix = not args.augment_before_mix
    config.log_every = args.log_every
    config.eval_every = args.eval_every
    config.checkpoint_every = args.checkpoint_every
    config.save_all = args.save_all
    config.freeze_c_layers = args.freeze_c_layers
    config.memory_config = config.model_kwargs.memory_config
    config.use_slow_memory = config.memory_config.get("optimizer", None) is not None
    config.slow_lr_multiplier = args.slow_lr_multiplier  # TODO: add to memory config
    print("Restore Path = ", full_config.restore_path)

    if zone == "europe-west4-a":
        print(f"Changing configs to {zone}")
        data_dir = "gs://aimimagenet"
        config.dataset_configs.train_config.data_dir = data_dir
        for key in config.dataset_configs.eval_configs.keys():
            config.dataset_configs.eval_configs[key].data_dir = data_dir
        full_config.restore_path_bucket = "eu-aim-exps"
        full_config.data_dir = data_dir

    if zone == "us-central1-f":
        print(f"Changing configs to {zone}")
        data_dir = "gs://us-data-dir"
        config.dataset_configs.train_config.data_dir = data_dir
        for key in config.dataset_configs.eval_configs.keys():
            config.dataset_configs.eval_configs[key].data_dir = data_dir
        full_config.restore_path_bucket = "us-aim-exps"
        full_config.data_dir = data_dir

    if zone == None:
        args.use_fake_data = True

    if args.use_fake_data:
        config.dataset_configs.train_config.fake_data = True
        for key in config.dataset_configs.eval_configs.keys():
            config.dataset_configs.eval_configs[key].fake_data = True

    config.use_c = True

    print("config.model_kwargs.verbose = ", config.model_kwargs.verbose)
    print("args.verbose = ", args.verbose)
    config.model_kwargs.verbose = args.verbose

    print("config.build_on_top = ", config.build_on_top)

    config.sam_rho = args.sam_rho
    config.run_name = args.run_name
    full_config.hack_include_top = args.hack_include_top
    config.local = args.local
    config.return_magnitudes = args.return_magnitudes
    config.model_kwargs.return_c_mag = args.return_magnitudes

    config.zone = zone
    config.name = (
        f"{config.model_class}-{args.exp_no}-{config.exp_type}-{args.run_name}"
    )

    config.bfloat16 = args.bfloat16  # TODO: FIX THIS, Why Twice
    config.decouple_wd = args.decouple_wd
    config.train_vit = args.train_vit

    return full_config, config


class EXPERIMENT_STATE(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    ema_params: Optional[hk.Params]
    ema_state: Optional[hk.State]


def _build_train_input(config):
    num_devices = jax.device_count()
    global_batch_size = config.train_batch_size
    bs_per_device, ragged = divmod(global_batch_size, num_devices)
    print("train_dataset_config -> ")
    print(config.dataset_configs.train_config)
    if ragged:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"num devices {num_devices}"
        )
    return all_datasets.load(
        bs_per_device=bs_per_device,
        batch_dims=[jax.local_device_count(), bs_per_device],
        **config.dataset_configs.train_config,
    )


def _forward_fn(inputs, is_training, config, **kwargs):

    model_module = importlib.import_module(f"models.{config.model_file}")
    net_kwargs = {
        "num_classes": config.dataset_configs.train_config.num_classes,
        **config.model_kwargs,
    }
    net = getattr(model_module, config.model_class)(**net_kwargs)
    if config.get("transpose", False):
        images = jnp.transpose(inputs["images"], (3, 0, 1, 2))  # HWCN -> NHWC
    else:
        images = inputs["images"]
    if config.bfloat16 and is_training:
        images = utils.to_bf16(images)

    return net(images, is_training=is_training, **kwargs)


def build_schedule_fn(lr_schedule):
    which_schedule = getattr(optax, lr_schedule.name)
    schedule = which_schedule(**lr_schedule.kwargs)

    def _get_learning_rate(step) -> float:
        return schedule(step)

    return _get_learning_rate


def _optimizer(learning_rate: jnp.ndarray, optimizer_config):

    optimizer_fn = getattr(optax, optimizer_config.name)

    if "adamW" in optimizer_config.name:
        mask = functools.partial(
            utils.params_label_fn,
            layers_to_change=optimizer_config.get("memory_layers", []) + ["bn", "ln"],
            labels=[True, False],
        )
        optimizer_config.kwargs["mask"] = mask

    GC = optimizer_config.get("clip_grads", False)
    AGC = optimizer_config.get("AGC", False)

    if AGC:
        print(f"Adaptive Clip Grads: {AGC}")
        clip_fn = optax.adaptive_grad_clip(AGC)
    elif GC:
        print(f"Clip Grads: {GC}")
        clip_fn = optax.clip(max_delta=GC)
    else:
        clip_fn = optax.identity()

    print(f"Optimizer: {optimizer_config.name}")
    print(f"Optimizer kwargs: {optimizer_config.kwargs}")

    optimizer = optax.chain(
        clip_fn,
        optimizer_fn(learning_rate=learning_rate, **optimizer_config.kwargs),
    )

    multi_steps: int = optimizer_config.get("multi_steps", 0)
    if multi_steps:
        print(f"Using Multi-Step Grad Accumulation: {multi_steps}")
        opt = optax.MultiSteps(optimizer, every_k_schedule=multi_steps)
        # optimizer = (opt.init, opt.update)
        optimizer = optax.GradientTransformation(init=opt.init, update=opt.update)  # type: ignore
    return optimizer


def _freeze_layers(_params, config, _optimizer):
    rich.print("Output of label fn: ", config.label_fn(_params))
    opt = _optimizer(jl_utils.bcast_local_devices(jnp.zeros([], jnp.int32)))
    tx = optax.multi_transform(
        {"trainable": opt, "frozen": optax.set_to_zero()}, config.label_fn
    )
    _opt_state = jax.pmap(tx.init)(_params)
    return _opt_state


def _use_fast_slow_train(_params, config, _optimizer):
    rich.print("Output of label fn: ", config.label_fn_fast_slow(_params))
    opt_1 = _optimizer(jl_utils.bcast_local_devices(jnp.zeros([], jnp.int32)))
    opt_2 = _optimizer(jl_utils.bcast_local_devices(jnp.zeros([], jnp.int32)))
    tx = optax.multi_transform(
        {"type_1": opt_1, "type_2": opt_2}, config.label_fn_fast_slow
    )
    _opt_state = jax.pmap(tx.init)(_params)
    return _opt_state


def _use_freeze_fast_slow(_params, config, optimizer):
    rich.print("Output of label fn: ", config.label_fn_freeze_fast_slow(_params))
    opt_1 = optimizer(jl_utils.bcast_local_devices(jnp.zeros([], jnp.int32)))
    opt_2 = optimizer(jl_utils.bcast_local_devices(jnp.zeros([], jnp.int32)))
    tx = optax.multi_transform(
        {"type_1": opt_1, "type_2": opt_2, "frozen": optax.set_to_zero()},
        config.label_fn_freeze_fast_slow,
    )
    _opt_state = jax.pmap(tx.init)(_params)
    return _opt_state


def get_freeze_fast_slow_label_fn(config: Any) -> Callable:

    freeze_label_fn = config.label_fn
    fast_slow_label_fn = config.label_fn_fast_slow

    def label_fn_freeze_fast_slow(params: hk.Params) -> hk.Params:
        labels = freeze_label_fn(params)
        fast_slow_labels = fast_slow_label_fn(params)

        for (mod, var), label in tree.flatten_with_path(labels):  # type: ignore
            if label == "trainable":
                labels[mod][var] = fast_slow_labels[mod][var]
        return labels

    return label_fn_freeze_fast_slow


def init_model(net, init_rng, inputs):
    init_net = jax.pmap(
        lambda *a: net.init(*a, is_training=True),
        axis_name="i",
    )
    init_key = jl_utils.bcast_local_devices(init_rng)
    _params, _state = init_net(init_key, inputs)
    return _params, _state, init_net, init_key


def _get_opt(_params, _optimizer):
    opt_init, _ = _optimizer(jl_utils.bcast_local_devices(jnp.zeros([], jnp.int32)))
    _opt_state = jax.pmap(opt_init)(_params)
    return _opt_state


def get_train_input(_build_train_input):
    _train_input = _build_train_input()
    inputs = next(_train_input)

    wb_utils.upload_sample_to_wandb(
        image=inputs["images"][0][0], label=int(inputs["labels"][0][0])
    )

    return _train_input, inputs


def print_param_metrics(_params: hk.Params):
    num_params = hk.data_structures.tree_size(_params)
    num_params = num_params / jax.local_device_count()
    print(f"Net parameters: {num_params}")

    params_count: Any = jtu.tree_map(lambda x: x.size, jl_utils.get_first(_params))

    db_values_params = utils.count_params(params_count, var_name="db_values")
    print(f"db_values_params: {db_values_params:,}")

    db_keys_params = utils.count_params(params_count, var_name="db_keys")
    print(f"db_keys_params: {db_keys_params:,}")

    retrieve_memory_params = utils.count_params(
        params_count, mod_name="retrieve_memory"
    )
    print(f"retrieve_memory_params: {retrieve_memory_params:,}")

    apply_memory_params = utils.count_params(params_count, mod_name="apply_memory")
    print(f"apply_memory_params: {apply_memory_params:,}")

    consolidator_params = utils.count_params(params_count, mod_name="consolidator")
    print(f"consolidator_params: {consolidator_params:,}")

    model_size_metrics = {
        "num_params": f"{num_params / 1e6:.3f}",
        "db_values_params": f"{db_values_params / 1e6:.3f}",
        "db_keys_params": f"{db_keys_params / 1e6:.2f}",
        "retrieve_memory_params": f"{retrieve_memory_params / 1e6:.3f}",
        "apply_memory_params": f"{apply_memory_params / 1e6:.3f}",
        "consolidator_params": f"{consolidator_params / 1e6:.2f}",
    }
    rich.print("model_size_metrics", model_size_metrics)
    wandb.config.update(model_size_metrics, allow_val_change=True)


def get_microbatch(inputs, micro_batch_size, train_batch_size):
    # Input validation
    print("num_devices: ", len(jax.devices()))
    batch_size = train_batch_size // len(jax.devices())
    micro_batch_size = micro_batch_size // len(jax.devices())
    if batch_size % micro_batch_size != 0:
        raise ValueError(
            f"batch_size {batch_size} should be divisible by micro_batch_size {micro_batch_size}"
        )
    # Rest of the input validation and code
    print("Inputs.shape", inputs["images"].shape, inputs["labels"].shape)
    print(f"Using microbatch size {micro_batch_size} with batch size {batch_size}")

    images = inputs["images"]
    labels = inputs["labels"]
    learning_rate = inputs.get("learning_rate", 0.1)  # TODO: remove this default
    microbatches = []
    for i in range(0, batch_size, micro_batch_size):
        microbatch_inputs = {
            "images": images[i : i + micro_batch_size],
            "labels": labels[i : i + micro_batch_size],
            "learning_rate": learning_rate,
        }
        microbatches.append(microbatch_inputs)
    return microbatches


def get_microbatches(inputs, config):
    get_microbatch_fn = functools.partial(
        get_microbatch,
        micro_batch_size=config.micro_batch_size,
        train_batch_size=config.train_batch_size,
    )
    return jax.vmap(get_microbatch_fn, out_axes=0)(inputs)


def meta_batch_inputs(_train_input, config) -> Generator[Batch, None, None]:

    if config.memory_config.meta_batching.get("fake_meta_batching"):
        return _train_input

    num_devices = jax.device_count()
    global_batch_size = config.train_batch_size
    bs_per_device, ragged = divmod(global_batch_size, num_devices)

    if ragged:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"num devices {num_devices}"
        )

    n = config.memory_config.meta_batching.get("num_meta_batches", num_devices)

    bg = meta_batching.BufferGenerator(
        _train_input=_train_input,
        buffer_size=config.memory_config.meta_batching.buffer_size,
        batch_size=bs_per_device,
        num_devices=num_devices,
        num_meta_batches=n,
        min_overlap=config.memory_config.meta_batching.min_overlap,
        num_classes=config.dataset_configs.train_config.num_classes,
        im_size=config.image_size,
    )

    meta_gen = bg.generate_batches()

    data = next(meta_gen)

    output_types = {k: tf.as_dtype(v.dtype) for k, v in data.items()}
    output_shapes = {k: tf.TensorShape(v.shape) for k, v in data.items()}

    ds_meta_gen = tf.data.Dataset.from_generator(
        lambda: meta_gen,
        output_types=output_types,
        output_shapes=output_shapes,
    )

    ds_meta_gen = ds_meta_gen.prefetch(tf.data.experimental.AUTOTUNE)
    ds_meta_gen_numpy = ds_meta_gen.as_numpy_iterator()

    return ds_meta_gen_numpy  # type: ignore


def _initialize_train(
    init_rng,
    config,
    full_config,
    _build_train_input,
    net,
    _optimizer,
    get_train_input=get_train_input,
    get_microbatches=get_microbatches,
    init_model=init_model,
    _get_opt=_get_opt,
):

    _train_input, inputs = get_train_input(_build_train_input)

    if config.memory_config.get("use_meta_batching"):
        _train_input = meta_batch_inputs(_train_input, config)

    if config.get("num_repeat_c_steps"):
        inputs = get_microbatches(inputs, config)[0]

    if full_config.include_top:
        return _train_input

    _params, _state, init_net, init_key = init_model(net, init_rng, inputs)
    print_param_metrics(_params)
    if config.freeze_layers and config.get("use_fast_slow_train"):
        _opt_state = _use_freeze_fast_slow(_params, config, _optimizer)
    elif config.freeze_layers:
        _opt_state = _freeze_layers(_params, config, _optimizer)
    elif config.get("use_fast_slow_train"):
        _opt_state = _use_fast_slow_train(_params, config, _optimizer)
    else:
        _opt_state = _get_opt(_params, _optimizer)
    if config.use_ema:
        _ema_params, _ema_state = init_net(init_key, inputs)
    else:
        _ema_params, _ema_state = None, None
    train_state = EXPERIMENT_STATE(_params, _state, _opt_state, _ema_params, _ema_state)
    return _train_input, train_state


def _one_hot(value, config):
    """One-hot encoding potentially over a sequence of labels."""
    y = jax.nn.one_hot(
        x=value, num_classes=config.dataset_configs.train_config.num_classes
    )
    return y


def l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def _get_normal_pytree(pytree, config):
    memory_layers = config.memory_config.memory_layers
    exceptions = memory_layers + ["bn", "ln"]
    return utils._get_sub_pytrees(pytree, ["ResNet"], exceptions)


def _get_memory_pytree(pytree, config):
    # TODO: TEST WITH CONSOLIDATOR PARAMS IN MEMORY PARAMS.
    memory_layers = config.memory_config.memory_layers
    return utils._get_sub_pytrees(pytree, memory_layers, ["bn", "ln"])


def _get_memory_components_pytree(pytree, config):
    # TODO: TEST WITH CONSOLIDATOR PARAMS IN MEMORY PARAMS.
    memory_layers = config.memory_config.memory_component_layers
    return utils._get_sub_pytrees(pytree, memory_layers, ["bn", "ln"])


def process_mix_labels(y, inputs, _one_hot):
    print("Using mixup or cutmix!")
    y1 = _one_hot(inputs["mix_labels"])
    y = inputs["ratio"][:, None] * y + (1.0 - inputs["ratio"][:, None]) * y1
    return y


def apply_label_smoothing(y, config):
    spositives = 1.0 - config.label_smoothing
    snegatives = (
        config.label_smoothing / config.dataset_configs.train_config.num_classes
    )
    y = spositives * y + snegatives
    return y


def get_l2_loss(params: hk.Params, config: Any) -> Dict[str, Array]:
    l2_params = _get_normal_pytree(params, config)
    l2_params_memory = _get_memory_pytree(params, config)

    memory_weight_decay_loss = config.optimizer.stable_memory_decay * l2_loss(
        l2_params_memory
    )

    train_weight_decay_loss = config.optimizer.weight_decay * l2_loss(l2_params)

    return {
        "decay_loss": memory_weight_decay_loss + train_weight_decay_loss,
        "weight_decay_loss": train_weight_decay_loss,
        "memory_decay_loss": memory_weight_decay_loss,
    }


def gather_metrics(
    logits,
    inputs,
    loss,
    prefix,
    **kwargs,
):
    metrics = utils.topk_correct(logits, inputs["labels"], prefix=prefix)
    # Average top-1 and top-5 correct labels
    if prefix == "train_":
        metrics = jax.tree_map(jnp.mean, metrics)
    # Metrics will be pmeaned so don't divide here
    metrics[f"{prefix}loss"] = loss

    for k, v in kwargs.items():
        metrics[k] = jnp.array(v)

    return metrics


def collect_loss_and_metrics(
    params: hk.Params | None,
    logits: Array,
    inputs: Dict[str, Any],
    config: Any,
    _one_hot: Callable,
    is_training: bool,
) -> Tuple[Array, Dict[str, Array]]:

    y = _one_hot(inputs["labels"])

    if is_training:

        if "mix_labels" in inputs:  # Handle cutmix/mixup label mixing
            y = process_mix_labels(y, inputs, _one_hot)

        if config.label_smoothing > 0:  # get smoothy
            y = apply_label_smoothing(y, config)

        if config.bfloat16:  # Cast logits to float32
            logits = logits.astype(jnp.float32)

    which_loss = getattr(utils, config.which_loss)
    reduction = "mean" if is_training else None
    loss = which_loss(logits, y, reduction=reduction)

    l2_losses = {}
    if is_training and params is not None:
        l2_losses = get_l2_loss(params, config)

        decay_loss = l2_losses["decay_loss"]
        if config.decouple_wd:
            decay_loss = decay_loss * inputs["learning_rate"]
        loss = loss + decay_loss

    prefix = "train_" if is_training else "eval_"

    metrics = gather_metrics(logits, inputs, loss, prefix, **l2_losses)
    return loss, metrics


def _loss_fn(
    params: hk.Params,
    state: hk.State,
    inputs: dict[str, Any],
    rng: KeyArray,
    config: Any,
    _one_hot: Callable,
    net: hk.TransformedWithState,
):

    outputs, state = net.apply(
        params,
        state,
        rng,
        inputs,
        is_training=True,
    )

    loss, metrics = collect_loss_and_metrics(
        params=params,
        logits=outputs["logits"],
        inputs=inputs,
        config=config,
        _one_hot=_one_hot,
        is_training=True,
    )

    if "aux_losses" in outputs:
        metrics = collect_aux_losses_and_metrics(config, outputs, metrics)
        loss = loss + metrics["aux_loss"]

    scaled_loss = loss / jax.device_count()  # Grads get psummed so do divide
    return scaled_loss, (metrics, state, outputs)


def collect_aux_losses_and_metrics(
    config: Any, outputs: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    aux_losses = outputs["aux_losses"]

    mc = config.memory_config
    conf_loss = aux_losses["conf_loss"] if mc.get("use_conf_loss") else 0
    same_k_loss = aux_losses["same_k_loss"] if mc.get("use_same_k_loss") else 0
    same_b_loss = aux_losses["same_b_loss"] if mc.get("use_same_b_loss") else 0
    same_b_loss = jnp.maximum(0, same_b_loss - mc.get("MIN_B_LOSS", 0))
    aux_loss = mc.get("AUX_LOSS_WEIGHT", 1.0) * (conf_loss + same_k_loss + same_b_loss)

    metrics.update(
        {
            "aux_loss": jnp.array(aux_loss),
            "same_k_loss": jnp.array(same_k_loss),
            "same_b_loss": jnp.array(same_b_loss),
            "conf_loss": jnp.array(conf_loss),
        }
    )

    model_metrics = outputs.get("metrics", {})
    metrics.update(model_metrics)

    return metrics


def apply_proximity_loss(
    model_outputs: Dict[str, Any],
    params: hk.Params,
    config: Any,
) -> float:

    _prox_loss: float = config.memory_config.force_proximity_loss  # type: ignore

    if not config.local:
        model_outputs = jax.lax.pmean(model_outputs, "i")

    params, org_db_values = hk.data_structures.partition(config.predicate_fn, params)

    updated_db_values = model_outputs["final_db_values"]

    l2_prox = jtu.tree_map(
        lambda x, y: jnp.sum(jnp.square(x - y)), org_db_values, updated_db_values
    )

    l2_prox = sum(jtu.tree_leaves(l2_prox))
    l2_prox = l2_prox / hk.data_structures.tree_size(org_db_values)

    return _prox_loss * l2_prox


def get_better_memory_updates_loss(
    loss_list: List[float],
    b_multiplier: float,
) -> jax.Array:

    running_difference = jnp.array(0)
    for micro_step in range(len(loss_list) - 1):
        difference = loss_list[micro_step + 1] - loss_list[micro_step]
        running_difference += jnp.max(jnp.array([0.0, difference]))
    return b_multiplier * running_difference


def get_b_multiplier(inputs: Dict[str, Any], config: Any) -> Any:
    """
    it gets force_better_memory_updates_loss from memory_config,
        if its 0, or its not set,
            it checks if force_better_updates_schedule is set,
                if it is, it uses that to get the b_multiplier
                it also checks if the current step is less
                than better_wait_steps, if it is, it returns 0
            else it returns 0
        else it returns the value from memory_config
    """
    b_multiplier = config.memory_config.get("force_better_memory_updates_loss", 0)

    if b_multiplier == 0.0 and config.get("force_better_updates_schedule"):

        curr_step = inputs.get("global_step", 0)
        better_wait_steps = jnp.array(config.better_wait_steps)

        b_multiplier = jax.lax.select(
            jnp.greater(curr_step, better_wait_steps),
            config.force_better_updates_schedule(curr_step),
            jnp.array(0.0),
        )

    return b_multiplier


def get_sam_gradient(in_params, in_states, inputs, rng, rho, config, grad_fn):
    """Returns the gradient of the SAM loss loss, updated state and logits.

    See https://arxiv.org/abs/2010.01412 for more details.

    Args:
    model: The model that we are training.
    rho: Size of the perturbation.
    """
    print("Using SAM!")
    # compute gradient on the whole batch

    grads, outputs = grad_fn(in_params, in_states, inputs, rng)

    grads = utils.dual_vector(grads)
    noised_params = jax.tree_map(lambda a, b: a + rho * b, in_params, grads)

    grads, noised_outputs = grad_fn(noised_params, in_states, inputs, rng)

    states = outputs[1]
    metrics = noised_outputs[0]

    sam_outputs = [metrics, states]

    if config.use_c:
        sam_outputs.append(outputs[2])

    return grads, sam_outputs


def _collect_magnitudes(params_pytree, grads_pytree, updates_pytree, metrics, prefix):

    param_mag = memory_utils._calc_tree_magnitude(params_pytree)
    grad_mag = memory_utils._calc_tree_magnitude(grads_pytree)
    up_mag = memory_utils._calc_tree_magnitude(updates_pytree)

    metrics[f"mag/{prefix}param"] = param_mag
    metrics[f"mag/{prefix}grad"] = grad_mag
    metrics[f"mag/{prefix}up"] = up_mag

    return metrics


def _get_tree_magnitudes(pytree: dict[str, Any], config: Any) -> dict[str, Any]:

    memory_component_layers = config.memory_config.get("memory_component_layers", [])
    memory_layers = config.memory_config.get("memory_layers", [])
    mem_c = memory_component_layers + memory_layers

    component_names = [i + 1 for i in range(len(mem_c))]
    mem_c_labels = list(zip(mem_c, component_names))

    param_labels = utils.chain_label_fns(mem_c_labels=mem_c_labels, pytree=pytree)

    predicate_fn = lambda mod_name, var_name, _: param_labels[mod_name][var_name]

    trees = hk.data_structures.partition_n(
        predicate_fn, pytree, n=len(mem_c_labels) + 1
    )

    def get_tree_mag(tree):
        tree_mag = jtu.tree_map(memory_utils.calculate_magnitude, tree)
        # return tree_mag
        return jnp.mean(jnp.array(jtu.tree_leaves(tree_mag)))

    trees_mag = {
        "normal": get_tree_mag(trees[0]),
    }

    for label, i in mem_c_labels:
        trees_mag[label] = get_tree_mag(trees[i])

    return trees_mag


def _collect_all_magnitudes(params, grads, updates, metrics, config):

    params_trees_mag = _get_tree_magnitudes(params, config)
    grads_trees_mag = _get_tree_magnitudes(grads, config)
    updates_trees_mag = _get_tree_magnitudes(updates, config)

    metrics["mag/params"] = params_trees_mag
    metrics["mag/grads"] = grads_trees_mag
    metrics["mag/updates"] = updates_trees_mag

    return metrics


def collect_normal_magnitudes(params, grads, updates, metrics, config):
    get_normal_pytree = functools.partial(_get_normal_pytree, config=config)
    normal_params = get_normal_pytree(params)
    normal_grads = get_normal_pytree(grads)
    normal_updates = get_normal_pytree(updates)

    metrics = _collect_magnitudes(
        normal_params, normal_grads, normal_updates, metrics, prefix=""
    )

    if "memory_component_layers" in config.memory_config:

        # TODO: Collect mag from individual components using a for loop

        get_memory_components_pytree = functools.partial(
            _get_memory_components_pytree, config=config
        )

        mem_c_params = get_memory_components_pytree(params)
        mem_c_grads = get_memory_components_pytree(grads)
        mem_c_updates = get_memory_components_pytree(updates)

        metrics = _collect_magnitudes(
            mem_c_params, mem_c_grads, mem_c_updates, metrics, prefix="mem_c_"
        )

    return metrics


def collect_pos_emb_magnitude(params, metrics, config) -> Dict[str, Any]:

    if config.memory_config.get("use_pos_emb"):

        pos_emb_params = utils._get_sub_pytrees(
            params,
            include_strings=["pos_emb"],
            exclude_strings=[],
        )

        pos_emb_mag = memory_utils._calc_tree_magnitude(pos_emb_params)

        metrics["mag/mem_pos_emb"] = pos_emb_mag

    return metrics

def collect_db_values_magnitudes(updated_db_values, orginal_db_values, metrics, prefix):
    m_fn = lambda x: jtu.tree_map(memory_utils.calculate_magnitude, x)
    s_fn = lambda a, b: jtu.tree_map(lambda x, y: x - y, a, b)

    org_mag = m_fn(orginal_db_values)
    new_mag = m_fn(updated_db_values)

    u = s_fn(updated_db_values, orginal_db_values)
    u_mag = m_fn(u)

    diff_mag = s_fn(new_mag, org_mag)

    metrics[f"{prefix}/org"] = org_mag
    metrics[f"{prefix}/new"] = new_mag
    metrics[f"{prefix}/update"] = u_mag
    metrics[f"{prefix}/diff"] = diff_mag

    return metrics


def process_ema(
    params,
    states,
    global_step,
    ema_params,
    ema_states,
    config,
):
    ema_fn = getattr(utils, config.get("which_ema", "tf1_ema"))

    def ema(x, y):
        return ema_fn(x, y, config.ema_decay, global_step)

    ema_params = jax.tree_map(ema, ema_params, params)
    ema_states = jax.tree_map(ema, ema_states, states)
    return ema_params, ema_states


def get_freeze_c_updates(
    params,
    grads,
    opt_states,
    learning_rate,
    _optimizer,
    config,
):
    opt = _optimizer(learning_rate)
    tx = optax.multi_transform(
        {"trainable": opt, "frozen": optax.set_to_zero()}, config.label_fn
    )
    updates, opt_states = tx.update(grads, opt_states, params)
    return updates, opt_states


def get_fast_slow_updates(
    params, grads, opt_states, learning_rate, _optimizer, metrics, config
):
    lr_2 = learning_rate * config.lr_multiplier
    opt_1 = _optimizer(learning_rate)
    opt_2 = _optimizer(lr_2)
    tx = optax.multi_transform(
        {"type_1": opt_1, "type_2": opt_2}, config.label_fn_fast_slow
    )
    updates, opt_states = tx.update(grads, opt_states, params)
    metrics["lr_2"] = lr_2
    return updates, opt_states, metrics


def get_freeze_fast_slow_updates(
    params, grads, opt_states, learning_rate, _optimizer, metrics, config
):

    lr_2 = learning_rate * config.lr_multiplier
    opt_1 = _optimizer(learning_rate)
    opt_2 = _optimizer(lr_2)
    tx = optax.multi_transform(
        {
            "type_1": opt_1,
            "type_2": opt_2,
            "frozen": optax.set_to_zero(),
        },
        config.label_fn_freeze_fast_slow,
    )

    updates, opt_states = tx.update(grads, opt_states, params)
    metrics["lr_2"] = lr_2

    return updates, opt_states, metrics


def freeze_stage_2_except_cls_kernels(updates):

    stage_2_layers = [
        "ResBlock_7",
        "ResBlock_8",
        "ResBlock_9",
        "ResBlock_10",
        "ResBlock_11",
        "ResBlock_12",
    ]
    exceptions = ["bn", "ln"]

    def predicate_fn(mod_name: str, var_name: str, _) -> bool:
        include = any(s in mod_name or s in var_name for s in stage_2_layers)
        exclude = not any(ex in mod_name or ex in var_name for ex in exceptions)
        return include and exclude

    to_freeze_grads, normal_grads = hk.data_structures.partition(predicate_fn, updates)
    cls_num_kernels: int = normal_grads["AiM_ResNet"]["cls_token"].size // 196

    def _freeze(x: Array, k: int) -> Array:
        if x.shape[-2] > x.shape[-1]:
            # Downsample Conv
            return x.at[:, :, :, : -(k // 4)].set(0)

        elif x.shape[-2] < x.shape[-1]:
            # Upsample Conv
            return x.at[:, :, :, :-k].set(0)

        else:
            return x.at[:, :, :, : -(k // 4)].set(0)

    frozen_grads = jtu.tree_map(
        functools.partial(_freeze, k=cls_num_kernels), to_freeze_grads
    )

    updates = hk.data_structures.merge(frozen_grads, normal_grads)  # type: ignore

    return updates


def apply_consolidated_updates(
    params: hk.Params, model_outputs: dict[str, Any], config: Any
) -> Tuple[hk.Params, dict[str, Array]]:

    # TODO: Make this more efficient by using jax.ops.index_update

    new_db_values = model_outputs["new_db_values"]
    new_db_keys = model_outputs["new_db_keys"]
    metrics = model_outputs["metrics"]

    prox_loss, prox_metrics = memory_assets.consolidation_proximity_loss(
        old_db_values=params["AiM_ResNet"]["db_values"],
        old_db_keys=params["AiM_ResNet"]["db_keys"],
        new_db_values=new_db_values,
        new_db_keys=new_db_keys,
        total_memories_changed=metrics["total_memories_changed"],
        config=config,
    )

    metrics.update(prox_metrics)
    metrics["ploss/loss"] = prox_loss

    params["AiM_ResNet"]["db_values"] = new_db_values  # type: ignore
    params["AiM_ResNet"]["db_keys"] = new_db_keys  # type: ignore

    return params, metrics


def apply_db_updates(
    model_outputs: dict[str, Any],
    params: Union[hk.Params, Any],
    config: Any,
    DECAY: Optional[float] = None,  # type: ignore
    global_step: Optional[int] = None,
    metrics: dict[str, Any] = {},
    prefix: str = "train",
) -> Tuple[hk.Params, dict[str, Any], Array]:
    """
    Args:
        model_outputs (_type_)
        params (Union[hk.Params, Any]):
        config (_type_):
        DECAY (Optional[float], optional): Defaults to None.
        metrics (dict, optional): Defaults to {}.
        prefix (str, optional): Defaults to "train_db_ema".

    Returns:
        Tuple[Union[hk.Params, Any], hk.Params, dict]: params, updated_db_values, metrics
    """
    if not config.local:
        model_outputs = jax.lax.pmean(model_outputs, "i")

    updated_params, diff_metrics = apply_consolidated_updates(
        params, model_outputs, config
    )

    if DECAY is None:

        if config.memory_config.get("prox_update_db"):

            if (
                "train_db_momentum_schedule" in config.memory_config
                and global_step is not None
            ):
                DECAY: float = config.memory_config.db_update_schedule(global_step)

            else:
                DECAY = 0.0

        else:
            DECAY = 1.0

    metrics[f"{prefix}/decay"] = jnp.array(DECAY)
    metrics.update(diff_metrics)

    ema = lambda org, update: DECAY * org + (1.0 - DECAY) * update
    params = jtu.tree_map(ema, params, updated_params)

    return params, metrics, model_outputs["indices"]


def _train_fn(
    params,
    states,
    opt_states,
    inputs,
    rng,
    global_step,
    ema_params,
    ema_states,
    config,
    _get_learning_rate,
    _loss_fn,
    _optimizer,
):
    """Runs one batch forward + backward and run a single opt step."""

    learning_rate = _get_learning_rate(global_step)
    inputs["learning_rate"] = learning_rate
    inputs["global_step"] = global_step

    in_params = params
    if config.bfloat16:
        in_params, states = jax.tree_map(utils.to_bf16, (params, states))

    grad_fn = jax.grad(_loss_fn, argnums=0, has_aux=True)

    if config.sam_rho > 0:
        grad_fn = functools.partial(
            get_sam_gradient, config=config, rho=config.sam_rho, grad_fn=grad_fn
        )

    grads, outputs = grad_fn(in_params, states, inputs, rng)

    metrics, states = outputs[:2]
    model_outputs = outputs[2]

    if config.bfloat16:
        states, metrics, grads = jax.tree_map(utils.from_bf16, (states, metrics, grads))

    # Sum gradients and average losses for pmap
    if not config.local:
        grads = jax.lax.psum(grads, "i")
        metrics = jax.lax.pmean(metrics, "i")
    # Compute updates and update parameters

    metrics["learning_rate"] = learning_rate

    if config.freeze_layers and config.get("use_fast_slow_train"):
        print("Using Freeze Fast Slow updates")
        updates, opt_states, metrics = get_freeze_fast_slow_updates(
            params=params,
            grads=grads,
            opt_states=opt_states,
            learning_rate=learning_rate,
            _optimizer=_optimizer,
            metrics=metrics,
            config=config,
        )

    elif config.freeze_layers:
        print("Using Freeze updates")
        updates, opt_states = get_freeze_c_updates(
            params=params,
            grads=grads,
            opt_states=opt_states,
            learning_rate=learning_rate,
            _optimizer=_optimizer,
            config=config,
        )
    elif config.get("use_fast_slow_train"):
        print("Using Fast Slow updates")
        updates, opt_states, metrics = get_fast_slow_updates(
            params=params,
            grads=grads,
            opt_states=opt_states,
            learning_rate=learning_rate,
            _optimizer=_optimizer,
            metrics=metrics,
            config=config,
        )
    else:
        print("Normal updates")
        _, opt_apply = _optimizer(learning_rate)
        updates, opt_states = opt_apply(grads, opt_states, params)

    if config.memory_config.get("cls_freeze_stage_2"):
        print("Freezing stage 2 cls kernels")
        updates = freeze_stage_2_except_cls_kernels(updates)

    params = optax.apply_updates(params, updates)

    if ema_params is not None:
        ema_params, ema_states = process_ema(
            params,
            states,
            global_step,
            ema_params,
            ema_states,
            config,
        )

    if config.memory_config.get("use_sam_loss_fn"):

        params, metrics, indices = apply_db_updates(
            model_outputs=model_outputs["pre_model_outputs"],
            params=params,
            config=config,
            global_step=global_step,
            metrics=metrics,
            prefix="train",
        )

        # metrics = collect_memory_magnitudes(
        #     params=params,
        #     grads=grads,
        #     updates=updates,
        #     indices=indices,
        #     metrics=metrics,
        #     config=config,
        # )

    if config.return_magnitudes:
        metrics = _collect_all_magnitudes(
            params=params, grads=grads, updates=updates, metrics=metrics, config=config
        )

    outputs = {
        "params": params,
        "states": states,
        "opt_states": opt_states,
        "ema_params": ema_params,
        "ema_states": ema_states,
        "metrics": metrics,
    }

    # if config.local:
    #     outputs["model_outputs"] = model_outputs
    # outputs["grads"] = grads
    # outputs["updates"] = updates

    return outputs


def train_step(
    inputs, train_state, global_step, rng, train_fn, *unused_args, **unused_kwargs
):

    out = train_fn(
        params=train_state.params,
        states=train_state.state,
        opt_states=train_state.opt_state,
        inputs=inputs,
        rng=rng,
        global_step=global_step,
        ema_params=train_state.ema_params,
        ema_states=train_state.ema_state,
    )

    _params, _state = out["params"], out["states"]
    _opt_state = out["opt_states"]
    _ema_params, _ema_state = out["ema_params"], out["ema_states"]

    scalars = jl_utils.get_first(out["metrics"])
    train_state = EXPERIMENT_STATE(_params, _state, _opt_state, _ema_params, _ema_state)

    return scalars, train_state


def _build_eval_input(config):
    """Builds the evaluation input pipeline."""
    bs_per_device = config.eval_batch_size
    eval_datasets = []
    for which_dataset_config in config.dataset_configs.eval_configs.values():
        eval_datasets.append(
            [
                all_datasets.load(
                    bs_per_device=bs_per_device,
                    batch_dims=[jax.local_device_count(), bs_per_device],
                    **which_dataset_config,
                ),
                which_dataset_config.which_dataset,
            ]
        )
    return eval_datasets


def _eval_fn(params, state, inputs, config, net, _one_hot):
    """Evaluate a single batch and return loss and top-k acc."""

    outputs, state = net.apply(
        params,
        state,
        None,
        inputs,
        is_training=False,
    )

    _, metrics = collect_loss_and_metrics(
        params=None,
        logits=outputs["logits"],
        inputs=inputs,
        config=config,
        _one_hot=_one_hot,
        is_training=False,
    )

    return jax.lax.psum(metrics, "i")


def metrics_to_flat_metrics(metrics):
    flat_metrics = {}
    for ds_name, ds_metrics in metrics.items():
        for metric_name, metric_res in ds_metrics.items():
            flat_metrics[f"{ds_name}/{metric_name}"] = metric_res
    return flat_metrics


def _eval_epoch(params, state, eval_fn, _build_eval_input):
    """Evaluates an epoch."""
    eval_metrics = {}
    _eval_datasets = _build_eval_input()
    for ds, which_ds in _eval_datasets:
        num_samples = 0.0
        summed_metrics = None
        for inputs in ds:
            # Account for pmaps
            num_samples += np.prod(inputs["labels"].shape[:2])
            metrics = eval_fn(params, state, inputs)
            # Accumulate the sum of metrics for each step.
            metrics = jax.tree_map(lambda x: jnp.sum(x[0], axis=0), metrics)
            if summed_metrics is None:
                summed_metrics = metrics
            else:
                summed_metrics = jax.tree_map(jnp.add, summed_metrics, metrics)
        mean_metrics = jax.tree_map(lambda x: x / num_samples, summed_metrics)
        eval_metrics[which_ds] = jax.device_get(mean_metrics)
    flat_metrics = metrics_to_flat_metrics(eval_metrics)
    return flat_metrics


def _ceval_fn(params, state, inputs, config, net, _one_hot):
    """Evaluate a single batch and return loss and top-k acc."""
    # inputs = jl_utils.get_first(inputs)
    outputs, state = net.apply(
        params, state, None, inputs, is_training=False, return_metrics=False
    )
    logits = outputs["logits"]
    c_updates = outputs["c_updates"]
    y = _one_hot(inputs["labels"])
    which_loss = getattr(utils, config.which_loss)
    loss = which_loss(logits, y, reduction=None)
    metrics = utils.topk_correct(logits, inputs["labels"], prefix="eval_")
    metrics["eval_loss"] = loss
    return jax.lax.psum(metrics, "i"), jax.lax.pmean(c_updates, "i")
    # return metrics, c_updates

def evaluate(
    _params,
    _state,
    _ema_params,
    _ema_state,
    global_step,
    config,
    _eval_epoch,
    _c_eval_epoch=None,
    **unused_args,
):
    metrics = _eval_epoch(_params, _state)
    if _c_eval_epoch is not None:
        c_metrics = _c_eval_epoch(_params, _state)
        metrics.update({f"c_{key}": val for key, val in c_metrics.items()})
    return metrics


def keep_only_latest_checkpoint(bucket, save_path):
    all_checkpoints = utils.list_blobs(bucket)
    latest_checkpoint, checkpoints_to_delete = utils.clean_run_checkpoints(
        save_path, all_checkpoints, bucket
    )
    print("Kept Checkpoint -> ", latest_checkpoint)
    print("Deleted Checkpoints -> ", len(checkpoints_to_delete))


def build_checkpointer(save_path, checkpoint_dir, keep_latest_checkpoint_only=False):
    bucket = checkpoint_dir[5:]

    def save_checkpoint(step, train_state):

        to_float = lambda x: x.astype(jnp.float32)

        # TODO: remove to_float, is messing with opt_state, convert only
        # bfloat16 to float32

        state_dict = {
            "step": int(step),
            "params": train_state.params,
            "state": train_state.state,
            "ema_params": train_state.ema_params,
            "ema_state": train_state.ema_state,
            "opt_state": train_state.opt_state,
        }

        save_dir = os.path.join(save_path, f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        python_state_path = os.path.join(save_dir, "checkpoint.dill")
        with open(python_state_path, "wb") as f:
            dill.dump(state_dict, f)
        print(
            f"Saved step_{step} checkpoint to {os.path.join(bucket, python_state_path)}"
        )
        utils.upload_blob(
            bucket_name=bucket,
            source_file_name=python_state_path,
            destination_blob_name=python_state_path,
        )
        os.remove(python_state_path)

        if keep_latest_checkpoint_only:
            keep_only_latest_checkpoint(bucket, save_path)

    return save_checkpoint


def get_checkpoint_names(current_save_dir, checkpoint_bucket):
    print(f"Looking for checkpoints in {current_save_dir} in {checkpoint_bucket}")
    all_dirs = utils.list_blobs(checkpoint_bucket)
    current_checkpoints = [path for path in all_dirs if current_save_dir in path]
    return current_checkpoints


def load_checkpoint(python_state_path):
    with open(python_state_path, "rb") as f:
        pretrained_state = dill.load(f)
    print(f"Restored checkpoint from {python_state_path}")
    return pretrained_state


def get_latest_checkpoint(checkpoints, bucket):
    python_state_path = natsort.natsorted(checkpoints)[-1]
    idx = python_state_path.rfind("/")
    os.makedirs(python_state_path[:idx], exist_ok=True)
    print("Downloading checkpoint from ", os.path.join(bucket, python_state_path))
    utils.download_blob(bucket, python_state_path, python_state_path)
    return load_checkpoint(python_state_path)


def get_checkpoints(config, full_config):

    # First find the latest checkpoint in the current run directory.
    include_top = full_config.include_top

    current_run_dir = f"checkpoint_{config.name}"
    run_checkpoints = get_checkpoint_names(current_run_dir, config.bucket)
    if len(run_checkpoints):
        include_top = True
        return run_checkpoints, config.bucket, include_top

    # if there are no checkpoints in the current run directory, then look for
    # checkpoints in the restore path, in the restore path bucket.
    print("No checkpoints found in current run directory")

    if full_config.restore_path:
        restore_path_checkpoints = get_checkpoint_names(
            full_config.restore_path, full_config.restore_path_bucket
        )
        if len(restore_path_checkpoints):
            return (
                restore_path_checkpoints,
                full_config.restore_path_bucket,
                include_top,
            )
    return None, None, None


def opt_state_float_to_int(opt_state_float):
    to_int = lambda x: x.astype(jnp.int32)
    opt_state_float = opt_state_float._replace(
        mini_step=jtu.tree_map(to_int, opt_state_float.mini_step)
    )
    opt_state_float = opt_state_float._replace(
        gradient_step=jtu.tree_map(to_int, opt_state_float.gradient_step)
    )
    return opt_state_float


def MultiTransformState_float_to_int(opt_state_float):
    x_1 = opt_state_float_to_int(opt_state_float.inner_states["type_1"].inner_state)
    x_2 = opt_state_float_to_int(opt_state_float.inner_states["type_2"].inner_state)
    opt_state_float.inner_states["type_1"] = opt_state_float.inner_states[
        "type_1"
    ]._replace(inner_state=x_1)
    opt_state_float.inner_states["type_2"] = opt_state_float.inner_states[
        "type_2"
    ]._replace(inner_state=x_2)
    return opt_state_float


def extract_train_state_from_pretrained_state(pretrained_state):
    step = jl_utils.get_first(pretrained_state["step"])
    _params = pretrained_state["params"]
    _state = pretrained_state["state"]
    _ema_params = pretrained_state["ema_params"]
    _ema_state = pretrained_state["ema_state"]
    fields = getattr(pretrained_state["opt_state"], "_fields", None)
    if fields and "mini_step" in fields:
        rich.print("Converting opt_state floats to ints")
        rich.print("*" * 80)
        _opt_state = opt_state_float_to_int(
            opt_state_float=pretrained_state["opt_state"]
        )
    elif (
        isinstance(pretrained_state["opt_state"], optax.MultiTransformState)
        and hasattr(pretrained_state["opt_state"], "inner_states")
        and "type_1" in pretrained_state["opt_state"].inner_states
        and "type_2" in pretrained_state["opt_state"].inner_states
    ):
        rich.print("Converting opt_state floats to ints")
        rich.print("*" * 80)
        _opt_state = MultiTransformState_float_to_int(
            opt_state_float=pretrained_state["opt_state"]
        )
    else:
        _opt_state = pretrained_state["opt_state"]
    train_state = EXPERIMENT_STATE(_params, _state, _opt_state, _ema_params, _ema_state)
    return step, train_state


def overwrite_with_fn(train_state, pretrained_state, fn, kwargs):
    print(f"Overwriting train_state with {fn.__name__}")
    fn = functools.partial(fn, **kwargs)
    _params = fn(train_state.params, pretrained_state["params"])
    _state = fn(train_state.state, pretrained_state["state"])
    _ema_params = fn(train_state.ema_params, pretrained_state["ema_params"])
    _ema_state = fn(train_state.ema_state, pretrained_state["ema_state"])
    fields = getattr(train_state.opt_state, "_fields", None)
    if fields and "mini_step" in fields:
        to_int = lambda x: x.astype(jnp.int32)
        opt_state = train_state.opt_state
        opt_state = opt_state._replace(
            mini_step=jtu.tree_map(to_int, opt_state.mini_step)
        )
        opt_state = opt_state._replace(
            gradient_step=jtu.tree_map(to_int, opt_state.gradient_step)
        )
        _opt_state = opt_state
    else:
        _opt_state = train_state.opt_state
    train_state = EXPERIMENT_STATE(_params, _state, _opt_state, _ema_params, _ema_state)
    return train_state


def restore_state_from_checkpoint(init_rng, config, full_config, _initialize_train):

    checkpoints = None
    pretrained_state = None

    if config.local and full_config.restore_path:
        pretrained_state = load_checkpoint(full_config.restore_path)

    if not config.local:
        checkpoints, bucket, include_top = get_checkpoints(config, full_config)
        full_config.include_top = include_top
        if checkpoints is not None:
            pretrained_state = get_latest_checkpoint(checkpoints, bucket)

    if pretrained_state is None:
        print("Starting from scratch.")
        _train_input, train_state = _initialize_train(init_rng)
        return 0, train_state, _train_input

    if not full_config.train_checkpoint_all_hosts:
        pretrained_state = jl_utils.bcast_local_devices(pretrained_state)

    if full_config.include_top:
        _train_input = _initialize_train(init_rng)
        step, train_state = extract_train_state_from_pretrained_state(pretrained_state)
        return step, train_state, _train_input

    elif config.build_on_top or full_config.transfer_params:
        full_config.include_top = False
        _train_input, train_state = _initialize_train(init_rng)
        exceptions = config.get("build_exceptions", [])
        print(f"Build on top exceptions: {exceptions}")
        train_state = overwrite_with_fn(
            train_state,
            pretrained_state,
            utils.transfer_params,
            {"verbose": config.model_kwargs.verbose >= 4, "exceptions": exceptions},
        )
        return 0, train_state, _train_input

    else:
        print(f"{full_config.hack_include_top =}")
        _train_input, train_state = _initialize_train(init_rng)
        train_state = overwrite_with_fn(
            train_state, pretrained_state, utils.overwrite_leaves, {}
        )
        step = jl_utils.get_first(pretrained_state["step"])
        init_step = step if full_config.hack_include_top else 0
        print("Starting from step", init_step)
        return init_step, train_state, _train_input
