from __future__ import annotations

import functools
import warnings
import time

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxline import utils as jl_utils

import wandb

warnings.simplefilter(action="ignore", category=FutureWarning)
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jmp
import rich
from absl import app
from etils import eapp

# TYPE CHECKING
from haiku import TransformedWithState
from jax.random import KeyArray
from optax import GradientTransformation

Array = jax.Array


from configs import memory_configs
from functions import utils
from functions import memory_assets, memory_utils
from trainers import trainer_wrappers
from configs import aim_train_config


@dataclass
class Args:
    run_name: str = ""
    train_vit: bool = False
    num_epochs: int = 90
    warmup_epochs: int = 5
    batch_size: int = 1024
    eval_batch_size: int = 50
    train_augment_name: str = ""
    augment_before_mix: bool = False
    lr: float = 0.1
    slow_lr_multiplier: float = 0.1
    weight_decay: float = 1e-4
    stable_memory_decay: float = 0.0  # Is overwritten by memory config.
    dropout: float = 0.0
    sam_rho: float = 0.0
    freeze_c_layers: bool = False
    exp_no: str = "61.6"
    config_file: str = "aim_train_config"
    use_old_aim: bool = False
    bucket: str = ""
    restore_path: str = ""
    include_top: bool = False
    hack_include_top: bool = False
    restore_path_bucket: str = ""
    transfer_params: bool = False
    log_every: int = 1000
    eval_every: int = 10000
    checkpoint_every: int = 1000
    save_all: bool = True
    vm_name: str = ""
    use_fake_data: bool = False
    local: bool = False
    verbose: int = 3
    return_magnitudes: bool = False
    bfloat16: bool = False
    force_no_overwrite: bool = False
    force_batch_size: int = 0
    real_data: bool = False
    keep_latest: bool = False
    decouple_wd: bool = False
    scale_lr_with_bs: bool = True


def turn_wandb_offline():
    import os

    os.environ["WANDB_API_KEY"] = "my_api_key"
    os.environ["WANDB_MODE"] = "offline"
    return



#  _             _
# | |_ _ __ __ _(_)_ __
# | __| '__/ _` | | '_ \
# | |_| | | (_| | | | | |
#  \__|_|  \__,_|_|_| |_|
#


def _sam_c_forward_fn(
    params: hk.Params,
    state: hk.State,
    rng: KeyArray,
    inputs: Dict[str, Any],
    is_training: bool,
    net: TransformedWithState,
    config: Any,
    _one_hot: Callable,
) -> Tuple[Dict[str, Any], hk.State]:

    pre_model_outputs, pre_state = net.apply(
        params,
        state,
        rng,
        inputs,
        is_training=is_training,
        consolidate_memory=True,
    )

    _, pre_metrics = trainer_wrappers.collect_loss_and_metrics(
        params=params,
        logits=pre_model_outputs["logits"],
        inputs=inputs,
        config=config,
        _one_hot=_one_hot,
        is_training=is_training,
    )

    updated_params, diff_metrics = trainer_wrappers.apply_consolidated_updates(
        params, pre_model_outputs, config
    )

    optimize_kwargs = {
        "s2_block_outputs": pre_model_outputs["s2_block_outputs"],
        "new_db_values": pre_model_outputs["new_db_values"],
        "new_db_keys": pre_model_outputs["new_db_keys"],
    }

    post_model_outputs, post_state = net.apply(
        updated_params,
        pre_state,
        rng,
        inputs,
        is_training=is_training,
        consolidate_memory=False,
        optimize_kwargs=optimize_kwargs,
    )

    _, post_metrics = trainer_wrappers.collect_loss_and_metrics(
        params=updated_params,
        logits=post_model_outputs["logits"],
        inputs=inputs,
        config=config,
        _one_hot=_one_hot,
        is_training=is_training,
    )

    aux_losses = post_model_outputs["aux_losses"]

    mc = config.memory_config
    conf_loss = aux_losses["conf_loss"] if mc.get("use_conf_loss") else 0
    same_k_loss = aux_losses["same_k_loss"] if mc.get("use_same_k_loss") else 0
    same_b_loss = aux_losses["same_b_loss"] if mc.get("use_same_b_loss") else 0
    same_b_loss = jnp.maximum(0, same_b_loss - mc.get("MIN_B_LOSS", 0))

    aux_loss = mc.get("AUX_LOSS_WEIGHT", 1.0) * (conf_loss + same_k_loss + same_b_loss)

    if is_training:
        post_metrics.update(
            {
                "aux_loss": jnp.array(aux_loss),
                "same_k_loss": jnp.array(same_k_loss),
                "same_b_loss": jnp.array(same_b_loss),
                "conf_loss": jnp.array(conf_loss),
            }
        )

    sam_metrics = {
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "diff_metrics": diff_metrics,
    }

    outputs = dict(
        pre_model_outputs=pre_model_outputs,
        post_model_outputs=post_model_outputs,
        sam_metrics=sam_metrics,
    )

    return outputs, pre_state


def _sam_loss_fn(
    params: hk.Params,
    state: hk.State,
    inputs: Dict[str, Any],
    rng: KeyArray,
    sam_net: Callable,
):

    outputs, state = sam_net(
        params,
        state,
        rng,
        inputs,
        is_training=True,
    )

    metrics = outputs["sam_metrics"]
    loss = metrics["post_metrics"]["train_loss"]
    aux_loss = metrics["post_metrics"]["aux_loss"]
    prox_loss = metrics["diff_metrics"]["ploss/loss"]
    loss = loss + prox_loss + aux_loss

    scaled_loss = loss / jax.device_count()  # Grads get psummed so do divide
    return scaled_loss, (metrics, state, outputs)


def _meta_loss_fn(
    params: hk.Params,
    state: hk.State,
    inputs: dict[str, Any],
    rng: KeyArray,
    config: Any,
    _one_hot: Callable,
    meta_net: hk.TransformedWithState,
):

    outputs, state = meta_net.apply(
        params,
        state,
        rng,
        inputs,
        is_training=True,
    )

    vmapped_fn = jax.vmap(
        trainer_wrappers.collect_loss_and_metrics,
        in_axes=(None, 0, 0, None, None, None),
    )

    inputs_all = jax.lax.all_gather(inputs, axis_name="i")
    current_device = jax.lax.axis_index("i")
    neighbours = config.communication_instr_set[current_device]
    input_neighbours = jtu.tree_map(lambda x: x[neighbours], inputs_all)
    print("input_neighbours", utils.tree_shape(input_neighbours))

    all_losses, all_metrics = vmapped_fn(
        params,
        outputs["logits"],
        input_neighbours,
        config,
        _one_hot,
        True,
    )

    n = config.memory_config.meta_batching.get("num_meta_batches", 8)
    first_metrics = jtu.tree_map(lambda x: x[current_device % n], all_metrics)
    print("all_losses", all_losses.shape)
    print("all_metrics", utils.tree_shape(all_metrics))

    # jax.debug.print("all_metrics = {all_metrics}", all_metrics=all_metrics)
    # jax.debug.print("first_metrics = {first_metrics}", first_metrics=first_metrics)

    fn = jnp.sum if config.memory_config.get("sum_meta_losses") else jnp.mean

    loss = fn(all_losses)

    metrics = jtu.tree_map(jnp.mean, all_metrics)
    metrics["first_metrics"] = first_metrics

    print("first_metrics", utils.tree_shape(metrics["first_metrics"]))

    if "aux_losses" in outputs:

        outputs["aux_losses"] = jtu.tree_map(fn, outputs["aux_losses"])

        metrics = trainer_wrappers.collect_aux_losses_and_metrics(
            config, outputs, metrics
        )

    loss = loss + metrics.get("aux_loss", 0)

    scaled_loss = loss / jax.device_count()  # Grads get psummed so do divide
    return scaled_loss, (metrics, state, outputs)

    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #


def _sam_eval_fn(
    params: hk.Params,
    state: hk.State,
    inputs: Dict,
    sam_net: Callable,
):
    """Evaluate a single batch and return loss and top-k acc."""

    outputs, state = sam_net(
        params,
        state,
        None,
        inputs,
        is_training=False,
    )

    metrics = outputs["sam_metrics"]["post_metrics"]

    return jax.lax.psum(metrics, "i")

    #  _             _          _
    # | |_ _ __ __ _(_)_ __    | |    ___    ___   _____
    # | __| '__/ _` | | '_ \   | |   / _ \  / _ \ |   _ \
    # | |_| | | (_| | | | | |  | \_ | |_| || |_| || ||_| |
    #  \__|_|  \__,_|_|_| |_|  \___| \___/  \___/ | |___/
    #                                             |_|


def build_default_train_functions(config: Any, full_config: Any):
    _build_train_input = functools.partial(trainer_wrappers._build_train_input, config)

    _forward_fn = functools.partial(trainer_wrappers._forward_fn, config=config)
    net = hk.transform_with_state(_forward_fn)

    build_get_learning_rate_fn = functools.partial(
        trainer_wrappers.build_schedule_fn, config.optimizer.lr_schedule
    )
    _get_learning_rate = build_get_learning_rate_fn()

    _optimizer = functools.partial(
        trainer_wrappers._optimizer, optimizer_config=config.optimizer
    )

    _initialize_train = functools.partial(
        trainer_wrappers._initialize_train,
        config=config,
        full_config=full_config,
        _build_train_input=_build_train_input,
        net=net,
        _optimizer=_optimizer,
    )

    _one_hot = functools.partial(trainer_wrappers._one_hot, config=config)

    if config.memory_config.get("use_sam_loss_fn"):
        print("Using sam loss fn")

        sam_net = functools.partial(
            _sam_c_forward_fn, net=net, config=config, _one_hot=_one_hot
        )

        _loss_fn = functools.partial(
            _sam_loss_fn,
            sam_net=sam_net,
        )

    elif config.memory_config.get("use_meta_loss_fn"):
        print("Using meta loss fn")
        _loss_fn = functools.partial(
            _meta_loss_fn,
            config=config,
            meta_net=net,
            _one_hot=_one_hot,
        )

    else:
        _loss_fn = functools.partial(
            trainer_wrappers._loss_fn,
            config=config,
            net=net,
            _one_hot=_one_hot,
        )

    loss_fn = _loss_fn

    _train_fn = functools.partial(
        trainer_wrappers._train_fn,
        config=config,
        _get_learning_rate=_get_learning_rate,
        _loss_fn=loss_fn,
        _optimizer=_optimizer,
    )

    donate_argnums = (0, 1, 2, 6, 7) if config.use_ema else (0, 1, 2)
    train_fn = jax.pmap(_train_fn, axis_name="i", donate_argnums=donate_argnums)

    train_step = functools.partial(
        trainer_wrappers.train_step,
        train_fn=train_fn,
    )

    return (
        train_step,
        net,
        loss_fn,
        _one_hot,
        _initialize_train,
        _optimizer,
        _train_fn,
    )


def build_eval_functions(
    config: Any,
    net: TransformedWithState,
    _one_hot: Callable,
):
    """
    Defined functions are:
        1. net -> from build_train_functions
        2. _one_hot -> from build_train_functions

    """
    _build_eval_input = functools.partial(
        trainer_wrappers._build_eval_input, config=config
    )

    _eval_fn = functools.partial(
        trainer_wrappers._eval_fn, config=config, net=net, _one_hot=_one_hot
    )
    eval_fn = jax.pmap(_eval_fn, axis_name="i")

    _eval_epoch = functools.partial(
        trainer_wrappers._eval_epoch,
        eval_fn=eval_fn,
        _build_eval_input=_build_eval_input,
    )

    _c_eval_epoch = None

    if config.memory_config.get("use_sam_eval_fn"):
        print("Using sam eval")

        sam_net = functools.partial(
            _sam_c_forward_fn, net=net, config=config, _one_hot=_one_hot
        )

        _c_eval_fn = functools.partial(_sam_eval_fn, sam_net=sam_net)
        c_eval_fn = jax.pmap(_c_eval_fn, axis_name="i")

        _c_eval_epoch = functools.partial(
            trainer_wrappers._eval_epoch,
            eval_fn=c_eval_fn,
            _build_eval_input=_build_eval_input,
        )

    evaluate = functools.partial(
        trainer_wrappers.evaluate,
        config=config,
        _eval_epoch=_eval_epoch,
        _c_eval_epoch=_c_eval_epoch,
    )

    return evaluate


def pre_modify_args(args: Args):

    if args.force_no_overwrite:
        return args

    model_kwargs, _, _ = memory_configs.get_memory_config(args.exp_no)
    memory_config = model_kwargs["memory_config"]
    version_kwargs = memory_config.get("version_kwargs", {})

    args.checkpoint_every = version_kwargs.get(
        "checkpoint_every", args.checkpoint_every
    )
    args.num_epochs = version_kwargs.get("num_epochs", args.num_epochs)
    args.warmup_epochs = version_kwargs.get("warmup_epochs", args.warmup_epochs)

    args.lr = version_kwargs.get("lr", args.lr)
    args.stable_memory_decay = version_kwargs.get(
        "stable_memory_decay", args.stable_memory_decay
    )

    args.weight_decay = version_kwargs.get("weight_decay", args.weight_decay)

    if not args.local:
        args.batch_size = version_kwargs.get("batch_size", args.batch_size)

        if args.force_batch_size:
            args.batch_size = args.force_batch_size

        args.eval_batch_size = version_kwargs.get(
            "eval_batch_size", args.eval_batch_size
        )

    args.dropout = version_kwargs.get("dropout", args.dropout)
    args.bfloat16 = version_kwargs.get("bfloat16", args.bfloat16)

    args.train_augment_name = version_kwargs.get(
        "train_augment_name", args.train_augment_name
    )
    args.augment_before_mix = version_kwargs.get(
        "augment_before_mix", args.augment_before_mix
    )

    args.return_magnitudes = version_kwargs.get(
        "return_magnitudes", args.return_magnitudes
    )
    args.keep_latest = version_kwargs.get("keep_latest", args.keep_latest)
    args.restore_path = version_kwargs.get("restore_path", args.restore_path)
    args.include_top = version_kwargs.get("include_top", args.include_top)
    args.hack_include_top = version_kwargs.get(
        "hack_include_top", args.hack_include_top
    )
    args.scale_lr_with_bs = version_kwargs.get(
        "scale_lr_with_bs", args.scale_lr_with_bs
    )

    if args.local and args.restore_path:
        args.restore_path = "misc/checkpoint_AiM_ResNet-62.0-SimpleDB_noE_0M_LAug_bloat-_step_375340_checkpoint.dill"

    args.verbose = model_kwargs.get("verbose", args.verbose)
    args.train_vit = memory_config.get("train_vit", args.train_vit)

    rich.inspect(
        args,
        value=False,
        title="args",
        docs=False,
    )

    return args


def post_modify_config(full_config: Any, config: Any, args: Args):

    # config.num_databases = config.memory_config.router_kwargs.num_databases

    config.num_repeat_c_steps = config.memory_config.get("num_repeat_c_steps", None)
    if config.num_repeat_c_steps is not None:
        config.micro_batch_size = config.train_batch_size // config.num_repeat_c_steps

    config.train_db_ema = config.memory_config.get(
        "train_db_ema"
    )  # DEPRECATED, use_schedule

    if "train_db_momentum_schedule" in config.memory_config:
        _schedule: dict = config.memory_config["train_db_momentum_schedule"]
        _schedule["kwargs"]["decay_steps"] = full_config.training_steps
        _schedule["kwargs"]["warmup_steps"] = full_config.warmup_steps

        config.memory_config.db_update_schedule = trainer_wrappers.build_schedule_fn(
            _schedule
        )

    if "force_better_updates_schedule" in config.memory_config:
        _schedule: dict = config.memory_config["force_better_updates_schedule"]
        _schedule["kwargs"]["decay_steps"] = full_config.training_steps
        _schedule["kwargs"]["warmup_steps"] = (
            full_config.training_steps - full_config.warmup_steps * 2
        )

        config.force_better_updates_schedule = trainer_wrappers.build_schedule_fn(
            _schedule
        )

        config.better_wait_steps = config.memory_config.get("better_wait_steps", 0)

        # # plot the schedule for debugging
        # x = np.arange(0, full_config.training_steps, 100000)
        # y = [config.force_better_updates_schedule(i) for i in x]
        # from matplotlib import pyplot as plt

        # plt.plot(x, y)
        # plt.show()
        # exit()

    config.use_c_loss_fn = config.memory_config.get("use_c_loss_fn", False)

    config.freeze_layers = config.memory_config.get("freeze_layers", False)
    config.layers_to_change = config.memory_config.get("layers_to_change", [])
    config.layers_exceptions = config.memory_config.get("layers_exceptions", [])
    config.build_exceptions = config.memory_config.get("build_exceptions", [])

    config.bloat_control = config.memory_config.bloat_control

    if config.freeze_layers:
        config.label_fn = functools.partial(
            utils.freeze_params_label_fn,
            layers_to_freeze=config.layers_to_change,
            exceptions=config.layers_exceptions,
        )

    if config.memory_config.get("memory_lr_multiplier"):
        config.lr_multiplier = config.memory_config.memory_lr_multiplier
        assert config.lr_multiplier != 1, "memory_lr == lr"

        config.use_fast_slow_train = True
        config.label_fn_fast_slow = functools.partial(
            utils.params_label_fn,
            layers_to_change=config.memory_config.fast_lr_layers,
            exceptions=[],
            labels=["type_1", "type_2"],
        )

    if config.freeze_layers and config.get("use_fast_slow_train"):
        config.label_fn_freeze_fast_slow = (
            trainer_wrappers.get_freeze_fast_slow_label_fn(config)
        )

    db_layer_name: str = config.memory_config.db_layer_name
    config.predicate_fn = lambda _, v_name, __: db_layer_name not in v_name

    config.optimizer["memory_layers"] = config.memory_config.get("memory_layers", [])

    aim_train_config.optim_asserts(config.optimizer)

    full_config.vm_name = args.vm_name

    if config.memory_config.get("use_meta_loss_fn"):
        from models import aim_resnet

        n = config.memory_config.meta_batching.get("num_meta_batches", 8)
        config.communication_instr_set = aim_resnet.get_instr_set(n)

    if config.bfloat16:
        import importlib

        mp_policy = jmp.get_policy("p=f32,c=bf16,o=bf16")
        bn_policy = jmp.get_policy("p=f32,c=f32,o=bf16")

        hk.mixed_precision.set_policy(hk.BatchNorm, bn_policy)
        hk.mixed_precision.set_policy(hk.LayerNorm, bn_policy)
        rich.print(f"Set hk.BatchNorm policy to {bn_policy}")

        model_module = importlib.import_module(f"models.{config.model_file}")
        model_class = getattr(model_module, config.model_class)
        hk.mixed_precision.set_policy(model_class, mp_policy)
        rich.print(f"Set {config.model_class} policy to {mp_policy}")

    rich.inspect(
        config.model_kwargs["memory_config"], value=False, title="memory_config"
    )

    # FOR WANDB SWEEPS ADDING VERSION KWARGS TO FULL CONFIG
    if not args.local:
        version_kwargs = config.memory_config.get("version_kwargs", {})
        for k, v in version_kwargs.items():
            if k not in full_config:
                full_config[k] = v
    return full_config, config


def post_process_metrics(scalars: dict[str, Any]) -> Tuple[dict[str, Any], Any]:

    scalars = jtu.tree_map(lambda x: x.tolist(), scalars)

    hists = {}
    scalars, avg_conf, counter = process_diff_metrics(scalars)

    hists = {
        # "diff_avg_conf": diff_avg_conf,
        # "diff_counter": diff_counter,
        "avg_conf": avg_conf,
        "counter": counter,
    }

    if "diff_metrics" in scalars:
        (
            scalars["diff_metrics"],
            diff_avg_conf,
            diff_counter,
        ) = process_diff_metrics(scalars["diff_metrics"])

    return scalars, hists


def process_diff_metrics(scalars: dict[str, list[float]]):
    avg_conf = scalars.pop("avg_conf", [])
    counter = scalars.pop("counter", [])

    # num_bins = len(avg_conf) if len(avg_conf) < 512 else 512

    # conf_hist = np.histogram(avg_conf, bins=num_bins)[0]
    # counter_hist = np.histogram(counter, bins=num_bins, density=True)[0]

    return scalars, avg_conf, counter


def test_loop(args: Args):
    # Parse args -
    turn_wandb_offline()
    args.batch_size = 2
    _args = pre_modify_args(args)
    full_config, config = trainer_wrappers.build_config_with_args(_args)
    full_config, config = post_modify_config(full_config, config, _args)
    if not config.get("train_vit", False):
        config = utils.update_db_kwargs(config)
    full_config.restore_path = ""

    # Train functions -
    (
        train_step,
        net,
        loss_fn,
        _one_hot,
        _initialize_train,
        _optimizer,
        _train_fn,
    ) = build_default_train_functions(config, full_config)

    wandb.init(
        project="Test",
        entity="joylunkad",
        name=config.name,
        config=full_config.to_dict(),
        reinit=True,
        id=config.name,
        mode="offline",
        resume="allow",
    )

    restore_state_from_checkpoint = functools.partial(
        trainer_wrappers.restore_state_from_checkpoint,
        config=config,
        full_config=full_config,
        _initialize_train=_initialize_train,
    )

    rng_key: KeyArray = jax.random.PRNGKey(41)
    rng_key_seq = hk.PRNGSequence(rng_key)

    init_rng = next(rng_key_seq)

    init_step, train_state, _train_input = restore_state_from_checkpoint(init_rng)

    # Write Unit Test for foward pass using net

    inputs = next(_train_input)
    rng = jl_utils.bcast_local_devices(next(rng_key_seq))
    scaled_loss, (metrics, state, outputs) = jax.pmap(loss_fn, axis_name="i")(
        train_state.params,
        train_state.state,
        inputs,
        rng,
    )
    rich.print(scaled_loss)
    rich.print(utils.tree_shape(metrics))
    rich.print(jtu.tree_map(jnp.mean, metrics))

    inputs = next(_train_input)

    scalars, train_state = train_step(
        inputs,
        train_state,
        global_step=jl_utils.bcast_local_devices(40000),
        rng=jl_utils.bcast_local_devices(next(rng_key_seq)),
    )
    rich.print("Scalars: ", scalars)

    if "updates" in outputs:
        rich.print(jtu.tree_map(memory_utils.calculate_magnitude, outputs["updates"]))



def main(args: Args):

    if args.local:
        test_loop(args)
        exit()
    # Parse args -

    _args = pre_modify_args(args)
    full_config, config = trainer_wrappers.build_config_with_args(_args)
    full_config, config = post_modify_config(full_config, config, _args)

    # Train functions -
    (
        train_step,
        net,
        loss_fn,
        _one_hot,
        _initialize_train,
        _optimizer,
        _train_fn,
    ) = build_default_train_functions(config, full_config)

    # Eval functions -
    evaluate = build_eval_functions(config, net, _one_hot)

    # Checkpointing Functions -
    build_checkpointer = functools.partial(
        trainer_wrappers.build_checkpointer,
        checkpoint_dir=full_config.checkpoint_dir,
    )

    restore_state_from_checkpoint = functools.partial(
        trainer_wrappers.restore_state_from_checkpoint,
        config=config,
        full_config=full_config,
        _initialize_train=_initialize_train,
    )

    config.name = f"{config.model_class}-{args.exp_no}-{config.exp_type}"
    config.name += f"-{args.run_name}" if len(args.run_name) else ""

    wandb.login(key="my_api_key")
    wandb.init(
        project="Consolidate",  # If changing, change in wb_utils.py as well.
        entity="joylunkad",
        name=config.name,
        config=full_config,
        id=config.name,
        resume="allow",
    )

    rng_key: KeyArray = jax.random.PRNGKey(full_config.random_seed)
    rng_key_seq = hk.PRNGSequence(rng_key)

    init_rng = next(rng_key_seq)

    init_step, train_state, _train_input = restore_state_from_checkpoint(init_rng)

    previous_best_model_eval_metric = 0.0
    save_model_fn = build_checkpointer(
        save_path=f"checkpoint_{config.name}",
        keep_latest_checkpoint_only=full_config.get(
            "keep_latest_checkpoint_only", False
        ),
    )
    num_samples = 0.0
    summed_scalars = None

    start = time.perf_counter()

    for step in range(init_step, full_config.training_steps):
        inputs = next(_train_input)
        num_samples += 1
        scalars, train_state = train_step(
            inputs,
            train_state,
            global_step=jl_utils.bcast_local_devices(step),
            rng=jl_utils.bcast_local_devices(next(rng_key_seq)),
        )

        # scalars, _ = train_step(
        #     inputs,
        #     train_state,
        #     global_step=jl_utils.bcast_local_devices(step),
        #     rng=jl_utils.bcast_local_devices(next(rng_key_seq)),
        # )

        if summed_scalars is None:
            summed_scalars = scalars
        else:
            summed_scalars = jax.tree_map(jnp.add, summed_scalars, scalars)

        if (
            (step % config.log_every == 0)
            or step == full_config.training_steps - 1
            or step == 10
            or step == 100
        ):
            scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
            scalars = jax.device_get(scalars)
            scalars, hists = post_process_metrics(scalars)
            time_per_step = (time.perf_counter() - start) / (num_samples)
            scalars["step"] = step
            scalars["time_per_step"] = time_per_step
            wandb.log(scalars, step)
            if step % config.eval_every == 0 or step == 10:
                scalars.update(hists)
            rich.print(f"[Step {step}] Train scalars: {scalars}")

            num_samples = 0.0
            start = time.perf_counter()
            summed_scalars = None

        if (
            (step % config.eval_every == 0 and step)
            or step == full_config.training_steps - 1
            or step == 1000
        ):
            eval_scalars = evaluate(
                train_state.params,
                train_state.state,
                train_state.ema_params,
                train_state.ema_state,
                jl_utils.bcast_local_devices(step),
            )
            eval_scalars, hists = post_process_metrics(eval_scalars)
            wandb.log(eval_scalars, step)
            # eval_scalars.update(hists)
            rich.print(f"[Step {step}] Eval scalars: {eval_scalars}")

            start = time.perf_counter()

        if (
            step % config.checkpoint_every == 0 and step
        ) or step == full_config.training_steps - 1:
            if config.save_all:
                if step == init_step:
                    continue
                save_model_fn(step, jl_utils.get_first(train_state))
            else:
                best_model_eval_metric = eval_scalars[  # type: ignore
                    full_config.best_model_eval_metric
                ]
                print("Prev: ", previous_best_model_eval_metric)
                print("Current: ", best_model_eval_metric)

                if previous_best_model_eval_metric <= best_model_eval_metric:
                    if step == init_step:
                        continue
                    previous_best_model_eval_metric = best_model_eval_metric
                    if full_config.train_checkpoint_all_hosts:
                        save_model_fn(step, train_state)
                    else:
                        save_model_fn(step, jl_utils.get_first(train_state))

            start = time.perf_counter()

    return full_config, config


def main_wrapper(args: Args):

    full_config, config = main(args)

    try:
        print("Inside try")
        from functions.tpu_utils import delete_tpu

        print("Deleting TPU: ", full_config.vm_name)
        print("Deleting TPU: ", config.zone)

        if len(full_config.vm_name):
            delete_tpu(name=full_config.vm_name, zone=config.zone)

    except:
        pass

    return full_config, config


if __name__ == "__main__":

    full_config, config = app.run(
        main_wrapper, flags_parser=eapp.make_flags_parser(Args)  # type: ignore
    )
