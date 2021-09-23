import functools
import multiprocessing

import optax
import ray

from mesh_transformer import util
from mesh_transformer.TPU_cluster import TPUCluster
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerV2
from mesh_transformer.util import clip_by_global_norm, additive_weight_decay
from ray_tpu import create_tpu, wait_til, get_connection, start_ray

import time
import jax
from jax.experimental import maps
import os
import requests 
from jax.config import config
colab_tpu_addr = os.environ['COLAB_TPU_ADDR'].split(':')[0]
url = f'http://{colab_tpu_addr}:8475/requestversion/tpu_driver0.1_dev20210607'
requests.post(url)

# The following is required to use TPU Driver as JAX's backend.
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']

def build_model(params, tpu_name, region, preemptible, version=1):
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    cores_per_replica = params["cores_per_replica"]
    tpu_size = jax.device_count()
    print("tpu_size", tpu_size, cores_per_replica, "cores_per_replica")

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]

    assert tpu_size in [8, 32, 128, 256, 512]

    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    # create_tpu(tpu_name, region, f"v3-{tpu_size}", preemptible)
    # assert wait_til(tpu_name, region, {'state': 'READY', 'health': 'HEALTHY'})

    # conns = get_connection(tpu_name, region)
    # print ("conns", conns)

    # assert len(conns) * 8 == tpu_size, "wrong size TPU for config"

    # head_info = ray.init(include_dashboard=False, object_store_memory=10**9)
    # address = head_info['redis_address']

    # with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
    #     p.map(functools.partial(start_ray, address=address, version=version), conns)

    len_conns = 8

    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1, use_psum=(version == 1)),
        optax.scale_by_adam(),
        additive_weight_decay(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr))
    )

    params["optimizer"] = opt

    # added by RG
    start = time.time()

    if tpu_size < cores_per_replica:
        msg = f"each shard needs a separate device, but device count ({tpu_size}) < shard count ({cores_per_replica})"
        raise ValueError(msg)
    print(f"jax devices: {tpu_size}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (tpu_size // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)
    # added by RG
    

    if version == 2:
        model_fn = functools.partial(CausalTransformerV2, params)
    elif version == 1:
        print("initializing network function") #added by RG
        # model_fn = functools.partial(CausalTransformer, params)
        model_fn = functools.partial(CausalTransformer, params)
    else:
        raise Exception(f"Version {version} does not exist")

    t = TPUCluster((tpu_size // cores_per_replica, cores_per_replica), len_conns, model_fn, version=version)
    return t
