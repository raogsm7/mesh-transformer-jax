import argparse
import json
import threading
import time
from queue import Queue, Empty

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm

from flask import Flask, request, make_response, jsonify
app = Flask(__name__)

requests_queue = Queue()

"""
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"context":"eleutherai", "top_p": 0.9, "temp": 0.75}' \
  http://localhost:5000/complete
"""


def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/complete', methods=['POST', 'OPTIONS'])
# <PLACTEV> added generation length
def complete():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        content = request.json

        if requests_queue.qsize() > 100:
            return {"error": "queue full, try again later"}

        response_queue = Queue()

        requests_queue.put(({
                                "context": content["context"],
                                "gen_len": int(content["gen_len"]),
                                "top_p": float(content["top_p"]),
                                "temp": float(content["temp"])
                            }, response_queue))

        return _corsify_actual_response(jsonify({"completion": response_queue.get()}))
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))


#<PLACTEV> placeholder for stopping the API
@app.route("/stop", methods=['GET'])
def set_stop_run():
    return "Stopped"

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={"port": 5000, "host": "0.0.0.0"}).start()


    args = parse_args()
    params = json.load(open(args.config))

    '''<PLACTEV> make sure config file has a value of '1' '''
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]       
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    params["sampler"] = nucleaus_sample

    '''<PLACTEV> Comment below nine lines to work with canonical slim checkpoint as it does not have a meta file to read from'''
    # opt = optax.chain(
    #     optax.scale(1 / gradient_accumulation_steps),
    #     clip_by_global_norm(1),
    #     optax.scale_by_adam(),
    #     optax.additive_weight_decay(0),
    #     optax.scale(-1),
    #     optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    #     )
    
    # params["optimizer"] = opt

    params["optimizer"] = optax.scale(0) #<PLACTEV> set optimizer to zeros


    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    '''<PLACTEV> Comment below four lines to work with canonical slim checkpoint as it does not have a meta file to read from'''
    # with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
    #     meta = json.load(f)
    # ckpt_step = meta["checkpoints"][-1]
    # print(f"using checkpoint {ckpt_step}")

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica * 1

    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        '''<PLACTEV> Comment below line to work with canonical slim checkpoint as it does not have a meta file to read from'''
        # network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/", devices.shape[1]) #<PLACTEV> make sure to point to slim ckpt

        print(f"network loaded in {time.time() - start:.06}s")

        '''<PLACTEV> Comment below three lines to work with canonical slim checkpoint as it does not have a meta file to read from'''
        local_shards = max(jax.local_device_count() // mesh_shape[1], 1) #<PLACTEV>
        # del network.state["opt_state"] #<PLACTEV>
        network.state = network.move_xmap(network.state, np.zeros(local_shards)) #<PLACTEV>
        # network.state = network.move_xmap(network.state, np.zeros(cores_per_replica)) #<PLACTEV>

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

        while True:
            all_ctx = []
            all_top_p = []
            all_temp = []
            # all_gen_len = [] #<PLACTEV> add generation length
            all_q = []

            k = False #<PLACTEV> logic for sleeping message
            while len(all_ctx) < total_batch:
                try:
                    o, q = requests_queue.get(block=False)
                    all_ctx.append(o["context"])
                    all_top_p.append(o["top_p"])
                    all_temp.append(o["temp"])
                    all_gen_len = o['gen_len'] #<PLACTEV> add generation length
                    all_q.append(q)
                except Empty:
                    if len(all_ctx):
                        print('\nprocessing... ᕕ( ᐛ )ᕗ\n') #<PLACTEV> logic for processing message
                        break
                    else:
                        time.sleep(0.01)
                        if not k: print('\nsleeping (ー。ー) zzz') #<PLACTEV> logic for sleeping message
                        k = True #<PLACTEV> logic for sleeping message

            start = time.time()
            while len(all_ctx) < total_batch:
                all_ctx.append("whatever")
                all_top_p.append(1)
                all_temp.append(1)

            all_tokenized = []
            all_length = []
            for ctx in all_ctx:
                padded_tokens = np.zeros(seq).astype(np.uint32)
                length = 0

                try:
                    if all_length == []: print('\nPROMPT:\n', ctx) #<PLACTEV> logic for printing received prompt from the curl call
                    tokens = tokenizer.encode(ctx)
                    provided_ctx = len(tokens)
                    pad_amount = seq - provided_ctx

                    pad_amount = max(pad_amount, 0)

                    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)[-seq:]
                    length = len(tokens)
                except:
                    print("oops exception")

                all_tokenized.append(padded_tokens)
                all_length.append(length)

            output = network.generate(np.array(all_tokenized),
                                      np.array(all_length),
                                      all_gen_len, #<PLACTEV> add generation length
                                      {
                                          "top_p": np.array(all_top_p),
                                          "temp": np.array(all_temp)
                                      })
            

            '''<PLACTEV> this block of code works for out_axes=['batch', 'shard', ...] and with returning only 'return outputs' in generate_sample function.'''
            # ans = []
            # for o, q in zip(output[0][0],all_q):
            #     print('shape of o out of output of generate ->', o.shape)
            #     for i in o:
            #         ans.append(i[0])
            #     print('answer ->', ans)
            #     q.put(tokenizer.decode(ans))

            '''Neede to run jax 0.2.22 version as jax 0.12 is no longer available. It runs on jax 0.2.22 and jaxlib==0.1.69
            <PLACTEV> this block of code works with the canonical generate_xmap out_axes = ['batch', ...] and with returning 
            only 'return outputs' in generate_sample function.'''
            for o, q in zip(output, all_q):
                o = np.array(o[0,:])[:,0]
                q.put(tokenizer.decode(o))
            
            # for o, q in zip(output[1][0][:, :, 0], all_q): # <PLACTEV> this only works with jax 0.2.12
                # q.put(tokenizer.decode(o)) # <PLACTEV> this only works with jax 0.2.12

            print(f"completion done in {time.time() - start:06}s")