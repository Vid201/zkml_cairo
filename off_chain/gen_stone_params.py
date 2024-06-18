# https://github.com/anoma/juvix/blob/e2a6344f29a4b81204cc2a694e043113c16e9ad4/scripts/gen_stone_params.py
# thank you!

import json
import sys
import math

if len(sys.argv) != 2:
    sys.exit("Usage: gen_stone_params.py prog_public_input.json")

f = open(sys.argv[1])
data = json.load(f)
f.close()

n = int(math.log2(data["n_steps"])) - 6
if n < 0:
    sys.exit("Too few execution steps (at least 64 required)")

fri_step_list = [0, 4]
while n > 3:
    fri_step_list.append(3)
    n -= 3
if n > 0:
    fri_step_list.append(n)

out = {
    "field": "PrimeField0",
    "channel_hash": "poseidon3",
    "commitment_hash": "keccak256_masked160_lsb",
    "n_verifier_friendly_commitment_layers": 9999,
    "pow_hash": "keccak256",
    "statement": {
        "page_hash": "pedersen"
    },
    "stark": {
        "fri": {
            "fri_step_list": fri_step_list,
            "last_layer_degree_bound": 64,
            "n_queries": 18,
            "proof_of_work_bits": 24,
        },
        "log_n_cosets": 4,
    },
    "use_extension_field": False,
    "verifier_friendly_channel_updates": True,
    "verifier_friendly_commitment_hash": "poseidon3"
}

data_path = "cpu_air_params.json"
with open(data_path, "w") as f:
    f.write(json.dumps(out))