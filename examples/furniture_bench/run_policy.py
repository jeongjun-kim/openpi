import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import furniture_bench_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

config = _config.get_config("pi0_base")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base")

print(checkpoint_dir)
# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = furniture_bench_policy.make_furniture_bench_example()
print(policy._output_transform)
result = policy.infer(example)
print(result)
# Delete the policy to free up memory.
del policy
print("Actions shape:", result["actions"].shape)