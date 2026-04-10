
# Running openpi models remotely

We provide utilities for running openpi models remotely. This is useful for running inference on more powerful GPUs off-robot, and also helps keep the robot and policy environments separate (and e.g. avoid dependency hell with robot software).

## Starting a remote policy server

To start a remote policy server, you can simply run the following command:

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

The `env` argument specifies which $\pi_0$ checkpoint should be loaded. Under the hood, this script will execute a command like the following, which you can use to start a policy server, e.g. for checkpoints you trained yourself (here an example for the DROID environment):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

This will start a policy server that will serve the policy specified by the `config` and `dir` arguments. The policy will be served on the specified port (default: 8000).

## Querying the remote policy server from your robot code

We provide a client utility with minimal dependencies that you can easily embed into any robot codebase.

First, install the `openpi-client` package in your robot environment:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

Then, you can use the client to query the remote policy server from your robot code. Here's an example of how to do this:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

Here, the `host` and `port` arguments specify the IP address and port of the remote policy server. You can also specify these as command-line arguments to your robot code, or hard-code them in your robot codebase. The `observation` is a dictionary of observations and the prompt, following the specification of the policy inputs for the policy you are serving. We have concrete examples of how to construct this dictionary for different environments in the [simple client example](../examples/simple_client/main.py).

## Action inpainting (chunk overlap)

When executing action chunks open-loop, you can improve smoothness between consecutive chunks by forwarding the tail of the previous chunk as `initial_actions` for the next inference call. This constrains the denoising process so the beginning of the new chunk matches the end of the old one.

```python
next_initial_actions = None

for step in range(num_steps):
    if step % actions_to_execute == 0:
        observation = get_observation()
        output = policy.infer(observation, initial_actions=next_initial_actions)
        chunk = output["actions"]  # (action_horizon, action_dim)

        # Save the overlap region for next call.
        tail_start = actions_to_execute  # e.g. 26
        tail_end = tail_start + tail_actions_to_keep  # e.g. 30
        next_initial_actions = chunk[tail_start:tail_end]

    action = chunk[step % actions_to_execute]
    execute(action)
```

Key points:

- `initial_actions` is in the same (unnormalized) action space as the policy output. Normalization is handled internally.
- This is sampler-level constrained generation — it overrides coordinates in the denoising loop, not prompt conditioning.
- CFG + inpainting is not yet supported; enabling both will raise `NotImplementedError`.
- **Correlation-aware inpainting** (experimental, off by default): propagates the constraint to uncorrelated dimensions via an action-correlation Cholesky factor. Currently only valid with global z-score normalization — it is incompatible with per-timestep or quantile normalization. To try it, set `use_correlation_inpainting: true` in the model config and run `compute_norm_stats.py --compute-action-correlation` to precompute the Cholesky factor.
