# Go2 Joystick ONNX Deploy

Fresh ROS 2/Unitree deployment node for `Go2JoystickFlatTerrain` RSL-RL ONNX policies.

This deploy stack is based on Ubuntu 22.04 and ROS 2 Humble.

## File Structure

```text
deploy/
├── CMakeLists.txt
├── README.md
├── config.yaml
├── locomotion_node.cpp
├── model/
│   └── policy_deploy.onnx
├── package.xml
├── run_build.sh
├── terminal_cmd.py
└── wireless_cmd.py
```

## Policy File

Export from training:

```sh
train-rsl-ppo \
  --env_name=Go2JoystickFlatTerrain \
  --export_onnx_only \
  --load_run_name=<run_name> \
  --checkpoint_num=<checkpoint>
```

Then place the exported file where `config.yaml` points:

```sh
mkdir -p deploy/model
cp logs/<run_name>/policy_deploy.onnx deploy/model/policy_deploy.onnx
```

Set the matching policy path in `config.yaml`:

```yaml
policy:
  path: "model/policy_deploy.onnx"
```

The node expects:

```text
input:  obs      [1, 450]
output: actions  [1, 12]
```

## Build

Source your ROS 2 and Unitree SDK environment first, then:

```sh
cd deploy
./run_build.sh
source install/share/go2_joystick_deploy/local_setup.bash
```

Source `install/share/go2_joystick_deploy/local_setup.bash` in every terminal
that uses ROS CLI tools with this package.

## Run With Unitree MuJoCo

```sh
cd deploy
./locomotion_node --net lo --config config.yaml
```

In another terminal, run the existing command publisher:

```sh
python ../deploy/terminal_cmd.py
```

Use `wireless_cmd.py` instead of `terminal_cmd.py` to publish commands from a
Unitree wireless controller.

During policy control, the node keeps policy actions in MuJoCo policy order and
maps them to Unitree SDK motor order (`FR`, `FL`, `RR`, `RL`) when publishing
low-level commands. The target joint command is:

```text
target_q = default_angles + clipped_policy_action * action_scale
```

## Real Robot Notes

Hardware setup details depend on the onboard computer, network interface, Unitree
SDK installation, and robot safety procedure; for a fuller hardware reference,
see [ASIG-X/REASAN](https://github.com/ASIG-X/REASAN).

For the real robot, set this in `config.yaml`:

```yaml
unitree_services:
  enable: true
```

That makes the node call Unitree `SportClient::StandDown()` and
`MotionSwitcherClient::ReleaseMode()` before low-level control. Keep it `false`
for `unitree_mujoco`.

On shutdown or Ctrl+C, the node interpolates to `sit_angles` and then sends a
damping command.
