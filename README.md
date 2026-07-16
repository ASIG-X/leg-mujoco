# Unitree Go2 Locomotion

This is my practice of using MuJoCo Playground for training Unitree Go2 using PPO with real-world deployment. The training part is largely based on MuJoCo Playground with actual proprioceptions and a minimal set of rewards. 

This repository is aimed for beginners such as undergraduate and graduate students. Please feel free to contact me if you have questions. 

Author: Ella Yixuan Qiu  
Email: yixuan.qiu [at] rug.nl

## Go2 environment structure

The Go2 implementation lives in `mujoco_playground/_src/locomotion/go2/`:

- `base.py`: common Go2 MJX environment setup, asset loading from
  `mujoco_menagerie/unitree_go2`, PD gain configuration, and sensor helpers.
- `joystick.py`: velocity-command tracking task used by
  `Go2JoystickFlatTerrain` and `Go2JoystickRoughTerrain`.
- `go2_constants.py`: XML paths, root body name, foot site/geom names, and
  sensor names.
- `randomize.py`: domain randomization for friction, mass/inertia, PD gains,
  and torso center-of-mass offsets.
- `getup.py` and `handstand.py`: additional Go2 recovery and balance tasks.
- `xmls/`: MJCF scene/model files for flat and rough terrain variants,
  including feet-only and full-collision models.

The joystick policy uses a 12-dimensional action in MuJoCo actuator order:

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

The actor observation is one 45-dimensional frame consisting of local angular
velocity, projected gravity, command, joint-position error, joint velocity, and
previous action. With `history_len=10`, exported policies expect:

```text
input:  obs      [1, 450]
output: actions  [1, 12]
```

## Installation 
```sh
conda create -n env_mujocoplayground python=3.12 -y  
conda activate env_mujocoplayground # make sure environment is always activated in following commands
cd leg-mujoco
pip install -e . # install Go2 customized MuJoCo Playground
# install required dependencies
pip install rsl-rl-lib wandb
```

## Train policy
```sh
train-rsl-ppo --env_name=Go2JoystickFlatTerrain
```

## Play policy
It is necessary to specify `play_only`, `load_run_name`, and `checkpoint_num` (optional).  
```sh
train-rsl-ppo \
  --env_name=Go2JoystickFlatTerrain \
  --play_only \
  --load_run_name=<run_name> \
  --checkpoint_num=<number>
```

If `checkpoint_num` is not specified, the latest checkpoint is automatically chosen.

## Export policy to ONNX
This is for Sim-to-Sim validation and real-world deployment, we use [ONNX format](https://onnx.ai/onnx/intro/index.html) to deploy.  
```sh
train-rsl-ppo \
  --env_name=Go2JoystickFlatTerrain \
  --export_onnx_only \
  --load_run_name=<run_name> \
  --checkpoint_num=<number>
```

## Real-world Deployment
The ROS 2/Unitree deployment code lives in [deploy](./deploy/) and runs exported
ONNX policies for `Go2JoystickFlatTerrain`. See [deploy/README.md](./deploy/README.md)
for setup, configuration, sim-to-sim, and real-robot notes.

<img src="./assets/mujocopg-deploy.gif" alt="Go2 deployment demo" width="500">

## Acknowledgements

This work is based on and adapted from:

- [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground),
  the upstream MuJoCo Playground project.
- [aatb-ch/mujoco_playground/tree/go2](https://github.com/aatb-ch/mujoco_playground/tree/go2),
  whose Go2 branch informed the Go2 environment structure.

## License
This is essentially an engineering practice of MuJoCo Playground on Unitree Go2. For more information on licensing, please refer to the original MuJoCo Playground.


## Disclaimer
This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

**Safety Warning:** This code is intended for research and educational purposes only. Deploying code on physical robots involves significant risks, including but not limited to hardware damage, property damage, and personal injury.

1. **Hardware Safety:** The control policies and algorithms provided may generate commands that exceed the physical limits of the robot hardware, potentially leading to overheating, mechanical failure, or permanent damage.  
2. **Operational Environment:** Ensure the robot is operated in a safe, controlled environment with adequate space and safety barriers. Do not operate the robot near people, pets, or fragile objects.  
3. **Emergency Stop:** Always have a reliable, hardware-based emergency stop (E-Stop) mechanism immediately accessible when running the robot. Software-based stops may fail.
4. **Supervision:** Never leave the robot unattended while it is powered on or operating.
By using this software, you acknowledge and agree that you are solely responsible for ensuring the safety of the deployment and for any consequences resulting from its use. The authors assume no responsibility for any damage to equipment, injury to persons, or other losses caused by the use of this code.

**The authors strictly oppose any military use of this work!**
