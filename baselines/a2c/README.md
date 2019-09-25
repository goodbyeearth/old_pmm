# A2C

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.run --alg=a2c --env=PongNoFrameskip-v4` runs the algorithm for 40M frames = 10M timesteps on an Atari Pong. See help (`-h`) for more options
- also refer to the repo-wide [README.md](../../README.md#training-models)

## Files
- `run_atari`: file used to run the algorithm.
- `policies.py`: contains the different versions of the A2C architecture (MlpPolicy, CNNPolicy, LstmPolicy...).
- `a2c.py`: - Model : class used to initialize the step_model (sampling) and train_model (training)
	- learn : Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.
- `runner.py`: class used to generates a batch of experiences

- `a2c_collect.py`: 采集样本。python collect_data_distillation.py --alg=a2c --submodule=a2c_collect --env=PongNoFrameskip-v4 --network=cnn --load_path=initial_parameter/updn_data/simple_agent.npz --nsteps=20 --num_timesteps=64000
- `a2c_compute_FM`: 根据EWC来计算fisher matrix。
- `a2c_distillation`: 结合几个不同策略智能体的样本，进行策略蒸馏。
- `a2c_distillation_FM`: 结合EWC进行策略蒸馏。
- `a2c_kfac_distillation`: 结合kfc进行策略蒸馏。
- `a2c_kfac_fm`: 根据kfc计算fisher matrix.'collect_data_distillation', '--alg=a2c', '--submodule=a2c_kfac_fm', '--env=PongNoFrameskip-v4','--network=cnn','--load_path=initial_parameter/26000_kfac','--nstep=1','--save_path=fisher','--num_timesteps=5000', '--num_env=1'