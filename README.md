# Multiagent-RL
Multiagent Reinforcement learning algorithm for Pommerman

How to run it:
python run.py --alg=a2c --env=PongNoFrameskip-v4 --num_timesteps=1e11 --network=cnn --load_path=XXX --save_path=XXX --num_env==XX


# 黄
## featurize
maps是多层的11*11矩阵

(编号不完全准确，只是给出大致顺序)

0: bomb_blast_strength

1: bomb_life

2: 我的智能体的位置

3: ammo

4: blast_strength

5: can_kick

6：队友的位置

6-8或7-8: 敌人的位置

9-18: 分别为棋盘编号0-9的物体的位置

19: step_count
