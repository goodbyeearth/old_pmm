import numpy as np


def old_featurize(obs):
    # TODO: history of n moves?
    board = obs['board']

    # convert board items into bitmaps
    maps = [board == i for i in range(10)]
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    # duplicate ammo, blast_strength and can_kick over entire map
    #创建一个由常数填充的数组,第一个参数是数组的形状，第二个参数是数组中填充的常数。
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[obs['position']] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'].value)
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e.value for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))
    # maps.append(np.full(board.shape, obs['step_count']))

    #assert len(maps) == NUM_CHANNELS
    return np.stack(maps, axis=2)

def featurize(obs,player):
    board = obs['board']
    maps = []
    # maps.append(board == 3)
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    player = player + 10        # 我的智能体在棋盘的编号
    maps.append(board == player)

    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'].value)
    else:
        maps.append(np.zeros(board.shape))

    for e in obs['enemies']:
        maps.append(board == e.value)

    for i in [0,1,2,4,5,6,7,9]:
        maps.append(board == i)

    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape,obs['step_count']/799))

    # result = np.stack(maps, axis=2)
    # print("After stack, shape:", result.shape)
    # TODO:axis好像有问题
    return np.stack(maps, axis=2)

def available_action(state):
    position = state['position']
    board = state['board']
    x = position[0]
    y = position[1]
    if state['can_kick']:
        avail_path = [0,3,6,7,8]
    else:
        avail_path = [0,6,7, 8]
    action = [0, -9999, -9999, -9999, -9999, 0]
    if state['ammo'] == 0:
        action[-1] = -9999
    if (x - 1) >= 0:
        if board[x-1,y] in avail_path:
            #可以往上边走
            action[1] = 0
    if (x + 1) <= 10:
        if board[x+1, y] in avail_path:
            #可以往下边走
            action[2] = 0
    if (y - 1) >= 0:
        if board[x,y-1] in avail_path:
            #可以往坐边走
            action[3] = 0
    if (y + 1) <= 10:
        if board[x,y+1] in avail_path:
            #可以往右边走
            action[4] = 0
    action = np.asarray(action).reshape(-1, 6)
    return action