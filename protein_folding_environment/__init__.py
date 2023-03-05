from gym.envs.registration import register

register(
    id='ProteinFolding2DEnv',
    entry_point='protein_folding_environment.environment_2d:ProteinFolding2DEnv',
    max_episode_steps=300,
)

register(
    id='ProteinFolding3DEnv',
    entry_point='protein_folding_environment.environment_3d:ProteinFolding3DEnv',
    max_episode_steps=300,
)

register(
    id='ProteinFoldingLRFActions',
    entry_point='protein_folding_environment.environment_3d:ProteinFoldingLRFActions',
    max_episode_steps=300,
)