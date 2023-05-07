from gym.envs.registration import register

register(
    id='ProteinFoldingSquareEnv',
    entry_point='protein_folding_environment.environment_2d:ProteinFoldingSquareEnv',
    max_episode_steps=100000,
)

register(
    id='ProteinFolding3DEnv',
    entry_point='protein_folding_environment.environment_3d:ProteinFolding3DEnv',
    max_episode_steps=100000,
)

register(
    id='ProteinFoldingLRF2DEnv',
    entry_point='protein_folding_environment.environment_LRF:ProteinFoldingLRF2DEnv',
    max_episode_steps=100000,
)

register(
    id='ProteinFoldingTriangularEnv',
    entry_point='protein_folding_environment.environement_triunghiular:ProteinFoldingTriangularEnv',
    max_episode_steps=100000,
)
