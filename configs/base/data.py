from vasae.configs.data import DataConfig

data_cfg = DataConfig(
    train_batchsize=128,
    valid_batchsize=128,
    test_batchsize=128,
    use_centralize=True,
    # meta_path,
    # layer_name,
)
