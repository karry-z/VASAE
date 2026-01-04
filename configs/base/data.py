from vasae.configs.data import DataConfig

data_cfg = DataConfig(
    train_batchsize=32,
    valid_batchsize=32,
    use_centralize=True,
    meta_path=r"/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2/meta.json",
    layer_name="transformer.h.5",
)
