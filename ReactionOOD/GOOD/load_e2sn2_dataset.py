from GOOD.data.good_datasets.good_e2sn2 import GOODE2SN2
dataset, meta_info = GOODE2SN2.load(dataset_root='/home/oyxy2019/projects/Explain_ws/ReactionOOD/GOOD/storage/datasets', domain='size', shift='no_shift', generate=True)
"""
    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
"""
print(dataset)
# from GOOD.kernel.main import initialize_model_dataset
# from munch import munchify
# config = {
#     'task': 'train',
#     'random_seed': 123,
#     'exp_round': None,
#     'log_file': 'default',
#     'gpu_idx': 0,
#     'ckpt_root': '/home/oyxy2019/projects/Explain_ws/ReactionOOD/GOOD/storage/checkpoints',
#     'ckpt_dir': '/home/oyxy2019/projects/Explain_ws/ReactionOOD/GOOD/storage/checkpoints/roundNone/GOODE2SN2_scaffold_no_shift/GSATvGIN_3l_meanpool_0.5dp/0.001lr_0wd/GSAT_1.0_True_20_0.5',
#     'save_tag': None,
#     'pytest': False,
#     'pipeline': 'Pipeline',
#     'clean_save': False,
#     'num_workers': 1,
#     'train': {
#         'weight_decay': 0,
#         'save_gap': 20,
#         'tr_ctn': False,
#         'ctn_epoch': 0,
#         'epoch': 0,
#         'alpha': 0.2,
#         'stage_stones': [100000],
#         'num_steps': 10,
#         'mile_stones': [150],
#         'max_epoch': 200,
#         'train_bs': 32,
#         'val_bs': 256,
#         'test_bs': 256,
#         'lr': 0.001
#     },
#     'model': {
#         'dim_hidden': 300,
#         'dim_ffn': 300,
#         'dropout_rate': 0.5,
#         'model_layer': 3,
#         'global_pool': 'mean',
#         'model_name': 'GSATvGIN'
#     },
#     'dataset': {
#         'dataloader_name': 'BaseDataLoader',
#         'dataset_root': '/home/oyxy2019/projects/Explain_ws/ReactionOOD/GOOD/storage/datasets',
#         'generate': True,
#         'dim_node': None,
#         'dim_edge': None,
#         'num_classes': None,
#         'num_envs': None,
#         'dataset_name': 'GOODE2SN2',
#         'domain': 'scaffold',
#         'shift_type': 'no_shift'
#     },
#     'ood': {
#         'extra_param': [True, 20, 0.5],
#         'ood_alg': 'GSAT',
#         'ood_param': 1.0
#     },
#     'tensorboard_logdir': '/home/oyxy2019/projects/Explain_ws/ReactionOOD/GOOD/storage/tensorboard/GOODE2SN2/no_shift/GSAT/1.0',
#     'log_path': '/home/oyxy2019/projects/Explain_ws/ReactionOOD/GOOD/storage/log/roundNone/GOODE2SN2_scaffold_no_shift/GSATvGIN_3l_meanpool_0.5dp/0.001lr_0wd/GSAT_1.0_True_20_0.5/default.log',
#     'test_ckpt': '/home/oyxy2019/projects/Explain_ws/ReactionOOD/GOOD/storage/checkpoints/roundNone/GOODE2SN2_scaffold_no_shift/GSATvGIN_3l_meanpool_0.5dp/0.001lr_0wd/GSAT_1.0_True_20_0.5/best.ckpt',
#     'id_test_ckpt': '/home/oyxy2019/projects/Explain_ws/ReactionOOD/GOOD/storage/checkpoints/roundNone/GOODE2SN2_scaffold_no_shift/GSATvGIN_3l_meanpool_0.5dp/0.001lr_0wd/GSAT_1.0_True_20_0.5/id_best.ckpt',
#     'device': 'cuda',
#     'metric': '<GOOD.utils.metric.Metric object>'
# }
# model, loader = initialize_model_dataset(munchify(config))
