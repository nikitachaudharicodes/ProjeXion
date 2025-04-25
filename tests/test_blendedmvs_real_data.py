# import shutil
# import pytest
# from pathlib import Path
# import torch
# from torch.utils.data import DataLoader
# import train

# from dataloaders.blendedMVS import BlendedMVS  # adjust import if needed
# from train import main                         # so test_train still works

# @pytest.fixture
# def lowres_dir(tmp_path):
#     """
#     Copy our shipped data_sample/BlendedMVS/dataset_low_res
#     into a temp dir so tests do not clobber our real data.
#     """
#     src = Path(__file__).parent.parent / 'data_sample' / 'BlendedMVS' / 'dataset_low_res'
#     dst = tmp_path / 'dataset_low_res'
#     shutil.copytree(src, dst)
#     return dst

# def test_length_and_item_shapes(lowres_dir):
#     ds = BlendedMVS(data_path=str(lowres_dir), subset=1, partition='train', context_size=2)
#     assert len(ds) == 2  # There should be 2 train examples
#     item = ds[0]
#     # dataset must return exactly 6 things:
#     assert isinstance(item, tuple) and len(item) == 4
#     imgs, intrinsics, extrinstics, depth = item

#     # shapes sanity checks:
#     assert imgs.ndim == 4                      # (1+ctx, 3, H, W)
#     assert depth.ndim == 3                     # (1, H, W)
#     assert K_ref.shape == (3, 3)
#     assert K_src.ndim == 3                     # (ctx, 3, 3)
#     assert Rt_ref.shape == (3, 4)
#     assert Rt_src.ndim == 3                    # (ctx, 3, 4)

# def test_collate_fn_with_context(lowres_dir):
#     ds = BlendedMVS(data_path=str(lowres_dir), subset=1, partition='train', context_size=2)
#     loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)
#     batch = next(iter(loader))
#     # collate_fn must return the 6 batched tensors
#     assert isinstance(batch, tuple) and len(batch) == 6
#     b_imgs, b_depth, b_K_ref, b_K_src, b_Rt_ref, b_Rt_src = batch

#     B = 2
#     assert b_imgs.shape[0]  == B
#     assert b_depth.shape[0] == B
#     assert b_K_ref.shape[0] == B
#     assert b_K_src.shape[0] == B
#     assert b_Rt_ref.shape[0] == B
#     assert b_Rt_src.shape[0] == B

# def test_online_sampling(lowres_dir):
#     # iteration over the dataset must yield exactly 6-tuples
#     ds = BlendedMVS(data_path=str(lowres_dir), subset=1, partition='train', context_size=2)
#     for item in ds:
#         assert isinstance(item, tuple) and len(item) == 6

# def test_train_runs(lowres_dir, monkeypatch):
#     # point train.main at our tmp low-res dir instead of 'data/dataset_low_res'
#     monkeypatch.chdir(str(lowres_dir.parent))   # so 'data/dataset_low_res' → lowres_dir parent
#     train.CHECKPOINTS.mkdir(parents=True, exist_ok=True)

#     # now call main exactly as test_train.py does:
#     main(
#       data_path=str(lowres_dir),
#       subset=0.01,
#       batch_size=2,
#       model='cnn',
#       epochs=2,
#       lr=0.0002,
#       optimizer='AdamW',
#       scheduler='ConstantLR',
#     )
#     # if we get here without AssertionError, success!
