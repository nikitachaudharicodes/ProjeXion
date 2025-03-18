from train import main

def test_train():
   main(
      data_path='data_sample/BlendedMVS/dataset_low_res',
      subset=1.0,
      batch_size=2,
      model='cnn',
      epochs=2,
      lr=0.02,
      optimizer='AdamW',
      scheduler='ConstantLR',
   )