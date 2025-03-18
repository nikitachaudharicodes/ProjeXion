from train import main

def test_train():
   main(
      data_path='data/dataset_low_res',
      subset=0.1,
      batch_size=2,
      model='cnn',
      epochs=2,
      lr=0.0002,
      optimizer='AdamW',
      scheduler='ConstantLR',
   )