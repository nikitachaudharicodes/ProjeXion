data_path=data/dataset_low_res
subset=0.1
batch_size=3
model=cnn
epochs=3
lr=0.0002
optimizer=AdamW
scheduler=ConstantLR
python train.py ${data_path} ${subset} ${batch_size} ${model} ${epochs} ${lr} ${optimizer} ${scheduler}