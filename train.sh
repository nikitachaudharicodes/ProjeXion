data_path=data/dataset_low_res
subset=1
batch_size=1
model=cnn
epochs=50
lr=0.0002
optimizer=AdamW
scheduler=ReduceLROnPlateu
echo "python train.py ${data_path} ${subset} ${batch_size} ${model} ${epochs} ${lr} ${optimizer} ${scheduler}"
python train.py ${data_path} ${subset} ${batch_size} ${model} ${epochs} ${lr} ${optimizer} ${scheduler}
python utils/plots.py checkpoints/losses.txt 