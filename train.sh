data_path=data/dataset_low_res
subset=0.7
batch_size=32
context_size=5
model=mvsnet
loss=cauchy
epochs=8
lr=0.0005
optimizer=AdamW
scheduler=ConstantLR
echo "python train.py ${data_path} ${subset} ${context_size} ${batch_size} ${model} ${loss} ${epochs} ${lr} ${optimizer} ${scheduler}"
python train.py ${data_path} ${subset} ${context_size} ${batch_size} ${model} ${loss} ${epochs} ${lr} ${optimizer} ${scheduler}
python utils/plots.py checkpoints/losses.txt 
