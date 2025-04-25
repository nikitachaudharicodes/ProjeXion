data_path=data/dataset_low_res
subset=0.7
batch_size=64
context_size=4
model=mvsnet
epochs=20
lr=0.00002
optimizer=AdamW
scheduler=ConstantLR
echo "python train.py ${data_path} ${subset} ${context_size} ${batch_size} ${model} ${epochs} ${lr} ${optimizer} ${scheduler}"
python train.py ${data_path} ${subset} ${context_size} ${batch_size} ${model} ${epochs} ${lr} ${optimizer} ${scheduler}
python utils/plots.py checkpoints/losses.txt 
