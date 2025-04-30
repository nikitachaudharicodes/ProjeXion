# Ablations
run_name=n_depths_15
model=projexion
context_size=5
n_depths=15
loss=cauchy

# Constante values
data_path=data/dataset_low_res
subset=0.7
batch_size=32
epochs=8
lr=0.0005
optimizer=AdamW
scheduler=ConstantLR

echo "python train.py ${data_path} ${subset} ${context_size} ${n_depths} ${batch_size} ${model} ${loss} ${epochs} ${lr} ${optimizer} ${scheduler} ${run_name}"
python train.py ${data_path} ${subset} ${context_size} ${n_depths} ${batch_size} ${model} ${loss} ${epochs} ${lr} ${optimizer} ${scheduler} ${run_name}
