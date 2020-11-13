# python train.py --gpus=1 --train_slice=":100" --val_slice=":100" --precision=16 --min_epochs=1 --max_epochs=1 --no_upload
python train.py --tpu_cores=8 --train_slice="50%:" --val_slice=":" --precision=16 --min_epochs=1 --max_epochs=1 --use_english_weights --wte_path=data/used/german_wte_weight.pth --auto_lr_find=True --batch_size=2 --accumulate_grad_batches=16
python train.py --tpu_cores=8 --train_slice="50%:" --val_slice=":" --precision=16 --min_epochs=1 --max_epochs=1 --use_english_weights --wte_path=data/used/german_wte_weight.pth --use_onecycle --batch_size=2 --accumulate_grad_batches=16

gcloud compute tpus stop gerpt2 --zone=europe-west4-a
sudo shutdown -h now
