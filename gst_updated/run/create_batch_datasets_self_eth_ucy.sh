for dataset in eth hotel univ zara1 zara2; do
    python -u scripts/data/create_datasets_self_eth_ucy.py --dataset $dataset | tee -a logs/create_batch_datasets.txt
    python -u scripts/data/create_batch_datasets_self_eth_ucy.py --dataset $dataset | tee -a logs/create_batch_datasets.txt
    rm -rf datasets/self_eth_ucy/${dataset}/${dataset}_dset_train_trajectories.pt
    rm -rf datasets/self_eth_ucy/${dataset}/${dataset}_dset_val_trajectories.pt
    rm -rf datasets/self_eth_ucy/${dataset}/${dataset}_dset_test_trajectories.pt
done