# gst
Gumbel Social Transformer.

This is the implementation for the paper
### Learning Sparse Interaction Graphs of Partially Observed Pedestrians for Trajectory Prediction
GST is the abbreviation of our model Gumbel Social Transformer. All code was developed and tested on Ubuntu 18.04 with CUDA 10.2, Python 3.6.9, and PyTorch 1.7.1. <br/>

### Setup
##### 1. Create a Virtual Environment. (Optional)
```
virtualenv -p /usr/bin/python3 myenv
source myenv/bin/activate
```
##### 2. Install Packages
You can run either <br/>
```
pip install -r requirements.txt
```
or <br/>
```
pip install numpy
pip install scipy
pip install matplotlib
pip install tensorboardX
pip install pandas
pip install pyyaml
pip install jupyterlab
pip install torch==1.7.1
pip install tensorflow
```
or for the machine with 3090s,
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
##### 3. Create Folders and Dataset Files.
```
sh scripts/make_dirs.sh
sh scripts/data/create_batch_datasets_eth_ucy.sh
python scripts/data/create_batch_datasets_sdd.py 
```

### Training and Evaluation on Various Configurations
To train and evaluate a model with n=1, i.e., the target pedestrian pays attention to at most one partially observed pedestrian, run
```
sh scripts/train_sparse.sh
sh scripts/eval_sparse.sh
```
To train and evaluate a model with n=1 and temporal component as a temporal convolution network, run
```
sh scripts/train_sparse_tcn.sh
sh scripts/eval_sparse_tcn.sh
```
To train and evaluate a model with full connection, i.e., the target pedestrian pays attention to all partially observed pedestrians in the scene, run
```
sh scripts/train_full_connection.sh
sh scripts/eval_full_connection.sh
```
To train and evaluate a model in which the target pedestrian pays attention to all fully observed pedestrians in the scene, run
```
sh scripts/train_full_connection_fully_observed.sh
sh scripts/eval_full_connection_fully_observed.sh
```

### Important Arguments for Building Customized Configurations
- `--spatial_num_heads_edges`: n, i.e., the upperbound number of pedestrians that the target pedestrian can pay attention to in the scene. When n=0, it is defined as full connection, i.e., the target pedestrian pays attention to all pedestrians in the scene. Default is 4.
- `--only_observe_full_period`: The target pedestrian only pays attention to fully observed pedestrians. Default is False.
- `--temporal`: Temporal component. `lstm` is Masked LSTM, and `temporal_convolution_net` is temporal convolution network. Default is `lstm`.
- `--decode_style`: Decoding style. It has to match the option `--temporal`. `recursive` matches `lstm`, and `readout` matches `temporal_convolution_net`. Default is `recursive`.
- `--ghost`: Add a ghost pedestrian in the scene to encourage sparsity. When `--spatial_num_heads_edges` is set as zero, i.e., the target pedestrian pays attention to all pedestrians in the scene, `--ghost` has to be set as False. Default is False.
