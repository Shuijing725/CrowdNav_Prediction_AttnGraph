# gst
Gumbel Social Transformer trained with data provided in Nov, 2021 by Shuijing.

## How to set up
##### 1. Create a Virtual Environment.
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
##### 3. Create Folders, and download datasets and pretrained models.
```
sh run/make_dirs.sh
sh run/download_datasets_models.sh
```

## How to run
##### 1. Test pretrained models.
```
source myenv/bin/activate
sh tuning/211209-test_shuijing.sh 
```
And you should see outputs which look like below:
```
--------------------------------------------------
dataset:  sj
No ghost version.
new gst
new st model
Loaded configuration:  100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000
Test loss from loaded model:  -4.0132044252296755
dataset: sj | test aoe: 0.1579 | test aoe std: 0.0175 | test foe: 0.3216 | test foe std: 0.0408 | min aoe: 0.1339, min foe: 0.2638
```
##### 2. Test wrapper.
Run
```
cd gst_updated
sh run/download_wrapper_data_model.sh
```
to collect model and demo data for wrapper. Then run
```
python scripts/wrapper/crowd_nav_interface_parallel.py
```
And you should see outputs which look like below:
```
INPUT DATA
number of environments:  4
input_traj shape:  torch.Size([4, 15, 5, 2])
input_binary_mask shape: torch.Size([4, 15, 5, 1])

No ghost version.
new gst
new st model
LOADED MODEL
device:  cuda:0


OUTPUT DATA
output_traj shape:  torch.Size([4, 15, 5, 5])
output_binary_mask shape: torch.Size([4, 15, 1])

0.png is created.
1.png is created.
2.png is created.
3.png is created.
```

## Observation dimensions for traj pred (by Shuijing)
As far as I know, the wrapper for SRNN model does not need to handle multiple timesteps simultaneously. So the time dimension is not mentioned here.
### Output format
The output of traj pred model (or the input of the planner model) at each timestep t is a 3D float array with size = [number_of_pedestrains, number of prediction steps, 5], where 5 includes [mu_x, mu_y, sigma_x, sigma_y, correlation coefficient].

If number of parallel environments > 1, the array becomes 4D with size = [number of env, number_of_pedestrains, number of prediction steps, 5].

The order of the dimensions can be swapped, and the dimensions can be combined in row-major order if needed. For example, [number of env, number_of_pedestrains, number of prediction steps * 5] is good as long as the last dimension is in row-major order.

### Input format 
The input format of traj pred model at each timestep t is a 2D float array with size = [number of pedestrains, 2], where 2 includes [px, py] of each pedestrain. 

If needed, the displacements of all pedestrains can also be added. In this case, input size = [number of pedestrains, 4], where 4 includes [px, py, delta_px, delta_py] of each pedestrain. 

In addition, there is a binary mask with size = [number of pedestrains, 1] that indicates the presence of each human. If a human with ID = i is present, then mask[i] = True, else mask[i] = False.

If number of parallel environments > 1, similarly, the number_of_env is added as the first dimension in addition to the above sizes.

