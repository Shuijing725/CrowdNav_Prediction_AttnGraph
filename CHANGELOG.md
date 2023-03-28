# Changelog

All notable changes to this project will be documented in this file.

## 2023-03-28
### Changed
- Changed instructions for Pytorch installation (moved Pytorch out of requirements.txt, and added link of official website in readme.md)
- Fixed bug for 'visible_masks' in generate_ob function in crowd_sim_var_num.py and crowd_sim_pred_real_gst.py, so that the shape of observation will not throw errors when config.sim.human_num_range > 0

## 2023-01
Initial commit
