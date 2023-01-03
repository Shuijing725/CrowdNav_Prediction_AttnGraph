temporal=faster_lstm
decode_style=recursive
only_observe_full_period=

save_epochs=10
spatial=gumbel_social_transformer
rotation_pattern=random
lr=1e-3
spatial_num_heads=8
embedding_size=64
lstm_hidden_size=64
batch_size=1

deterministic=
detach_sample=
init_temp=0.5
clip_grad=10.
temp_epochs=100

for random_seed in 1000; do
    for num_epochs in 100; do
        for spatial_num_layers in 1; do
            for lr in 1e-3; do
                for dataset in sj; do
                    for spatial_num_heads_edges in 0; do
                        CUDA_VISIBLE_DEVICES=1 python -u scripts/experiments/train.py --spatial $spatial --temporal $temporal --lr $lr\
                            --dataset $dataset --temp_epochs $temp_epochs --num_epochs $num_epochs --save_epochs $save_epochs\
                            $detach_sample\
                            --spatial_num_heads $spatial_num_heads --spatial_num_layers $spatial_num_layers\
                            --decode_style $decode_style \
                            --spatial_num_heads_edges $spatial_num_heads_edges --random_seed $random_seed\
                            $deterministic --lstm_hidden_size $lstm_hidden_size --embedding_size $embedding_size\
                            --batch_size $batch_size \
                            --init_temp $init_temp $only_observe_full_period\
                            --obs_seq_len 5 --pred_seq_len 5 \
                            | tee -a logs/211130.txt
                    done
                done
            done
        done
    done
done

# for random_seed in 1000; do
#     for num_epochs in 100; do
#         for spatial_num_layers in 1; do
#             for lr in 1e-3; do
#                 for dataset in deathCircle hyang; do
#                     for spatial_num_heads_edges in 0; do
#                         CUDA_VISIBLE_DEVICES=1 python -u scripts/experiments/train.py --spatial $spatial --temporal $temporal --lr $lr\
#                             --dataset $dataset --temp_epochs $temp_epochs --num_epochs $num_epochs --save_epochs $save_epochs\
#                             $detach_sample\
#                             --spatial_num_heads $spatial_num_heads --spatial_num_layers $spatial_num_layers\
#                             --decode_style $decode_style \
#                             --spatial_num_heads_edges $spatial_num_heads_edges --random_seed $random_seed\
#                             $deterministic --lstm_hidden_size $lstm_hidden_size --embedding_size $embedding_size\
#                             --batch_size $batch_size \
#                             --init_temp $init_temp --only_observe_full_period\
#                             | tee -a logs/211112.txt
#                     done
#                 done
#             done
#         done
#     done
# done


# for random_seed in 1000; do
#     for num_epochs in 100; do
#         for spatial_num_layers in 1; do
#             for lr in 1e-3; do
#                 for dataset in deathCircle hyang; do
#                     for spatial_num_heads_edges in 16 8 4 2 1 0; do
#                         CUDA_VISIBLE_DEVICES=1 python -u scripts/experiments/train.py --spatial $spatial --temporal $temporal --lr $lr\
#                             --dataset $dataset --temp_epochs $temp_epochs --num_epochs $num_epochs --save_epochs $save_epochs\
#                             $detach_sample\
#                             --spatial_num_heads $spatial_num_heads --spatial_num_layers $spatial_num_layers\
#                             --decode_style $decode_style \
#                             --spatial_num_heads_edges $spatial_num_heads_edges --random_seed $random_seed\
#                             $deterministic --lstm_hidden_size $lstm_hidden_size --embedding_size $embedding_size\
#                             --batch_size $batch_size \
#                             --init_temp $init_temp $only_observe_full_period\
#                             | tee -a logs/211112.txt
#                     done
#                 done
#             done
#         done
#     done
# done







