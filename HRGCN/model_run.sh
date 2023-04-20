sudo update-alternatives --config gcc

sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda
pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html


python main.py \
--sagemaker False \
--num_node_types 3 \
--num_edge_types 11 \
--num_train 820 \
--source_types 0,1,2 \
--sampling_size 820 \
--batch_s 82 \
--mini_batch_s 82 \
--unzip False \
--s3_stage True \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 16 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 1 \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--eval_method svdd \
--model_path ../models/model_save_dgraph_base \
--data_path ../dataset \
--job_prefix test
