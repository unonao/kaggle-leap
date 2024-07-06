# メモ

## vast.ai
```
mkdir input
mkdir output

pip install pytorch-lightning
pip install wandb
pip install transformers
pip install hydra-core pandas polars webdataset xarray scipy google-cloud-storage scikit-learn timm netcdf4 pyarrow fastparquet
python3 -m pip install git+https://github.com/sail-sg/Adan.git  
pip install -U rich
python3 -m pip install papermill
pip install torch_geometric

mkdir output/preprocess
gcloud storage cp -r gs://kaggle-leap/kami/preprocess/normalize_007_diff_feat/ output/preprocess
gcloud storage cp -r gs://kaggle-leap/kami/preprocess/normalize_009_rate_feat/ output/preprocess
gcloud storage cp -r gs://kaggle-leap/kami/preprocess/tmelt_tice/ output/preprocess
gcloud storage cp gs://kaggle-leap/kami/118_valid_pred_ch384_6_year_submission.parquet input
```

local.yaml
```
input_dir: /root/kaggle-leap/input
output_dir: /root/kaggle-leap/output
exp_dir: /root/kaggle-leap/output/experiments
preprocess_dir: /root/kaggle-leap/output/preprocess

gcs_bucket: kaggle-leap
gcs_base_dir: kami
```

gcloud storage cp -r output/experiments/222_wo_transformer_vast_single gs://kaggle-leap/experiments

## command
python preprocess/make_sim_data/run.py exp=year_1
python experiments/225_smoothl1_loss/run.py exp=mid_001