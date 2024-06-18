# Kaggle leap

gscfuseの自動マウント(/etc/fstab に追記)
```
kaggle-leap /home/naoki.a.murakami/kaggle-leap/output gcsfuse rw,file_mode=777,dir_mode=777,allow_other,_netdev,only_dir=kami,rename_dir_limit=20
```
gcsfuse --only-dir kami kaggle-leap /home/naoki.a.murakami/kaggle-leap/output

```
# https://cloud.google.com/compute/docs/disks/add-local-ssd?hl=ja#formatandmount
echo UUID=`sudo blkid -s UUID -o value /dev/md0` input ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf
sudo update-initramfs -u
```

gcloud storage cp -r gs://kaggle-leap/kami/leap-atmospheric-physics-ai-climsim input
gcloud storage cp -r gs://kaggle-leap/kami/preprocess/make_webdataset_batch input
gcloud storage cp -r gs://kaggle-leap/kami/test.parquet input
gcloud storage cp -r gs://kaggle-leap/kami/train.parquet input
gcloud storage cp -r gs://kaggle-leap/kami/valid.parquet input
gcloud storage cp -r gs://kaggle-leap/kami/sample_submission.parquet input


## 特徴
- Docker によるポータブルなKaggleと同一の環境
- Hydra による実験管理
- 実験用スクリプトファイルを major バージョンごとにフォルダごとに管理
- 実験用スクリプトと設定を同一フォルダで局所的に管理して把握しやすくする

## Structure
```text
.
├── .jupyter-settings: jupyter-lab の設定ファイル。compose.yamlでJUPYTERLAB_SETTINGS_DIRを指定している
├── Dockerfile
├── Dockerfile.cpu
├── LICENSE
├── README.md
├── compose.cpu.yaml
├── compose.yaml
├── exp
├── input
├── notebook
├── output
├── utils
└── yamls: データのパスなど各スクリプトに共通する設定を管理
```

## Docker による環境構築

```sh
docker compose build

# bash に入る場合
docker compose run --rm kaggle bash 

# jupyter lab を起動する場合
docker compose up 
```

## スクリプトの実行方法

```sh
python experiments/sample/run.py exp=001
python experiments/sample/run.py exp=base
```

### Hydra による Config 管理
- 各スクリプトに共通する基本的な設定は yamls/config.yaml 内にある
- 各スクリプトによって変わる設定は、実行スクリプトのあるフォルダ(`{major_exp_name}`)の中に `exp/{minor_exp_name}.yaml` として配置することで管理。
    - 実行時に `exp={minor_exp_name}` で上書きする
    - `{major_exp_name}` と `{minor_exp_name}` の組み合わせで実験が再現できるようにする
