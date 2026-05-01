# Dataset Preparation Notes

## Official Sources

- 102 Category Flower Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Road Vehicle Images Dataset: https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset
- Stanford Background Dataset: http://dags.stanford.edu/projects/scenedataset.html

Place datasets under `data/` using this layout:

```text
data/
  flower102/
  road_vehicle/
  stanford_background/
    images/
    labels/
    splits/
  videos/
```

In this workspace, large datasets are stored outside the repo:

```text
/data/yzj/homework2_data/
```

The project keeps symlinks in `data/`:

```text
data/flower102 -> /data/yzj/homework2_data/flower102
data/road_vehicle -> /data/yzj/homework2_data/road_vehicle
data/stanford_background -> /data/yzj/homework2_data/stanford_background
data/videos -> /data/yzj/homework2_data/videos
```

Pretrained/downloaded model files should also stay outside the repo, for example:

```text
/data/yzj/homework2_data/pretrained/yolov8n.pt
```

Keep raw archives or manually downloaded files out of git. Save any generated split files or YOLO dataset YAML files through the task-specific scripts/configs so experiments are reproducible.

## Flower102

The assignment link points to the Oxford VGG Flower102 page. That page provides:

- `102flowers.tgz`
- `imagelabels.mat`
- `setid.mat`
- optional segmentations

The classification configs use `torchvision.datasets.Flowers102` by default:

```yaml
data:
  source: torchvision
  data_dir: data/flower102
  download: true
```

This is the simplest path because torchvision handles the official labels and train/val/test splits. The first training run may download the Oxford files automatically.

Manual source files, if needed:

```bash
mkdir -p data/flower102
wget -P data/flower102 https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget -P data/flower102 https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
wget -P data/flower102 https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
```

If you instead reorganize images into an ImageFolder layout with one directory per class, set `data.source: folder` in the classification config.

## Road Vehicle Images Dataset

The Road Vehicle dataset is hosted on Kaggle. Download normally requires a Kaggle account and API token.

Browser:

```text
https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset
```

Kaggle CLI:

```bash
mkdir -p data/road_vehicle
kaggle datasets download -d ashfakyeafi/road-vehicle-images-dataset -p data/road_vehicle --unzip
```

KaggleHub, which supports the newer `KGAT_...` token style:

```bash
python -c "import kagglehub; kagglehub.dataset_download('ashfakyeafi/road-vehicle-images-dataset', output_dir='/data/yzj/homework2_data/road_vehicle')"
```

If this fails with an authentication error, create a Kaggle API token:

1. Log in at https://www.kaggle.com/
2. Open account settings.
3. In the API section, choose "Create New Token".
4. Save the downloaded `kaggle.json` as:

```text
~/.kaggle/kaggle.json
```

5. Restrict the file permission:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

This workspace also keeps a local copy of the newer Kaggle access token at:

```text
.secrets/kaggle/access_token
```

The `.secrets/` directory is git-ignored. To restore the token into Kaggle's expected user config location, run:

```bash
scripts/setup_kaggle_token.sh
```

After download, inspect the label format. If it is already YOLO-format, update:

```text
configs/detection/road_vehicle_data.yaml
```

with the correct `train`, `val`, `nc`, and `names` values. If it is not YOLO-format, convert annotations before training.

Current downloaded layout:

```text
data/road_vehicle/trafic_data/
  data_1.yaml
  train/images/
  train/labels/
  valid/images/
  valid/labels/
```

`configs/detection/road_vehicle_data.yaml` is already pointed at this YOLO-format layout.

## Stanford Background Dataset

The Stanford page links the archive:

```text
http://dags.stanford.edu/data/iccv09Data.tar.gz
```

Download:

```bash
mkdir -p data/stanford_background
wget -O data/stanford_background/iccv09Data.tar.gz http://dags.stanford.edu/data/iccv09Data.tar.gz
tar -xzf data/stanford_background/iccv09Data.tar.gz -C data/stanford_background --strip-components=1
```

For the Stanford Background dataset, the segmentation loader supports raw semantic masks named like `labels/<image_stem>.regions.txt`. Negative mask values are treated as ignored pixels when the configs use `ignore_index: -1`.

Create split files if the archive does not provide the exact ones expected by the configs:

```text
data/stanford_background/splits/train.txt
data/stanford_background/splits/val.txt
```

Each line may contain just an image stem, for example:

```text
9005294
```

or an explicit image/mask pair:

```text
images/9005294.jpg labels/9005294.regions.txt
```
