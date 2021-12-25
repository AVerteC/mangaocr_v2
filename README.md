# mangaocr_v2
	Alan Chen
	This project processes Manga109 into yolo/coco format, and trains yolov5 with it.


# Use Ubuntu 20.04
# Install latest Nvidia drivers
# Install anaconda

# Create conda env for this project
	$conda create -n mangaocrV2 python=3.7 (or higher) anaconda
	$conda activate mangaocrV2

# Download yolov5 latest version
	Download the source zip from the latest release and unzip it to ./yolov5 directory ...

# Install pytorch with cuda support
	$conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Install yolov5 dependencies
	$conda install --file ./yolov5/requirements.txt
	$pip install -r requirements.txt
	$pip install wandb

# Setup data
	Download Manga109 and extract it to Manga109 directory ...
	Create directories ./datasets/Manga109
	Run manga109_to_yolov5.py
	Copy Manga109.yaml to ./yolov5/data

# Train yolov5
# Dual GPU (DistributedDataParallel) 
	$python -m torch.distributed.run --nproc_per_node 2 train.py --img 1280 --batch 4 --epochs 300 --data Manga109.yaml --weights yolov5m6.pt --cfg yolov5m6.yaml
# Single GPU (DataParallel) 
	$python train.py --img 1280 --workers 1 --batch 3 --epochs 300 --data Manga109.yaml --weights yolov5l6.pt --cfg yolov5l6.yaml

# Test yolov5
	$python val.py --weights runs/train/exp18(your path to the model)/weights/best.pt --data data/Manga109.yaml --task test --name manga109_best_1280 --imgsz 1280 --batch-size 2


# Known issues:
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
place this workaround in train.py:
`import os`
`os.environ['KMP_DUPLICATE_LIB_OK']='True'`
	
