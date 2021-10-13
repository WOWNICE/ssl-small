baseline_path=/raid/checkpoints_ep200_baseline
full_path=/raid/checkpoints_ep800
datapath=$1
dataset=$2

mobile_large_base=$baseline_path/mobile_large/checkpoint_0199.pth.tar
mobile_large_200=$full_path/mobile_large/checkpoint_0200.pth.tar
mobile_large_800=$full_path/mobile_large/checkpoint_0799.pth.tar

mobile_small_base=$baseline_path/mobile_small/checkpoint_0199.pth.tar
mobile_small_200=$full_path/mobile_small/checkpoint_0200.pth.tar
mobile_small_800=$full_path/mobile_small/checkpoint_0799.pth.tar

efficient_base=$baseline_path/efficient_b0/checkpoint_0199.pth.tar
efficient_200=$full_path/efficient_b0/checkpoint_0200.pth.tar
efficient_800=$full_path/efficient_b0/checkpoint_0799.pth.tar

resnet_base=$baseline_path/resnet18/checkpoint_0199.pth.tar
resnet_200=$full_path/resnet18/checkpoint_0200.pth.tar
resnet_800=$full_path/resnet18/checkpoint_0799.pth.tar

vit_base=$baseline_path/vit_tiny/checkpoint_0199.pth.tar
vit_200=$full_path/vit_tiny/checkpoint_0200.pth.tar
vit_800=$full_path/vit_tiny/checkpoint_0799.pth.tar

python main_transfer.py -a mobilenetv3_large --lr 1e-2 --batch-size 256 --pretrained $mobile_large_base --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $mobile_large_base.$dataset.finetune.txt
python main_transfer.py -a mobilenetv3_large --lr 1e-2 --batch-size 256 --pretrained $mobile_large_200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $mobile_large_200.$dataset.finetune.txt
python main_transfer.py -a mobilenetv3_large --lr 1e-2 --batch-size 256 --pretrained $mobile_large_800 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $mobile_large_800.$dataset.finetune.txt

python main_transfer.py -a mobilenetv3_small --lr 1e-2 --batch-size 256 --pretrained $mobile_small_base --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $mobile_small_base.$dataset.finetune.txt
python main_transfer.py -a mobilenetv3_small --lr 1e-2 --batch-size 256 --pretrained $mobile_small_200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $mobile_small_200.$dataset.finetune.txt
python main_transfer.py -a mobilenetv3_small --lr 1e-2 --batch-size 256 --pretrained $mobile_small_800 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $mobile_small_800.$dataset.finetune.txt

python main_transfer.py -a efficientnet_b0 --lr 1e-2 --batch-size 256 --pretrained $efficient_base --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $efficient_base.$dataset.finetune.txt
python main_transfer.py -a efficientnet_b0 --lr 1e-2 --batch-size 256 --pretrained $efficient_200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $efficient_200.$dataset.finetune.txt
python main_transfer.py -a efficientnet_b0 --lr 1e-2 --batch-size 256 --pretrained $efficient_800 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $efficient_800.$dataset.finetune.txt

python main_transfer.py -a resnet18 --lr 1e-2 --batch-size 256 --pretrained $resnet_base --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $resnet_base.$dataset.finetune.txt
python main_transfer.py -a resnet18 --lr 1e-2 --batch-size 256 --pretrained $resnet_200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $resnet_200.$dataset.finetune.txt
python main_transfer.py -a resnet18 --lr 1e-2 --batch-size 256 --pretrained $resnet_800 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $resnet_800.$dataset.finetune.txt

python main_transfer.py -a vit_tiny --lr 1e-2 --batch-size 256 --pretrained $vit_base --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $vit_base.$dataset.finetune.txt
python main_transfer.py -a vit_tiny --lr 1e-2 --batch-size 256 --pretrained $vit_200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $vit_200.$dataset.finetune.txt
python main_transfer.py -a vit_tiny --lr 1e-2 --batch-size 256 --pretrained $vit_800 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 $datapath --dataset $dataset --epochs 100 --schedule 40 70 --finetune > $vit_800.$dataset.finetune.txt
