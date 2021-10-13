sleep 12h
python main_lincls.py -a mobilenetv3_large --lr 30.0 --batch-size 256 --pretrained /raid/ssl-checkpoints/mobilev3_large_tau0.05_neg16384/checkpoint_0199.pth.tar --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 /raid/imagenet
