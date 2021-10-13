sh tools/all_metircs.sh resnet50 /raid/ssl-checkpoints/resnet50_tau0.2_neg65536/checkpoint_0199.pth.tar 512 0 &
sh tools/all_metircs.sh resnet50 /raid/ssl-checkpoints/resnet50_tau0.2_neg16384/checkpoint_0199.pth.tar 512 2 &
sh tools/all_metircs.sh resnet50 /raid/ssl-checkpoints/resnet50_tau0.2_neg4096/checkpoint_0199.pth.tar 512 4 &
sh tools/all_metircs.sh resnet50 /raid/ssl-checkpoints/resnet50_tau0.2_neg1024/checkpoint_0199.pth.tar 512 6 &
wait
sh tools/all_metircs.sh mobilenetv3_large /raid/ssl-checkpoints/mobilev3_large_tau0.2_neg16384/checkpoint_0199.pth.tar 1024 2 &
sh tools/all_metircs.sh mobilenetv3_large /raid/ssl-checkpoints/mobilev3_large_tau0.2_neg4096/checkpoint_0199.pth.tar 1024 4 &
sh tools/all_metircs.sh mobilenetv3_large /raid/ssl-checkpoints/mobilev3_large_tau0.2_neg1024/checkpoint_0199.pth.tar 1024 6 &
wait
sh tools/all_metircs.sh mobilenetv3_large /raid/ssl-checkpoints/mobilev3_large_tau0.05_neg16384/checkpoint_0199.pth.tar 1024 2 &
sh tools/all_metircs.sh mobilenetv3_large /raid/ssl-checkpoints/mobilev3_large_tau0.05_neg4096/checkpoint_0199.pth.tar 1024 4 &
sh tools/all_metircs.sh mobilenetv3_large /raid/ssl-checkpoints/mobilev3_large_tau0.05_neg1024/checkpoint_0199.pth.tar 1024 6 &
wait
