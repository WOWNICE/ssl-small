ARCH=$1
CKPT_DIR=$2
BS=$3

sh tools/complete_eval.sh $ARCH $CKPT_DIR/checkpoint_0000.pth.tar $BS 0 &
sh tools/complete_eval.sh $ARCH $CKPT_DIR/checkpoint_0001.pth.tar $BS 4 &
wait
sh tools/complete_eval.sh $ARCH $CKPT_DIR/checkpoint_0009.pth.tar $BS 0 &
sh tools/complete_eval.sh $ARCH $CKPT_DIR/checkpoint_0020.pth.tar $BS 4 &
wait
sh tools/complete_eval.sh $ARCH $CKPT_DIR/checkpoint_0050.pth.tar $BS 0 &
sh tools/complete_eval.sh $ARCH $CKPT_DIR/checkpoint_0100.pth.tar $BS 4 &
wait
sh tools/complete_eval.sh $ARCH $CKPT_DIR/checkpoint_0150.pth.tar $BS 0 &
sh tools/complete_eval.sh $ARCH $CKPT_DIR/checkpoint_0199.pth.tar $BS 4 &
wait
