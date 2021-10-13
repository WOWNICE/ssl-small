echo "=> EVAL instance discrimination accuracy"
CUDA_VISIBLE_DEVICES=`expr $4 + 0` python -m eval.instdisc -a $1 -b $3 -j 8 --pretrained $2 /raid/ssl-positive-eval/auged-val -s 50 --mode d-sd --dataset-mode normal &
echo "=> EVAL alignment, intra-class alignment, and uniformity"
CUDA_VISIBLE_DEVICES=`expr $4 + 1` python -m eval.stats-dist -a $1 -b $3 -j 8 --pretrained $2 /raid/ssl-positive-eval/auged-val -s 50 &
wait
