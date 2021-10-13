echo "=> pretext task on VAL..."
CUDA_VISIBLE_DEVICES=`expr $4 + 0` python -m eval.instdisc -a $1 -b $3 -j 8 --pretrained $2 /raid/ssl-positive-eval/auged-super-hard -s 50 --mode sd-sd --dataset-mode normal &
CUDA_VISIBLE_DEVICES=`expr $4 + 1` python -m eval.instdisc -a $1 -b $3 -j 8 --pretrained $2 /raid/ssl-positive-eval/auged-hard -s 50 --mode sd-sd --dataset-mode normal &
CUDA_VISIBLE_DEVICES=`expr $4 + 2` python -m eval.instdisc -a $1 -b $3 -j 8 --pretrained $2 /raid/ssl-positive-eval/auged-easy -s 50 --mode sd-sd --dataset-mode normal &
CUDA_VISIBLE_DEVICES=`expr $4 + 3` python -m eval.instdisc -a $1 -b $3 -j 8 --pretrained $2 /raid/ssl-positive-eval/auged-val -s 50 --mode sd-sd --dataset-mode normal &
wait
