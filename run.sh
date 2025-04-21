## finetune only D
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet50 --gpu 0\
                            --lr 0.001 --batch-size 128 \
                            --nlayer "bases_l1" --m 9

## finetune \beta and D1
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet50 --gpu 0\
                            --lr 0.001 --batch-size 128 \
                            --nlayer "bases_l2" --m 9 --m1 3

## finetune Dc
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet50 --gpu 0\
                            --lr 0.001 --batch-size 128 \
                            --nlayer "coeff_l1" --m 9 --kc 4

## finetune \beta, D1 and Dc
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet50 --gpu 0\
                            --lr 0.001 --batch-size 128 \
                            --nlayer "coeff_bases_l1" --m 9 --m1 4 --kc 4