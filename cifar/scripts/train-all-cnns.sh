(export CUDA_VISIBLE_DEVICES=0 && python train.py --quality 1 > logs/quality_1.log)&
(export CUDA_VISIBLE_DEVICES=1 && python train.py --quality 5 > logs/qualtiy_5.log)&
(export CUDA_VISIBLE_DEVICES=2 && python train.py --quality 10 > logs/quality_10.log)&
(export CUDA_VISIBLE_DEVICES=3 && python train.py --quality 15 > logs/quality_15.log)&
(export CUDA_VISIBLE_DEVICES=4 && python train.py --quality 20 > logs/quality_20.log)&
(export CUDA_VISIBLE_DEVICES=5 && python train.py --quality 25 > logs/quality_25.log)&
(export CUDA_VISIBLE_DEVICES=6 && python train.py --quality 50 > logs/quality_50.log)&
(export CUDA_VISIBLE_DEVICES=7 && python train.py --quality 75 > logs/quality_75.log)&
(export CUDA_VISIBLE_DEVICES=7 && python train.py --quality 100 > logs/quality_100.log)&
