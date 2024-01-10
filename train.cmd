REM 原论文设置
python train.py -I original -M freeze
python train.py -I original -M unfreeze
REM 不使用EW层
python train.py -I none

REM 一些平凡的初始化方式
python train.py -I eye -M unfreeze
python train.py -I rand -M unfreeze

REM 换用相关性矩阵
python train.py -I corr -M freeze
python train.py -I corr -M unfreeze
python train.py -I corr_softmax -M freeze
python train.py -I corr_softmax -M unfreeze
python train.py -I corr_log_softmax -M freeze
python train.py -I corr_log_softmax -M unfreeze
