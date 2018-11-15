#!/bin/bash

for exp in exp_guided_enc_dec
do
python run-nconv-cnn.py -mode train -exp "$exp"

for epoch in {1..3}
do

echo $"Evaluating Exp: $exp , Epoch: $epoch " 

python run-nconv-cnn.py -mode eval -exp "$exp" -chkpt "$epoch"

done
python run-nconv-cnn.py -mode eval -exp "$exp" -chkpt -1 -set val 
done
exit 0


