#!/usr/bin/env bash


input_path=$1
output_path=$2
test_num=$(python3 find_test_num.py --test_file=$input_path)
echo "test_num= " $test_num
read -n1 -r -p "Press any key to continue..." key
echo "Running Vanilla"

cp $input_path ./RNN-encoding-master/data/AIFirst_test_problem.txt
cd ./RNN-encoding-master/
python3 main.py test --load_dir=./save/RNN_vanilla_128/ \
                    --result_output=./vanilla_pred.csv \
                    --weights_path=./vanilla_weights.pickle
cd ../SMN_Daikon

echo "Running SMN"
cp $input_path ./SMN_Daikon/data/test.txt
bash ./go.sh $test_num pretest test

echo "Running avg_wordvec"
python3 avg_wordvec.py --problem_path=$input_path \
                        --vanilla_path=./RNN-encoding-master/vanilla_weights.pickle \
                        --smn_path=./SMN_Daikon/SMN_weights.pickle \
                        --ans_path=$output_path \
                        --test_num=$test_num
