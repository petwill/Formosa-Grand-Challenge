#!/bin/zsh

#change filename and parse context if necessary

out_dir=data/cleaned_data

if [ -d $out_dir ]
then
  rm -r $out_dir
fi
mkdir $out_dir

cnt=0

for dir in ./data/training_data/subtitle_with_TC/*; do
  echo $dir
  for file in $dir/*; do
    let cnt=cnt+1
    sed -E 's/^.*[\t]+//' $file > $out_dir/$cnt.txt
  done
done

for dir in ./data/training_data/subtitle_no_TC/*; do
  echo $dir
  for file in $dir/*; do
    if [ -d $file ]
    then
      #for file in $file/*; do
      #  let cnt=cnt+1
      #  sed -E 's/^.*：//' $file > $out_dir/$cnt.txt
      #done
      :
    else
      let cnt=cnt+1
      cp $file $out_dir/$cnt.txt
    fi
  done
done
echo "total number of files = "$cnt

# get jieba dictionary
cd data && wget https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big  
