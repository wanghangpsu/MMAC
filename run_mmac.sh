echo ----------------------------start training ------------------------------

for((integer = 0; integer <= 9; integer ++))
do
  foo2="python mmac.py --attack_dir attack$integer --model_dir model$integer"
  $foo2
done

