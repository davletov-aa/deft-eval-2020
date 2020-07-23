train_config_json_path=$1
grid_json_path=$2
default_test_file=$3

for ((i=1; i<=100; i++));
do
  python execute_command.py `python yield_command.py $i $train_config_json_path $grid_json_path $default_test_file`;
done
