for ((i=1; i<=100; i++));
do
  python execute_command.py `python yield_command.py $i configs/tasks_123_train_config_for_32gb_gpu.json grid.json`;
done
