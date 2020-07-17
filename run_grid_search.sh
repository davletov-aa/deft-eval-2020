for ((i=1; i<=100; i++));
do
  python execute_command.py `python yield_command.py $i $1 $2`;
done
