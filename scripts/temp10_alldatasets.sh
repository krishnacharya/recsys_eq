# want to compare uniform synthetic with temp = 0.1
python main_run.py --data synth-uniform --prob softmax --temp 10 &
python main_run.py --data synth-skewed --prob softmax --temp 10 &
python main_run.py --data movielens-100k --prob softmax --temp 10 &