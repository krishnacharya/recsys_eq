# want to compare uniform synthetic with temp = 100
python main_run.py --data synth-uniform --prob softmax --temp 100 &
python main_run.py --data synth-skewed --prob softmax --temp 100 &
python main_run.py --data movielens-100k --prob softmax --temp 100 &