import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--test_file',type=str,default='AIFirst_test_problem.txt')
args=parser.parse_args()

with open(args.test_file,'r') as file:
	i=0
	for line in file:
		i+=1
print(i-1,end='')