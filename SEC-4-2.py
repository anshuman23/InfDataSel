import os

os.system('python SEC-4-2.py --dataset adult --metric dp --seed 42')
os.system('python SEC-4-2.py --dataset bank --metric dp --seed 42')
os.system('python SEC-4-2.py --dataset celeba --metric dp --seed 42')
os.system('python SEC-4-2.py --dataset nlp --metric dp --seed 42')


os.system('python SEC-4-2.py --dataset adult --metric dp --seed 42 --model_type nn')
os.system('python SEC-4-2.py --dataset bank --metric dp --seed 42 --model_type nn')
os.system('python SEC-4-2.py --dataset celeba --metric dp --seed 42 --model_type nn')
os.system('python SEC-4-2.py --dataset nlp --metric dp --seed 42 --model_type nn')
