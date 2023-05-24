import os

os.system('python APPENDIX-D-HELPER.py --dataset adult --metric eop --seed 42')
os.system('python APPENDIX-D-HELPER.py --dataset bank --metric eop --seed 42')
os.system('python APPENDIX-D-HELPER.py --dataset celeba --metric eop --seed 42')
os.system('python APPENDIX-D-HELPER.py --dataset nlp --metric eop --seed 42')


os.system('python APPENDIX-D-HELPER.py --dataset adult --metric eop --seed 42 --model_type nn')
os.system('python APPENDIX-D-HELPER.py --dataset bank --metric eop --seed 42 --model_type nn')
os.system('python APPENDIX-D-HELPER.py --dataset celeba --metric eop --seed 42 --model_type nn')
os.system('python APPENDIX-D-HELPER.py --dataset nlp --metric eop --seed 42 --model_type nn')
