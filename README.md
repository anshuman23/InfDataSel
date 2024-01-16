## Code for paper: “What Data Benefits My Classifier?” Enhancing Model Performance and Interpretability through Influence-Based Data Selection

### Requirements
```
numpy
sklearn
pytorch
dtreeviz
pickle
matplotlib
argparse
modAL
typing
```

### Section 4.1 (Toy Data)
- Please run ```python SEC-4-1.py``` and all experiments will run.

### Section 4.2 (Algorithmic Performance)
- Please run ```python SEC-4-2.py``` and all experiments will run.
- Results will be stored as pickle files in the folder ```sec-4-2-outputs```.
- Note that this also runs results for the MLP neural network which are actually presented in Appendix D.2 in the paper.

### Section 5.1 (Distribution Shift - Fairness)
- Please run ```python SEC-5-1.py``` and all experiments for influence based trimming will run.
- Other baselines will be later added in a separate repository.

### Section 5.2 (Fairness Poisoning Attacks)
- There are multiple datasets and attack types to choose: ``` <choice> = nraa_attack_drug, nraa_attack_compas, nraa_attack_german, raa_attack_drug, raa_attack_compas, raa_attack_german```
- Please run ```python SEC-5-2.py --dataset <choice>``` to run the experiments of your choice.

### Section 5.3 (Adaptive Evasion Attacks)
- There are multiple datasets to choose from: ```<choice> = adult, bank, celeba, nlp```
- Please run ```python SEC-5-3.py --dataset <choice>``` to run experiments on the dataset of your choice.

### Section 5.4 (Online Learning)
- There are multiple datasets to choose from: ```<choice> = adult, bank, celeba, nlp```
- Please run ```python SEC-5-4.py --dataset <choice>``` to run experiments on the dataset of your choice.

### Section 5.5 (Active Learning)
- Please run ```python SEC-5-5.py``` and all experiments will run.
- Results will be stored as pickle files in the folder ```sec-5-5-outputs```.

### Appendix D.2 (EOP Results)
- Please run ```python APPENDIX-D.py``` and all experiments will run.
- Results will be stored as pickle files in the folder ```appendix-d-outputs```.
- Note that these are the EOP results for both Logistic Regression and MLP, and the Appendix D.1 results are run as part of Section 4.2 experiments.

### Appendix F (Visualizing Trees)
- Please run ```python APPENDIX-F.py <choice>``` where ```<choice> = adult, bank, celeba, nlp```.
- Three trees will be generated for utility, fairness, and robustness.

#### Notes
- Since the paper has multiple experiments, each has its own individual code files. There is no interdependence.
- Current datasets give desirable functionality with height reduced regular decision tree regressors. If you wish to use hierarchical shrinkage, please refer to the ```imodels``` python package, it is simple to replace the sklearn tree with these.
- The file ```acs_data_generator.py``` is custom code based on the ```folktables``` package and used to generate the distribution shift datasets-- it is provided here for reference.
