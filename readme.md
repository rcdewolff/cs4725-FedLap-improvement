# CS4725 FedLap improvements

This repo was forked from FedLap to try to improve it for a university project (the CS4725 course at TU Delft).

How to set up your Python environment: Create a virtual environment with Python 3.10, activate it, and run `pip install -r requirements.txt`.

The main relevant code for our paper (after some failed experimentation attempts) lives in the `src/simulations/gradient_inversion_attack.py` and `src/simulations/online_phase_dlg_experiment.py`.

You can run the experiments by running (**Warning: Can take ~1h to run**):

:

```bash
python src/simulations/online_phase_dlg_experiments.py
```

The results for this initial experiment can be seen in the `./example_experiment_results/` folder.

# (ORIGINAL README) FedLap
## To run the code
1. Clone the repository to your *local_directory* with:
    >git clone https://github.com/JavadAliakbari/FedLap.git /the/local/directory/  
    >cd /the/local/directory/

2. Run the following lines to download required packages:  
    >conda env create -f ./FedLap.yml  
    >conda activate FedLap  

3. You can choose the config file for each dataset by changing the **CONFIGPATH** in .env.  
You can also change hyper-parameters in **~/config/config_*dataset_name*.py** according to different testing scenarios.  
*dataset_name* can be **Cora**, **CiteSeer**, **PubMed**, **chameleon**, **Photo**, **Amazon-ratings**;

4. Run the main file with  
    > **python src/main.py**  

    You can access results in **results/dataset_name/**
5. You can also run the simulation file with  
    > **python src/simulations/simulation.py**

    You can access results in **results/Simulation/dataset_name/**

