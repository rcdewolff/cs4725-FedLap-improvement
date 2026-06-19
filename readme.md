# CS4725 FedLap improvements

This repo was forked from FedLap to try to improve it for a university project (the CS4725 course at TU Delft).

How to set up your Python environment: Create a virtual environment with Python 3.10, activate it, and run `pip install -r requirements.txt`.

## DLG attack

The main relevant code for our paper (after some failed experimentation attempts) lives in the `src/simulations/gradient_inversion_attack.py` and `src/simulations/online_phase_dlg_experiment.py`.

You can run the experiments on one dataset (set path in `.env`) by running (**Warning: Can take 30 minutes to 3 hours to run, depending on dataset**):

```bash
python src/simulations/online_phase_dlg_experiment.py
```

You can also run all experiments sequentially by running (may need to adjust path to virtual environment activation script):

```bash
bash run_experiments.sh
```

Final results and data_analysis related code can be found in the `final_results/` folder.

### Ablation study DLG attack

To run the 100-epoch defense ablation (no defense, clipping only, and
clipping plus noise), use:

```bash
python src/simulations/online_phase_dp_ablation_experiment.py
```

The ablation logs the duration of every epoch and writes both a run-level CSV
and per-client/per-epoch gradient-norm and clipping telemetry under the
dataset's `privacy_attack/dlg_ablation` result directory.

Run the ablation sequentially for all six report datasets with:

```bash
powershell -ExecutionPolicy Bypass -File .\run_ablation_experiments.ps1
```



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

