set -e

source .venv/Scripts/activate

configs=(
  #"./config/config_Cora.yml"
  #"./config/config_CiteSeer.yml"
  "./config/config_Chameleon.yml"
  "./config/config_Pubmed.yml"
  "./config/config_Photo.yml"
  "./config/config_Amazon-ratings.yml"
)

for cfg in "${configs[@]}"; do
  echo "=== Running with CONFIGPATH=$cfg ==="
  sed -i "s|^CONFIGPATH=.*|CONFIGPATH=$cfg|" .env
  python src/simulations/online_phase_dlg_experiment.py
done