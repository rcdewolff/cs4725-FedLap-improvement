$ErrorActionPreference = "Stop"

$configs = @(
    "./config/config_chameleon.yml",
    "./config/config_CiteSeer.yml",
    "./config/config_Cora.yml",
    "./config/config_Pubmed.yml",
    "./config/config_Photo.yml",
    "./config/config_Amazon-ratings.yml"
)

foreach ($configPath in $configs) {
    Write-Host "=== Running defense ablation with CONFIGPATH=$configPath ==="
    $env:CONFIGPATH = $configPath
    python src/simulations/online_phase_dp_ablation_experiment.py
    if ($LASTEXITCODE -ne 0) {
        throw "Defense ablation failed for $configPath with exit code $LASTEXITCODE."
    }
}
