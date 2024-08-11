from pathlib import Path

import yaml

# with open("globals.yml", "r") as stream:
#     data = yaml.safe_load(stream)

data =   {
    "RESULTS_DIR": "results",

    # Data files
    "DATA_DIR": "/netcache/wch/rome/data",
    "STATS_DIR": "/netcache/wch/rome/data/stats",
    "KV_DIR": "/share/projects/rewriting-knowledge/kvs",

    # Hyperparameters
    "HPARAMS_DIR": "hparams",

    # Remote URLs
    "REMOTE_ROOT_URL": "https://memit.baulab.info"
}

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
