# SQL (MariaDB) usage for Optuna on the cluster – step by step (beginner friendly)

This guide shows how to run a MariaDB server on your cluster and use it as Optuna storage for parallel/distributed hyperparameter optimization. It assumes Docker is not available but Apptainer (Singularity) is.

The workflow below is exactly what worked for our setup.

## 0) Why we need a DB (and why not SQLite)
- Optuna needs a shared storage so multiple workers (SLURM array tasks) can coordinate trials.
- SQLite works for single-process runs but is not reliable for many parallel writers on a shared filesystem.
- MariaDB (or PostgreSQL) is a robust shared DB you can host on one node while your array jobs run elsewhere.

## 1) One-time Python dependencies
Install the MySQL/MariaDB driver for Optuna in your environment:
```bash
pip install "optuna[mysql]" PyMySQL
```

## 2) Start a MariaDB server as a SLURM job
We use the provided script `slurm_scripts/run_mariadb.sh`. It runs a MariaDB container via Apptainer and mounts persistent data/config directories in your home by default.

- By default, the script stores everything under:
  - `$HOME/mariadb_service/mariadb/{data,conf,run,mariadb.sif}`
- If/when your project volume becomes available, you can place large files under that path by exporting:
  - `export PROJECT_ROOT_OVERRIDE=/vol/projects/BIFO/genomenet/yhan`
  - Then the script will use `${PROJECT_ROOT_OVERRIDE}/mariadb/...` instead of `$HOME/...`.

Before submitting the job, set strong secrets (only needed the first time you initialize the DB data dir):
```bash
export MARIADB_ROOT_PASSWORD='492568@HYC'
export MARIADB_PASSWORD='492568@HYC'
```
Submit the job (on a CPU partition with a long walltime):
```bash
sbatch slurm_scripts/run_mariadb.sh
```
Notes:
- The first run initializes the data directory and creates the database `optuna` and user `optuna_user`.
- Subsequent restarts reuse the same data; you do not need to re-export the passwords unless you’re reinitializing from scratch.

## 3) Confirm the DB is running and note the host
Check the job’s log, e.g. `slurm_output/mariadb_<JOBID>.log`. You should see lines like:
```
Node hostname: bioinf029-deploy-hpc
... ready for connections ... port: 3306
```
Use that hostname in your connection URL.

## 4) Test SQL connectivity from the login node
```bash
mysql -h bioinf029-deploy-hpc -P 3306 -u optuna_user -p -e "SHOW DATABASES;"
```
Enter your user password (`$MARIADB_PASSWORD`). You should see `optuna` listed.

If you get “Host 'X' is not allowed to connect”: your user may not have a '%' host entry. See Troubleshooting below (“E1”).

## 5) Set Optuna storage URL (passwords with special chars)
If your DB password has characters like `@`, `:`, `/`, `#`, `?`, `&`, you must URL-encode it. Easiest is this one-liner:
```bash
export OPTUNA_STORAGE="mysql+pymysql://optuna_user:$(python -c 'import os,urllib.parse;print(urllib.parse.quote_plus(os.environ["MARIADB_PASSWORD"]))')@bioinf009-deploy-hpc:3306/optuna"
```
Examples of manual encoding: `@`→`%40`, `:`→`%3A`, `/`→`%2F`, `#`→`%23`, `?`→`%3F`, `&`→`%26`.

## 6) Quick Optuna smoke test (single process)
```bash
python - <<'PY'
import os, optuna
storage = os.environ["OPTUNA_STORAGE"]
study = optuna.create_study(study_name="tinycnn_smoke", storage=storage, load_if_exists=True, direction="maximize")
study.optimize(lambda t: 1 - (t.suggest_float("x", -1, 1)**2), n_trials=5)
print("Trials:", len(study.trials))
PY
```
If this prints a small number of trials, your DB connection works.

## 7) Launch distributed HPO via SLURM
Our SLURM script `slurm_scripts/run_toy_slurm.sh` already supports HPO workers via `RUN_HPO=1`. Each array task runs one Optuna worker (keep `n_jobs=1` in code).

Example:
```bash
export OPTUNA_STORAGE  # ensure this is set as in step 5
export RUN_HPO=1 STUDY_NAME=tinycnn_dev N_TRIALS=400 HPO_MODE=standard
sbatch --array=1-20 slurm_scripts/run_toy_slurm.sh
```
- Keep the DB job running (it’s on a CPU node with long walltime).
- You can add more trials later by reusing the same `STUDY_NAME` and `OPTUNA_STORAGE`.

## 8) Evaluate/plot results
After training completes, aggregate and plot with the separate evaluation script:
```bash
python synthetic/eval_toy.py --output_dir <OUTPUT_DIR_FROM_RUNS>
```
This writes combined CSVs and PDFs under `<output_dir>/plots`.

---

## Troubleshooting (common errors)

### E1) ERROR 1130: Host 'A.B.C.D' is not allowed to connect
Cause: MariaDB user was created with host `localhost` only. Solution: add `'%'` host entry for the user.

Run this inside the existing DB job allocation (replace `JOBID`):
```bash
srun --jobid=JOBID --time=10 --pty bash
# now on the DB node
apptainer exec \
  --bind "$HOME/mariadb_service/mariadb/data:/var/lib/mysql" \
  --bind "$HOME/mariadb_service/mariadb/conf:/etc/mysql/conf.d" \
  --bind "$HOME/mariadb_service/mariadb/run:/run/mysqld" \
  "$HOME/mariadb_service/mariadb/mariadb.sif" \
  mariadb -uroot -p -S /run/mysqld/mysqld.sock
# enter the root password you used at initialization
CREATE USER IF NOT EXISTS 'optuna_user'@'%' IDENTIFIED BY 'YOUR_USER_PASSWORD';
GRANT ALL PRIVILEGES ON optuna.* TO 'optuna_user'@'%';
FLUSH PRIVILEGES;
exit
```
Retry from login node:
```bash
mysql -h <DB_HOST> -P 3306 -u optuna_user -p -e "SHOW DATABASES;"
```

### E2) URL includes '@' or other special chars in password → connection fails
Encode the password in the URL (see step 5). The one-liner avoids mistakes:
```bash
export OPTUNA_STORAGE="mysql+pymysql://optuna_user:$(python -c 'import os,urllib.parse;print(urllib.parse.quote_plus(os.environ["MARIADB_PASSWORD"]))')@<DB_HOST>:3306/optuna"
```

### E3) Access denied for user 'root'@'localhost' when using the socket
- The root password is set only at first initialization. If you forgot it and need to change it, the simplest path is to stop the DB job, remove the data directory, re-export passwords, and resubmit so the entrypoint reinitializes:
```bash
scancel <DB_JOBID>
rm -rf $HOME/mariadb_service/mariadb/data
export MARIADB_ROOT_PASSWORD='NewStrongRoot#Pass1'
export MARIADB_PASSWORD='NewStrongUser#Pass1'
sbatch slurm_scripts/run_mariadb.sh
```
Note: The Optuna DB is small; reinitialization is safe (you’ll lose trial history).

### E4) "Bind on unix socket: Read-only file system"
We fixed this by binding a writable run directory and pointing MariaDB to that socket path. The provided `run_mariadb.sh` already does this:
- Binds `$HOME/mariadb_service/mariadb/run` to `/run/mysqld`
- Sets `socket=/run/mysqld/mysqld.sock` in my.cnf

### E5) Permission denied writing to /vol/ paths
If you can’t write under `/vol/projects/BIFO/genomenet/...`, use `$HOME` as the project root for now (default). Later, when permissions are fixed, set:
```bash
export PROJECT_ROOT_OVERRIDE=/vol/projects/BIFO/genomenet/yhan
```
Then resubmit the DB job to move image/data/config to that path.

### E6) DB job time limit and restarts
- Prefer a CPU/general partition with long QoS (e.g., 7 days) for the DB job.
- If the job ends, simply resubmit it pointing at the same data directory; the DB will come back with the same users and tables.

---

## FAQ
- How big will the DB be?
  - Optuna stores trial metadata; size is small: roughly 1–10 KB per trial. Even 100k trials is typically well under 1 GB.
- Can I switch to PostgreSQL later?
  - Yes. Install `pip install "optuna[postgresql]" psycopg2-binary` and use a `postgresql+psycopg2://...` URL. The rest of the flow is similar.
- Do I need to keep exporting passwords every time?
  - Only when initializing a fresh data directory. For normal restarts with an existing data dir, the DB uses the stored credentials.

---

## Quick reference (copy/paste)

Start DB (first time):
```bash
export MARIADB_ROOT_PASSWORD='StrongRoot#Pass1'
export MARIADB_PASSWORD='StrongUser#Pass1'
sbatch slurm_scripts/run_mariadb.sh
```

Get host from log, then test:
```bash
mysql -h <DB_HOST> -P 3306 -u optuna_user -p -e "SHOW DATABASES;"
```

Set storage with encoded password:
```bash
export OPTUNA_STORAGE="mysql+pymysql://optuna_user:$(python -c 'import os,urllib.parse;print(urllib.parse.quote_plus(os.environ["MARIADB_PASSWORD"]))')@<DB_HOST>:3306/optuna"
```

Run HPO workers:
```bash
export RUN_HPO=1 STUDY_NAME=tinycnn_dev N_TRIALS=400 HPO_MODE=standard
sbatch --array=1-20 slurm_scripts/run_toy_slurm.sh
```

Aggregate:
```bash
python synthetic/eval_toy.py --output_dir <OUTPUT_DIR_FROM_RUNS>
```
