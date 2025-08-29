#!/bin/bash
#SBATCH --job-name=mariadb_service
#SBATCH --output=slurm_output/mariadb_%j.log
#SBATCH --error=slurm_output/mariadb_%j.log
#SBATCH --time=14-00:00:00
#SBATCH --partition=cpu   # Change to your CPU/general partition if available
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --qos=verylong

# This job starts a MariaDB server on the allocated node using Apptainer.
# Large files (image, data) are placed under /vol/projects/BIFO/genomenet/.
#
# Usage:
#   export MARIADB_ROOT_PASSWORD='StrongRoot#Pass1'
#   export MARIADB_PASSWORD='StrongUser#Pass1'
#   sbatch slurm_scripts/run_mariadb.sh
#
# After the job starts, check the log for the assigned hostname, e.g. c1234.
# Then set for Optuna in your shell:
#   export OPTUNA_STORAGE="mysql+pymysql://optuna_user:${MARIADB_PASSWORD}@c1234:3306/optuna"
# and run your HPO array.

module load apptainer 2>/dev/null || true
mkdir -p slurm_output

# Project paths (large files live here)
PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-$HOME/mariadb_service}"
MDB_ROOT="${PROJECT_ROOT}/mariadb"
MDB_IMAGE="${MDB_ROOT}/mariadb.sif"
MDB_DATA="${MDB_ROOT}/data"
MDB_CONF_DIR="${MDB_ROOT}/conf"
MDB_RUN_DIR="${MDB_ROOT}/run"

mkdir -p "${MDB_DATA}" "${MDB_CONF_DIR}" "${MDB_RUN_DIR}"
chmod 0777 "${MDB_RUN_DIR}"

# Create a minimal config
cat > "${MDB_CONF_DIR}/my.cnf" <<'CFG'
[mysqld]
bind-address=0.0.0.0
character-set-server=utf8mb4
collation-server=utf8mb4_unicode_ci
innodb_flush_log_at_trx_commit=2
socket=/run/mysqld/mysqld.sock
port=3306

[client]
socket=/run/mysqld/mysqld.sock
CFG

# Pull image once
if [ ! -f "${MDB_IMAGE}" ]; then
  echo "Pulling MariaDB image to ${MDB_IMAGE} ..."
  apptainer pull --disable-cache "${MDB_IMAGE}" docker://mariadb:11 || {
    echo "Failed to pull MariaDB image"; exit 1; }
fi

# Secrets (must be set by user before sbatch); set sane defaults if missing
MARIADB_ROOT_PASSWORD=${MARIADB_ROOT_PASSWORD:-ChangeMeRoot!}
MARIADB_DATABASE="optuna"
MARIADB_USER="optuna_user"
MARIADB_PASSWORD=${MARIADB_PASSWORD:-ChangeMeUser!}

echo "Node hostname: $(hostname -f 2>/dev/null || hostname)"
echo "MariaDB data dir: ${MDB_DATA}"
echo "MariaDB conf dir: ${MDB_CONF_DIR}"
echo "Use this host with OPTUNA_STORAGE once initialized (port 3306)."

# Start the server via docker entrypoint to initialize if needed
apptainer exec \
  --bind "${MDB_DATA}:/var/lib/mysql" \
  --bind "${MDB_CONF_DIR}:/etc/mysql/conf.d" \
  --bind "${MDB_RUN_DIR}:/run/mysqld" \
  "${MDB_IMAGE}" \
  bash -lc "export MARIADB_ROOT_PASSWORD='${MARIADB_ROOT_PASSWORD}'; \
            export MARIADB_DATABASE='${MARIADB_DATABASE}'; \
            export MARIADB_USER='${MARIADB_USER}'; \
            export MARIADB_PASSWORD='${MARIADB_PASSWORD}'; \
            # Prefer mariadbd (MariaDB >=10.5) over legacy mysqld
            if command -v docker-entrypoint.sh >/dev/null 2>&1; then \
              docker-entrypoint.sh mariadbd; \
            else \
              # Fallback: run mysqld_safe (assumes datadir already initialized)
              if command -v mariadbd >/dev/null 2>&1; then \
                mariadbd --datadir=/var/lib/mysql --socket=/run/mysqld/mysqld.sock --bind-address=0.0.0.0; \
              else \
                mysqld_safe --user=mysql; \
              fi; \
            fi"


