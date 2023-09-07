#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
SCRIPT_NAME=$(basename "$0")
cd "${SCRIPT_PATH}"

# prepare log file
mkdir -p logs
LOG_DATE="$(date +'%Y%m%d')"
LOG_LINK="logs/${SCRIPT_NAME}.log"
LOG_FILE="logs/${SCRIPT_NAME}.log-${LOG_DATE}"
exec &>>"$LOG_FILE"

(
flock -n 99 || exit 1
date +'{{{ %Y%m%d %H%M%S'

# update symbol link
if [ -L "$LOG_LINK" -o ! -e "$LOG_LINK" ];then
   [ -L "$LOG_LINK" ] && unlink "$LOG_LINK"
   ln -s "$(basename $LOG_FILE)" "$LOG_LINK"
fi

# compress/delete log file
find logs -type f -iname "${SCRIPT_NAME}.log-????????"    -mtime +3  -exec gzip -9v "{}" \;
find logs -type f -iname "${SCRIPT_NAME}.log-????????.gz" -mtime +30 -exec rm -v    "{}" \;

# setup anaconda
source "$HOME/.miniconda/etc/profile.d/conda.sh"
conda activate Torch
which deepspeed

# !!! importance
export PDSH_RCMD_TYPE=ssh

deepspeed \
    --launcher_args "source ${PWD}/setup_env.sh" \
    --hostfile hostfile \
    deepspeed_script.py \
    --deepspeed \
    --deepspeed_config "$PWD/deepspeed_config.json"

date +'}}} %Y%m%d %H%M%S'
) 99>"${SCRIPT_NAME}.lock"
