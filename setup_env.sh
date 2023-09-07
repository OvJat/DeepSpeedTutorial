#!/usr/bin/env bash

# reload /etc/profile
source  /etc/profile

# if zsh
[ -f "$HOME/.zshrc" ] && source  "$HOME/.zshrc"
# if bash
[ -f "$HOME/.bashrc" ] && source  "$HOME/.bashrc"

# setup anaconda
source "$HOME/.miniconda/etc/profile.d/conda.sh"
conda activate Torch
which deepspeed

# other environment set here.