#!/usr/bin/env python3
import os

# Lista dei comandi da eseguire
commands = [
    "echo 'Compiling project...'",
    "export mkPrefix=/u/sw/ && source ${mkPrefix}/etc/profile && module load gcc-glibc",
    "module add dealii",
    "cd build && cmake .. && make",
    # aggiungi altri comandi qui
]

# Esecuzione dei comandi
for cmd in commands:
    print(f"\n>>> Eseguo: {cmd}")
    os.system(cmd)
