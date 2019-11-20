#!/bin/bash

#SBATCH --workdir=/storage/jastudillo/Becker-RNN/v3/
#SBATCH --job-name=javi        # Nombre del trabajo
#SBATCH --time=0-0:10:00            # Timpo limite d-hrs:min:sec
#SBATCH --mem-per-cpu=2000mb         # Memoria por proceso
#SBATCH --partition=ialab-high        # Se tiene que elegir una partición de nodos con GPU
#SBATCH --gres=gpu:1                 # Usar 2 GPUs (se pueden usar N GPUs de marca especifica de la manera --gres=gpu:marca:N)
#SBATCH --nodelist=hydra
#SBATCH --output=preprocess.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=preprocess.err          # Output de errores (opcional)

date;hostname;pwd

export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
source /storage/jastudillo/tf1.15/bin/activate
python -u ./preprocess.py > preprocess.log

