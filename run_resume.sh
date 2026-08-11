#!/bin/bash
cd ~/G-Retriever
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
~/miniconda3/envs/gretriever/bin/python -m resume_webqsp 2>&1 | tee ~/resume.log
