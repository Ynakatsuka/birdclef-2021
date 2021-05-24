#!/bin/bash
python run.py "$@"
python src/misc/evaluate.py "$@"
