#!/bin/bash

for file in simulations/*.json; do python wruamodel.py "$file"; done
