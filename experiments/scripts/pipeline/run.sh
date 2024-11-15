#!/bin/bash

: '
This Bash script is responsible for running a multi-stage pipeline for 
building a hybrid recommendation model. 
'

set -e

# Define delay duration in seconds
DELAY_DURATION=0

run_preprocessing() {
  python3 -m scripts.components.preprocessing
  sleep $DELAY_DURATION
}

run_matrix_building() {
  python3 -m scripts.components.matrix_builder
  sleep $DELAY_DURATION
}

run_features_generation() {
  echo "$FEATURES_STAGE" | tr ',' '\n' | while read -r stage; do
    python3 -m scripts.components.features_generator --stage $stage
    sleep $DELAY_DURATION
  done
}

run_als() {
  echo "$ALS_STAGE" | tr ',' '\n' | while read -r stage; do
    python3 -m scripts.components.als --stage $stage
    sleep $DELAY_DURATION
  done
}

run_bpr() {
  echo "$BPR_STAGE" | tr ',' '\n' | while read -r stage; do
    python3 -m scripts.components.bpr --stage $stage
    sleep $DELAY_DURATION
  done
}

run_item2item() {
  echo "$ITEM2ITEM_STAGE" | tr ',' '\n' | while read -r stage; do
    python3 -m scripts.components.item2item --stage $stage
    sleep $DELAY_DURATION
  done
}

run_top_items() {
  echo "$TOP_ITEMS_STAGE" | tr ',' '\n' | while read -r stage; do
    python3 -m scripts.components.top_items --stage $stage
    sleep $DELAY_DURATION
  done
}

run_ensemble() {
  echo "$ENSEMBLE_STAGE" | tr ',' '\n' | while read -r stage; do
    python3 -m scripts.components.ensemble --stage $stage
    sleep $DELAY_DURATION
  done
}

# Default stages to run
RUN_PREPROCESSING=false
RUN_EDA=false
RUN_MATRIX_BUILDING=false
RUN_FEATURES_GENERATION=false
RUN_ALS=false
RUN_BPR=false
RUN_ITEM2ITEM=false
RUN_TOP_ITEMS=false
RUN_ENSEMBLE=false

# Default sub-stages to run
FEATURES_STAGE="1,2,3"
ALS_STAGE="1,2,3,4,5,6"
BPR_STAGE="1,2,3,4,5,6"
ITEM2ITEM_STAGE="1,2,3,4,5,6"
TOP_ITEMS_STAGE="1,2,3,4,5,6"
ENSEMBLE_STAGE="1,2,3,4,5,6"

# Parse command-line options
while getopts "p:m:f:a:b:i:t:e:" opt; do
  case $opt in
    p) RUN_PREPROCESSING=$OPTARG ;;
    m) RUN_MATRIX_BUILDING=$OPTARG ;;
    f) RUN_FEATURES_GENERATION=true; FEATURES_STAGE=$OPTARG ;;
    a) RUN_ALS=true; ALS_STAGE=$OPTARG ;;
    b) RUN_BPR=true; BPR_STAGE=$OPTARG ;;
    i) RUN_ITEM2ITEM=true; ITEM2ITEM_STAGE=$OPTARG ;;
    t) RUN_TOP_ITEMS=true; TOP_ITEMS_STAGE=$OPTARG ;;
    e) RUN_ENSEMBLE=true; ENSEMBLE_STAGE=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# Execute stages based on user input
$RUN_PREPROCESSING && run_preprocessing
$RUN_MATRIX_BUILDING && run_matrix_building
$RUN_FEATURES_GENERATION && run_features_generation
$RUN_ALS && run_als
$RUN_BPR && run_bpr
$RUN_ITEM2ITEM && run_item2item
$RUN_TOP_ITEMS && run_top_items
$RUN_ENSEMBLE && run_ensemble
