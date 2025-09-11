# Train model on full dataset to extract training dynamics
# python train.py --dataset cifar10 --gpuid 1 --epochs 200 --lr 0.1 --network resnet18 --batch-size 256 --task-name all-data --base-dir ./data-model/cifar10

# Get importance scores and sample embeddings
# python generate_importance_score.py --gpuid 1 --dataset cifar10 --base-dir ./data-model/cifar10 --task-name all-data --feature

# Select samples using D2 pruning and train ResNet 18 on the selected coreset
#!/bin/bash

# Set the constant parameters that don't change
# N_NEIGHBOR=10
# GAMMA=0.9

# Define the list of Corset Ratio values to iterate over
CORESET_RATIOS=(0.1 0.2 0.3 0.5 0.7)

# Loop through each ratio in the list
for RATIO in "${CORESET_RATIOS[@]}"
do
  # Announce which ratio is being used for the current run
  echo "================================================="
  echo "STARTING RUN WITH CORESET_RATIO = $RATIO"
  echo "================================================="

  # Execute the python training command
  # The $RATIO variable is used in the --task-name and --coreset-ratio arguments
  python train.py --dataset cifar10 --gpuid 1 --iterations 40000 \
    --task-name class-lb-graph-n=$N_NEIGHBOR-g=$GAMMA-$RATIO \
    --base-dir ./data-model/cifar10/moderate/ \
    --coreset --coreset-mode moderate --budget-mode uniform --sampling-mode graph \
    --data-score-path ./data-model/cifar10/all-data/data-score-all-data.pickle \
    --feature-path ./data-model/cifar10/all-data/train-features-all-data.npy \
    --coreset-key forgetting \
    --coreset-ratio $RATIO \
    --mis-ratio 0.0 \
    # --n-neighbor $N_NEIGHBOR \
    # --gamma $GAMMA \
    # --stratas 200 \
    # --graph-mode sum \
    # --graph-sampling-mode weighted
  
  echo "FINISHED RUN WITH CORESET_RATIO = $RATIO"
done

echo "All runs completed!"

