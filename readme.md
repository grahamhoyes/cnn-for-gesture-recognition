## Notes
This model was designed to be trained and run on Cuda. Significant memory is required to run the model, so you may be unable to run it on your local computer. All training and predictions were done on a Nvidia Tesla P100 GPU in Google Cloud Compute.

## Training Instructions
1. Specify hyperparameters in config.json
2. Specify model in model.py
3. Run main.py to initiate training

## Generating Predictions
Predictions will be generated based off the best model specified in models.json. The relevant folder must have model.pt and model.py in it.
1. Run grade.py