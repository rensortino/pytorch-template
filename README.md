# pytorch-template
A PyTorch template inspired by PyTorch Lightning for simplifying neural network code writing, with a ready-to-use folder structure

## Structure

Each model should have its own trainer and evaluator. Modules and utils can be shared 
### Trainers
Each trainer should inherit from the BaseTrainer class. Multiple layers of inheritance for evolving versions of a model are not suggested.
The Trainer class has the following responsibilities:

- Run training logic
- Run validation logic
- Log losses and other metrics
- Save last and best checkpoints
- Define optimizer and scheduler (?)
- Visualize results (each model / task visualizes results in different ways)

### Models
Models define the architecture of models that will be used in the trainers. 
These contain just the architecture of the model and the forward function. Also, they can include model-specific functions.


### Modules
This folder contains modules that can be used in multiple models. For instance, attention blocks or residual blocks that are easily pluggable into other models.

### Datamodules
Each file contains the definition of a dataset and its dataloaders, other than any other dataset-specific functions.

### Evaluators
These modules instantiate a model and run them on test sets. They also compute metrics and save images locally for visual inspection.