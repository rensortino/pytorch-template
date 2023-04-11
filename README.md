# pytorch-template
A PyTorch template for simplifying neural network code writing, with a ready-to-use folder structure

### Tips
If you wish to use inheritance for models/trainers, use just one layer of inheritance, from base class to specific class, otherwise weird things happen after a while (models instantiated multiple times and other redundant operations)
