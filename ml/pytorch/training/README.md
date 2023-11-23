
This will select cuda when running on nvidia gpu, apple gpu when running on apple silicon, and cpu otherwise.

1. conda create --name az_test_pytorch python=3.11
2. conda activate az_test_pytorch
3. pip install torch torchvision torchaudio
4. python train_resnet50_cifar10.py