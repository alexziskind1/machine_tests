
1. conda create --name az_test_matrix_multi python=3.11
2. conda activate az_test_matrix_multi
3. If not using accelerate framework (slower)
    pip install numpy
   else if using the accelerate framework (faster), build numpy
    pip install --no-binary :all: --no-use-pep517 numpy==1.25.2
4. python run.py