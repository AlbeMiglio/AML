# To start the project for the first time

## download pytorch manually
`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`

## download requirements
`pip install -r requirements.txt`

## download datasets
open git bash in the AML_project folder and run:

`./download_data_and_weights.sh`

## to evaluate the baseline model
`python evaluate_metricsBaseline.py`

## to evaluate the Standard late fusion model 
`python evaluate_metricsRGBD.py`

## to evaluate the CNN-5 or the custom resnet-10 model (change the import in the file)
`python evaluate_metricsRGBD_custom.py`
