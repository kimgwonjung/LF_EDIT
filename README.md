# Micro-cells Enabling Spatio-Angular Joint Attention for High-resolution Light Field Editing


## Overall Pipeline & Network Architecture  
![Overview](DEMO/Overview.png)
The left image shows the overall pipeline of LF Editing, and the right image illustrates the architecture of the Composition Network.

## Result
![Result](DEMO/DEMO.gif)

## Setup
Set up dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Generate a CSV file by writing the file paths of the Train, Validation, and Test images.
2. Run a Light Field Editing model:  
```bash  
python main.py --train_path <path/to/train.csv> --val_path <path/to/val.csv>  
```
3. The results are written to the folder 'Result'. If you want to specify an output path, use `--output_path <path/to/output_path>.`
4. To test the model, run the following command:
```bash
python main_test.py --test_path <path/to/test.csv>
```