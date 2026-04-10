# Analysis Scripts and Data

This repository contains scripts and supporting data for the analysis presented in the manuscript:

**A layer specific histological framework for synaptic quantification in hippocampal CA1"**  
*(Currently under peer review)*

---

## Contents
 
  Main folder includes `.py` (analysis script) necessary to reproduce the corresponding figure.
  
- `LICENSE.md`:  
  MIT license for code, and CC BY 4.0 license for data.

---

## How to use

- `Automated layer classification using ResNet`

First, prepare the dataset in the standard ImageFolder format, where each class is stored in a separate subdirectory:

dataset/
├── class1/
│   ├── img1.tif
│   ├── img2.tif
│   └── ...
├── class2/
│   ├── img1.tif
│   └── ...

Each subfolder name will be treated as a class label.

Next, run the script. A dialog window will appear prompting you to select the dataset directory.
Please choose the root folder containing the class subdirectories.

The script will then automatically:

Resize all images to 224 × 224 pixels
Normalize pixel values (mean = 0.5, std = 0.5 for each channel)
Split the dataset into training and validation sets (default: 85% / 15%)
Train a ResNet-34 model from scratch
Evaluate the model on the validation set

The following outputs will be generated in the models directory:

resnet34_trained.pth
→ trained model weights and metadata (class names, number of classes)
loss_curve.eps
→ training and validation loss curves
confusion_matrix.eps
→ confusion matrix for validation predictions

In addition, a classification report (precision, recall, F1-score) will be printed in the console.

A ResNet-34 model was trained using an ImageFolder-formatted dataset, with images resized to 224 × 224 pixels and normalized. The dataset was randomly split into training and validation subsets (85%/15%), and the model was trained using the Adam optimizer with a learning rate of 1 × 10⁻⁴. Performance was evaluated on the validation set using classification metrics and a confusion matrix.

- `synapse quantification`

First, place the 16-bit two-channel TIFF image(s) in the main folder.
When apply_log_filter_multichannel.py is run and the folder is selected, a new subfolder named LoG_output is automatically generated.

Within LoG_output, manually create a subfolder called ref, and save the RGB merged versions of the corresponding two-channel TIFF image(s) in this folder. These reference images are used in the subsequent manual segmentation step.

Then, run analyze_synaptic_puncta_blocks.py. A graphical window will open, allowing the user to draw segmented lines manually. Please proceed according to the displayed instructions.

---

## License

### 🔹 Code

All Python scripts (`.py`) in this repository are licensed under the [MIT License](LICENSE.md).  
You are free to use, modify, and redistribute the code with attribution.

**Note:** Redistribution or reuse of the data is not permitted until the manuscript is officially published.  
Please contact the corresponding author for any data-related inquiries.

---

## Contact

For questions, please contact:  
**Sotaro Ichinose** – [ichinose@gunma-u.ac.jp]

---

## Citation

(Include this section after acceptance)
