# Analysis Scripts and Data

This repository contains scripts and supporting data for the analysis presented in the manuscript:

**A layer specific histological framework for synaptic quantification in hippocampal CA1"**  
*(Currently under peer review)*

---

## Contents
 
  Main folder includes `.py` (analysis script) necessary to reproduce the corresponding figure.
  
- `LICENSE.md`:  
MIT license for code, and CC BY 4.0 license for data.

- `ResNet-example.zip`:  
This archive provides an example dataset for testing the ResNet classification workflow.
It contains 256 × 256 pixel images in PNG format, organized in ImageFolder structure with four classes: Other, SO, SR, and SLM.
These example images are intended to demonstrate the expected input format and can be used to quickly verify that the training and classification scripts run correctly.

---

## How to use

## 🔹ResNet classification

First, organize the dataset in standard ImageFolder format, with each class stored in a separate subfolder under the main dataset directory.
When the script is run, a dialog window will appear prompting the user to select this dataset folder, which is then used as the input for training and validation.

The script automatically resizes all images to 224 × 224 pixels, normalizes them, and splits the dataset into training and validation subsets. It then trains a ResNet-34 model from scratch using the specified training parameters and evaluates classification performance on the validation set.

The trained model, loss curve, and confusion matrix are saved automatically in the models folder. In addition, a classification report including precision, recall, and F1-score for each class is printed in the console, allowing the overall performance of the classifier to be assessed.

## 🔹ResNet fine-tuning

First, organize the additional training dataset in standard ImageFolder format, with each class stored in a separate subfolder under the main dataset directory.
When the script is run, a dialog window will appear prompting the user to select this dataset folder, which is then used as the input for fine-tuning.

The script automatically resizes all images to 224 × 224 pixels, normalizes them, and loads the previously trained ResNet-34 model from the models folder. It then performs additional training on the selected dataset using the specified fine-tuning parameters.

After fine-tuning, the updated model weights are saved by overwriting the existing checkpoint file (resnet34_trained.pth). Training loss for each epoch is printed in the console, allowing the progress of fine-tuning to be monitored.

## 🔹 synapse quantification

First, place the 16-bit two-channel TIFF image(s) in the main folder.
When apply_log_filter_multichannel.py is run and the folder is selected, a new subfolder named LoG_output is automatically generated, containing the processed output images for subsequent analysis.

Within LoG_output, manually create a subfolder called ref, and save the RGB merged versions of the corresponding two-channel TIFF image(s) in this folder. These reference images are used to visually identify laminar boundaries and serve as guides for the manual segmentation step in the next stage of the analysis.

Then, run analyze_synaptic_puncta_blocks.py. A graphical window will open, allowing the user to draw segmented lines manually according to the displayed instructions. Based on the defined boundaries, the script assigns each 256 × 256 pixel block to a laminar region and quantifies puncta number and area in each channel using threshold-based particle detection. The results are automatically saved as CSV files for each channel in the corresponding output folder.

---

## License

## 🔹 Code

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
