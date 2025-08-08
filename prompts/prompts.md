# PROMPTS

## Classification

1. Study the codebase in `third_party/mlx_vlm` and analyse how to run the gemma3 and paligemma models using a Python API. Dump your analysis in a Markdown file in `mlx-vlm-gemma.md`

2. I need to design a script called `evaluate_labels_only.py` that does the following. First, it should use the `fiftyone` package in Python for loading a dataset. Then for each image in the dataset it should pass it to a model from MLX VLM (like gemma3) with a prompt that asks it to identify all items in the image that are contained in a user-specified list. Finally, the labels from the dataset should be compared against the response from the MLX VLM model, and a suitable metric should be computed to determine how accurate the model is at identifying all relevant categories in each image. These results should be saved as JSON to a file. This script should be runnable as a command line tool using argparse. It's important that the user can pass in an arbitrary model to the CLI. Also it's important that intermediate results can be saved to a checkpoint file. Do not run the script or create any tests.

3. Read the script `evaluate_labels_only.py.` Now create a script `label_viewer.py` that uses `fiftyone` to view the images that have been analysed by the script evaluate_labels_only.py. This script should use argparse for CLI again. It should also attach the predicted labels for each image as metadata that `fiftyone` can display. Don't run the script.

4. Analyse the script `evaluate_labels_only.py`. This script outputs the results of an analysis into a JSON file. Write a new script `label_results_analysis.py` that computes relevant and interesting statistics for i) a single JSON file and ii) a comparison between multiple JSON files. Don't go into too much detail - try and produce statistics that maximise the amount of information learned such as the overall metric scores, the top and lowest performing categories, and perhaps some distributions. This script should then make this analysis available in the form of a static dashboard.


## Detection

1. I need to design a script called `detect_locations.py` that does the following. It should use the `fiftyone` package in Python for loading the dataset. Then for each image in the dataset it should use a model from the PaliGemma family (running with MLX VLM) to detect the location of items in the image. The location detection should work as follows. First, analyse the output from the script `evaluate_labels_only.py` to determine which items have been predicted for the image. Then for each item predicted, pass it to the PaliGemma model with the prompt 'detect <ITEM_NAME>'. Do this for all items predicted for the image. Then iterate over all images and save the results. Do not run the script.

2. Analyse the script `detect_locations.py`. I want you to update this scripts. The modifications should be as follows. The PaliGemma models will return bounding box predictions in a format like <loc0393><loc0109><loc0859><loc0811>, and this then needs to be converted to bounding box coordinates. The conversion process is as follows.:
<CONVERSION_INSTRUCTIONS>
Detailed Breakdown
The conversion process can be broken down into four main steps that occur within the extract_objs function.
a. Model Output Generation
First, the PaliGemma model processes an image and a text prompt (e.g., "detect bird"). It generates a string that includes special tokens for the location and label of the detected object.
A typical output string looks like this:
detect bird <loc0393><loc0109><loc0859><loc0811> bird
The crucial part is the sequence <loc0393><loc0109><loc0859><loc0811>, which represents the bounding box coordinates in the format $y_1, x_1, y_2, x_2$.
b. Parsing with a Regular Expression
The extract_objs function uses a regular expression (_SEGMENT_DETECT_RE) to find and extract the coordinate numbers and the object's name from the model's output string.
The regex is specifically designed to capture the four 4-digit numbers that follow the <loc...> tags.
c. Normalization and Conversion
This is the core of the conversion. The code takes the four extracted numbers (e.g., 0393, 0109, 0859, 0811) and converts them into normalized coordinates that range from 0.0 to 1.0.
This is done in the following line:
Python
y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
Here's what it does:
int(x): Converts the string number (e.g., "0393") into an integer (e.g., 393).
/ 1024: Divides the integer by 1024. This normalizes the coordinate, assuming the model was trained on a 1024x1024 coordinate space.
Using the example <loc0393><loc0109><loc0859><loc0811>:
$y_1 = 393 / 1024 \approx 0.384$
$x_1 = 109 / 1024 \approx 0.106$
$y_2 = 859 / 1024 \approx 0.839$
$x_2 = 811 / 1024 \approx 0.792$
The result is a bounding box defined by the points (y1, x1) and (y2, x2) in a normalized format.
d. Scaling to Image Pixels 📏
Finally, the normalized coordinates are scaled to match the dimensions of the original input image. The function takes the image's width and height as input to perform this final step.
This is done with this line of code:
Python
y1, x1, y2, x2 = map(round, (y1height, x1width, y2height, x2width))
The normalized y-coordinates (y1, y2) are multiplied by the image height.
The normalized x-coordinates (x1, x2) are multiplied by the image width.
map(round, ...) rounds the results to the nearest whole number to get the final pixel coordinates for the bounding box.
</CONVERSION_INSTRUCTIONS>

3. Create a script `detection/bbox_results_analysis.py` that is similar to `classification/label_results_analysis.py` in that it is a static dashboard. In this static dashboard you should only include Overview Metrics, AP per object (both best and lowest 10) and also some brief mention of other metrics like IoU threshold. DO NOT include any analysis about missed predictions. Also allow the ability to compare multiple datasets against each other.

4. Analyse the script `detect_locations.py`. Now create a script `bbox_viewer.py` that uses `fiftyone` to view the images that have been analysed by the script `detect_locations.py`. This script should use argparse for CLI again. It should also attach the predicted bounding boxes and labels for each image as metadata that `fiftyone` can display. Don't run the script.

## Conversion

1. Analyse the script `evaluate_labels_only.py`. Now write another script `convert_dataset_to_fiftyone_format.py` that takes a dataset that is in either COCO or Ultralytics YOLO format, and then converts it into a `fiftyone` dataset format that can be used as input for the script `evaluate_labels_only.py`. Do not run the script.
