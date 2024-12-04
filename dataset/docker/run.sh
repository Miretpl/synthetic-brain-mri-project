echo "Extract images from nil format to PNG"
python ./code/01_extract_png_images_from_nii.py
echo "Split images to train, validation and test sets"
python ./code/02_split_dataset_to_sets.py