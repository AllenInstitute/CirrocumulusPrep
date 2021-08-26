# This bash script is an example of how to freeze your prepared data and visualize with Cirrocumulus.

python example.py
cirro prepare_data processed.h5ad
cirro launch processed

# if uploading to gcloud, modify and run the below command
# gsutil -m cp -r processed gs://processed