# This bash script is an example of how to freeze your prepared data and visualize with Cirrocumulus.

# Activate your conda environment with the following packages:
# numpy, scanpy, pandas, anndata, scipy, h5py, pytorch, torchvision
conda activate cirrocumulus

python example_script.py

# Add --whitelist flag to only freeze certain fields.
# Ex. cirro prepare_data processed.h5ad --whitelist obsm (for embedding only)
cirro prepare_data processed.h5ad

cirro launch processed

# if uploading to gcloud, modify and run the below command
# gsutil -m cp -r [local_folder] gs://[gs_path]

# ex. whole directory:
# gsutil -m cp -r 10xv3 gs://cirrocumulus-brain/
# ex. single file:
# gsutil -m cp -r 10xv3/obsm/umap.parquet gs://cirrocumulus-brain/10xv3/obsm