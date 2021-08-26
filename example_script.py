# This python script provides an example pipeline using CirroPrep.

import CirroPrep as cirrop
import pandas as pd


def main():
    # read an h5ad file as AnnData object.
    adata = cirrop.read_file('demo_data.h5ad')

    # normalize and log values of the data matrix.
    cirrop.normalize(adata, log=True)

    # append a precalculated UMAP embedding.
    umap = pd.read_csv('umap.csv', index_col='sample_id')
    cirrop.append_embedding(adata, embedding=umap, name='UMAP')

    # calculate and append an autoencoder-generated embedding.
    # cirrop.append_embedding(adata, method='ae')

    # calculate and append cell clusters using Leiden.
    # cirrop.apply_leiden_clust()

    # append sample metadata with provided DataFrame after combining id and label columns and subsetting.
    meta = pd.read_csv('metadata.csv', index_col='sample_id')

    cirrop.combine_id_labels(meta, columns=['class', 'subclass', 'region', 'cluster'])
    meta = cirrop.subset_metadata(meta, columns=['class', 'subclass', 'region', 'cluster'])

    cirrop.append_metadata(adata, metadata_df=meta)

    # write modified h5ad file
    cirrop.write(adata, 'processed.h5ad')

    # next, freeze with cirro prepare_data
    # see example_sun.sh


if __name__ == "__main__":
    main()