# This python script provides an example pipeline using CirroPrep.

import CirroPrep as cirrop
import pandas as pd


def main():

    # The entire pipeline can be done with prepare() like below.
    # However the data must be formatted correctly. (ex. 'sample_id' is the name of the sample columns)
    # Otherwise, use the pipeline below as a guide.
    # cirrop.prepare(path_to_input='demo/demo_data.h5ad',
    #                path_to_output='demo/processed.h5ad',
    #                path_to_embed='demo/embed.csv',
    #                path_to_meta='demo/metadata.csv',
    #                norm=True)

    # read an h5ad file as AnnData object.
    adata = cirrop.read_file('demo/demo_data.h5ad')

    # normalize and log values of the data matrix.
    cirrop.normalize(adata, log=True)

    # append a precalculated UMAP embedding.
    embed = pd.read_csv('demo/embed.csv', index_col='sample_id')
    cirrop.append_embedding(adata, embedding=embed, name='EMBED')

    # calculate and append an autoencoder-generated embedding.
    # cirrop.append_embedding(adata, method='ae')

    # calculate and append cell clusters using Leiden.
    # cirrop.apply_leiden_clust()

    # append sample metadata with provided DataFrame after combining id and label columns and subsetting.
    meta = pd.read_csv('demo/metadata.csv', index_col='sample_id')

    cirrop.combine_id_labels(meta, columns=['class', 'subclass', 'region'])
    meta = cirrop.subset_metadata(meta, columns=['class', 'subclass', 'region'])

    cirrop.append_metadata(adata, metadata_df=meta)

    # write modified h5ad file
    cirrop.write(adata, 'demo/processed.h5ad')

    # next, freeze with cirro prepare_data
    # see example_sun.sh


if __name__ == "__main__":
    main()