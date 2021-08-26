import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
import h5py
import DenoisingAutoencoder


def read_file(path):
    adata = sc.read(path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


def subset_metadata(df, columns=None):
    if columns is None:
        # update this with default column names
        columns = ['class', 'subclass', 'region']
    return df[columns]


def combine_id_labels(df, columns=None):
    if columns is None:
        # update this with default column names
        columns = ['class', 'subclass', 'region']
    for col in columns:
        df[col] = df[col + '_id'] + '_' + df[col + '_label']


def append_metadata(adata, metadata_df, sample_col=None):
    if sample_col is not None:
        set_index_as_col(metadata_df, sample_col)

    meta_cols = metadata_df.columns.values
    adata.obs[meta_cols] = metadata_df[meta_cols]


def set_index_as_col(df, col):
    df.index = df[col]
    del df[col]


def append_embedding(adata, embedding=None, name="EMBED", sample_col=None, method='umap'):
    if embedding is not None:
        if type(embedding) is pd.DataFrame:
            if sample_col is not None:
                set_index_as_col(embedding, sample_col)
            embedding = embedding.loc[adata.obs.index]
            embedding = np.array(embedding)
        adata.obsm[name] = embedding
    elif method is 'umap':
        apply_umap(adata)
    elif method is 'ae':
        apply_ae_embed(adata)
    else:
        print('unknown embedding method')


def normalize(adata, target_sum=1e6):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def select_variable_genes(adata, n_top_genes=5000):
    adata.X = adata.X.toarray()
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata


def apply_umap(adata):
    sc.tl.umap(adata, n_components=3)


def apply_pca(adata):
    sc.tl.pca(adata, svd_solver='arpack')


def apply_leiden_clust(adata):
    sc.tl.leiden(adata)


def apply_ae_embed(adata, train_split=0.85, train_batch_size=1000, test_batch_size=1000, epochs=1000,
          embed_batch_size=1000):
    adata = select_variable_genes(adata, n_top_genes=5000)
    df = DenoisingAutoencoder.embed(adata, train_split, train_batch_size, test_batch_size, epochs,
          embed_batch_size)
    append_embedding(adata, embedding=df, name='AE_embed')


def h5sc_to_h5ad(in_file, out_file):
    f = h5py.File(in_file, 'r')
    obs_names = f['mm10-3.0.0_premrna/_barcodes/barcodekey'][()]
    obs = pd.DataFrame(index=obs_names)
    var_names = f['mm10-3.0.0_premrna/_features/featurename'][()]
    var = pd.DataFrame(index=var_names)
    x = f['mm10-3.0.0_premrna/data'][()]
    i = f['mm10-3.0.0_premrna/indices'][()]
    p = f['mm10-3.0.0_premrna/indptr'][()]
    dim = f['mm10-3.0.0_premrna/shape'][()]
    mat = csr_matrix((x, i, p), shape=tuple(reversed(tuple(dim))))
    adata = ad.AnnData(X=mat, obs=obs, var=var)
    adata.write(out_file)


def write(adata, out_path):
    adata.raw.to_adata().write(out_path)

def main():
    adata = read_file('one_batch_data.h5ad')

    normalize(adata)

    #append_embedding(adata, method='ae')
    append_embedding(adata, embedding=pd.read_csv('umap.csv'), name='UMAP', sample_col='sample_id')

    meta = pd.read_csv('metadata.csv')
    combine_id_labels(meta)
    subset_metadata(meta)
    append_metadata(adata, metadata_df=meta, sample_col='sample_id')

    write(adata, 'processed.h5ad')


if __name__ == "__main__":
    main()