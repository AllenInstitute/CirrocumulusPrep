import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
import h5py
import DenoisingAutoencoder


# Append a 3D embedding to given AnnData object.
# A pre-calculated embedding can be provided as a DataFrame or numpy array.
# If a DataFrame is provided, column with sample names must be provided if not the index.
# An embedding can be calculated if not provided by specifying 'method'.
# method = 'umap' will compute a UMAP embedding, method = 'ae' will compute an autoencoder-generated embedding.
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


# Applies a 3D autoencoder-generated embedding to a given AnnData object.
# 'train_split' specifies the proportion of data used for training with the remaining proportion used for
# calculating test error
# 'train_batch_size' specifies the number of training samples used per epoch
# 'test_batch_size' specifies the number of testing samples used per epoch
# 'embed_batch_size' specifies the chunk size when applying the embedding.
# A higher value will use more memory but less time, a smaller value will use less memory but take longer.
def apply_ae_embed(adata, train_split=0.85, train_batch_size=1000, test_batch_size=1000, epochs=1000,
          embed_batch_size=1000):
    adata = select_variable_genes(adata, n_top_genes=5000)
    df = DenoisingAutoencoder.embed(adata, train_split, train_batch_size, test_batch_size, epochs,
          embed_batch_size)
    append_embedding(adata, embedding=df, name='AE_embed')


# Helper method for apply_ae_embed().
# Returns a subset of given AnnData object to specified number of most variable genes.
def select_variable_genes(adata, n_top_genes=5000):
    adata.X = adata.X.toarray()
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata


# Applies a 3D UMAP embedding to given AnnData object.
def apply_umap(adata):
    sc.tl.umap(adata, n_components=3)


# Applies a 3D PCA embedding to given AnnData object.
def apply_pca(adata):
    sc.tl.pca(adata, svd_solver='arpack', n_comps=3)


# Append metadata DataFrame to given AnnData object.
# Column with sample names must be provided if not the index.
def append_metadata(adata, metadata_df, sample_col=None):
    if sample_col is not None:
        set_index_as_col(metadata_df, sample_col)

    meta_cols = metadata_df.columns.values
    adata.obs[meta_cols] = metadata_df[meta_cols]


# Helper method for append_metadata().
# Sets the index of a DataFrame to a given column, then removes the column.
def set_index_as_col(df, col):
    df.index = df[col]
    del df[col]


# Subset metadata DataFrame by given set of columns, or default.
def subset_metadata(df, columns=None):
    if columns is None:
        columns = ['class', 'subclass', 'region']
    return df[columns]


# Combine col_id and col_label into col for given set of columns, or default.
def combine_id_labels(df, columns=None):
    if columns is None:
        columns = ['class', 'subclass', 'region']
    for col in columns:
        df[col] = df[col + '_id'] + '_' + df[col + '_label']


# Read in transcriptomics data matrix at given path.
def read_file(path):
    adata = sc.read(path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


# Writes given AnnData object to specified path as H5ad.
def write(adata, out_path):
    adata.raw.to_adata().write(out_path)


# Normalize data matrix of a given AnnData object to target sum. Logarithm will be calculated by default.
def normalize(adata, target_sum=1e6, log=True):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    if log:
        sc.pp.log1p(adata)


# Calculates clusters using Leiden clustering algorithm.
def apply_leiden_clust(adata):
    sc.tl.leiden(adata)


# Converts a specified H5sc file to H5ad for processing.
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

