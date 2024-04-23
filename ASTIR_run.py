import os
import json
import argparse

import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc

from sklearn.metrics import adjusted_rand_score

from astir import from_anndata_yaml


def _plot_umap(adata):
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    fig = sc.pl.umap(adata, color=['pred'], size=5, return_fig=True)

    return fig


def main():
    parser = argparse.ArgumentParser(description='astir')
    parser.add_argument('--base_path', type=str, required=True,
                        help='configuration_path')
    args = parser.parse_args()

    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    args.dataset = config['dataset']
    args.markers = config['markers']
    args.threshold = config['results_threshold']
    args.max_epochs = config['max_epochs']
    args.learning_rate = config['learning_rate']
    args.batch_size = config['batch_size']
    args.delta_loss = config['delta_loss']
    args.n_init = config['n_init']
    args.n_init_epochs = config['n_init_epochs']
    args.seed = config['seed']

    # Create ASTIR Object
    astir = from_anndata_yaml(
        anndata_file=args.dataset,
        marker_yaml=args.markers,
        create_design_mat=True,
        random_seed=args.seed
    )

    # Run training
    astir.fit_type(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        delta_loss=args.delta_loss,
        n_init=args.n_init,
        n_init_epochs=args.n_init_epochs
    )

    # Print value counts
    print('Value counts of predictions:')
    print(astir.get_celltypes(threshold=args.threshold).value_counts())

    adata = ad.read_h5ad(args.dataset)

    adata.obs['pred'] = astir.get_celltypes(threshold=0.7).rename(columns={'cell_type': 'pred'})
    adata.obs['pred_prob'] = astir.get_celltype_probabilities().apply(lambda row: np.max(row), axis=1)
    adata.obs['prob_list'] = astir.get_celltype_probabilities().apply(lambda row: row.tolist(), axis=1)

    results_df = adata.obs[['batch', 'cell_id', 'cell_type', 'pred', 'pred_prob', 'prob_list']]
    results_df = results_df.rename(columns={'batch': 'image_id', 'cell_type': 'label'})

    results_df.to_csv(os.path.join(args.base_path, 'astir_results.csv'), index=False)

    # Plot UMAP of predictions
    adata.obs['pred'] = pd.Categorical(results_df['pred'])
    figure = _plot_umap(adata)
    figure.savefig(os.path.join(args.base_path, 'UMAP_predictions.pdf'), format="pdf", bbox_inches="tight")

    # Calculate ARI Score
    ari_score = adjusted_rand_score(results_df['label'], results_df['pred'])
    print("Adjusted Rand Index Score:", ari_score)


if __name__ == '__main__':
    main()
