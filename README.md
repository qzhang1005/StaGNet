# StaGNet: A Stable Gene Network Framework for Key Driver Detection Across Age Trajectories

StaGNet is a robust framework for identifying key driver genes across developmental time using bulk or pseudo-bulk transcriptomic data. The pipeline combines variance and mutual information-based feature selection, elastic net-based network construction with bootstrap aggregation, and robust centrality metrics to detect stable and biologically meaningful gene interactions.


# Core components

📊 Feature Selection: Removes low-variance genes and selects features most informative of developmental age using mutual information.

🔁 Network Construction: Builds robust gene-gene interaction networks by repeated bootstrapped ElasticNet regression with stability filtering.

🧠 Key Driver Identification: Calculates PageRank, betweenness, and eigenvector centralities, followed by robust normalization and weighted integration.

🎯 Focus on Developmental Time: Treats age (in months) as the sole regression target, excluding it from network construction to prevent data leakage.

📈 Visualization: Outputs interpretable subnetwork visualizations of top-ranked genes using force-directed layouts.


# Key Features

Deterministic and reproducible results using fixed seeds

Avoids common pitfalls like centrality metric collapse (e.g., NaNs)

Suited for developmental transcriptomics across cell types or brain regions


This pipeline was tested on neuronal samples from the BrainSpan dataset and is generalizable to other age-related transcriptomic studies.
