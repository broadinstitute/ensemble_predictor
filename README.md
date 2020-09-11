This Python package contains tools for simultaneously building and training multiple supervised regression models that take specified subsets of a feature matrix and use them to predict specified columns of a response matrix. We recommend following the vignettes in the included jupyter notebook `Ensemble Vignettes.ipynb` (which will also install the package). The included notebook FigurePanelGeneration.ipynb will generate all the panels in the manuscript "Gene expression has more power for predicting in vitro cancer cell vulnerabilities than genomics" by Dempster et al., available as a preprint at https://www.biorxiv.org/content/10.1101/2020.02.21.959627v3. 

Requirements:

python 2.7 or 2.5 or greater
numpy 1.16 or greater
scipy 1.1 or greater
sklearn 0.19 or greater
pandas 0.24 or greater

Tested with:

OSX 10.13.6 High Sierra
python 3.6.5
numpy 1.16.2
scipy 1.1.0
sklearn 0.19.2
pandas 0.24.1

To install, navigate to the directory in the console and type. The nomenclature and naming conventions assume you are trying to predict genetic perturbation responses from omics, but the tools can be used for any supervised regression task.

    pip install .

To import in Python:

    import ensemble_predictor

The main function provided in this package is `run_subset`. This allows you to evaluate a predefined list of models (all instances of `KFilteredFeatureForest`) using a supplied `pandas.DataFrame` feature matrix `X` to independently predict each column of a matrix od perturbation responses `Y`. Results are saved in a specified `directory`. Each default model has 100 estimators, max depth 6, and minimum samples per leaf of 5. Additionally, each model filters the feature list for features of a specific type (indicated by the suffix of their column names in `X`), then filters those features to the 1000 with highest correlation with the label according to the training set. The models will always retain "Cell" features regardless of correlation. The predefined models are:

"KitchenSink": Takes in all feature types
"Histology": Takes in all features ending in `"_Cell"`. No additional feature filtration.
"Expression": All features ending in "Exp" (RNASeq expression) or "Cell". 
"Mutation": All features ending in "Hot" (hotspot), "Dam" (damaging), or "Other" (other mutation), or "Cell"
"GSEA": All features ending in "GSEA" (single sample GSEA) or "Cell"
"Methylation": All features ending in   "MethTSS" (Methylation of transcription start site) or "MethCpG" (CpG island), or "Cell" 
"CN": All features ending in "CN" (copy number) or "Cell"
"Fusion": All features ending in "Fusion" (gene fusions) or "Cell"
"CFE": If a separate feature matrix is supplied with the `cfe` argument, this model will use the top 1000 most correlated features from that matrix.
"AllGenomics": Superset of "Mutation", "Methylation", "CN", and "Fusion"
"AllExpression": Superset of "Expression" and "GSEA"
"X+Exp": Superset of "Expression" and the indicated dataset.

Only a subset of these models can be used by passing a list of the names of the desired models with the parameter `included_models`. Additionally, a contiguous block of columns in `Y` can be specified for evaluation using `start_col` to indicate the start of the block and `n_col` to indicate the number of columns. This is useful for training the ensemble for many perturbations in parallel.

`KFilteredForest` and `KFilteredFeatureForest` are useful stand-alone classes. Their interface is similar to sklearn's `RandomForestRegressor` except that they expect `pandas` objects for `X` and `y` in `train` and `predict`. `KFilteredForest` identifies a set of `k` columns in `X` during training that are most highly correlated with `y` and only uses those for training and prediction. `KFilteredFeatureForest` is similar but first filters according to retain only columns whose names end in one of a list of specified `suffixes`. Both models allow you to pass a list of `reserved_columns` during construction giving the names of features that will always be retained and not filtered out. 