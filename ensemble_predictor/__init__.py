from __future__ import print_function
import numpy as np
import pandas as pd
from time import time
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, make_scorer, r2_score
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_regression, f_regression
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
import argparse
import os
import gc
import h5py

'''
This package contains a series of convenience functions and classes written in support of the manuscript "Gene expression has more power for predicting in
vitro cancer cell vulnerabilities than genomics" by Joshua M. Dempster et. al., the Broad Institute of MIT and Harvard.
'''

##############################################################
################### F U N C T I O N S ########################
##############################################################


def write_hdf5(df, filename):
	'''
	writes a numerical matrix in a convenient hdf5
	Parameters:
		df (`pandas.DataFrame`): matrix to be written
		filename (`str`): path to file
	'''
	if os.path.exists(filename):
		os.remove(filename)
	dest = h5py.File(filename)

	try:
		dim_0 = [x.encode('utf8') for x in df.index]
		dim_1 = [x.encode('utf8') for x in df.columns]

		dest_dim_0 = dest.create_dataset('dim_0', track_times=False, data=dim_0)
		dest_dim_1 = dest.create_dataset('dim_1', track_times=False, data=dim_1)
		dest.create_dataset("data", track_times=False, data=df.values)
	finally:
		dest.close()


def read_hdf5(filename):
	'''
	reads an hdf5 matrix saved with write_hdf5
	Parameters:
		filename (`str`): path to hdf5 file
	Returns:
		`pandas.DataFrame`
	'''
	src = h5py.File(filename, 'r')
	try:
		dim_0 = [x.decode('utf8') for x in src['dim_0']]
		dim_1 = [x.decode('utf8') for x in src['dim_1']]
		data = np.array(src['data'])

		return pd.DataFrame(index=dim_0, columns=dim_1, data=data)
	finally:
		src.close()


def pearson_score(ytrue, ypred):
	if all(ypred == np.mean(np.mean(ypred))) or all(ytrue == np.mean(np.mean(ytrue))):
		return 0
	return pearsonr(ytrue, ypred)[0]


def varfilter(X, threshold):
	return (np.std(np.array(X), axis=0) > threshold)


def _single_fit(column, X, Y, model_types, splitter, scoring, nfeatures=10, return_models=False):
	'''
	Takes a single column from `Y` as the label and builds a model for each model in `model_types` to predict it
	using the corresponding features in `X`. Intended to be used by `EnsembleRegressor`.
	'''
	y = Y[column]
	if y.std() == 0:
		raise ValueError("Column %s has 0 variance\n%r" %(
			column, y)
		)
	if y.isnull().any():
		raise ValueError('Column %s of y contains %i nulls' %(column, y.isnull().sum()))
	target_models = [val['ModelClass'](**val['kwargs']) for val in model_types]
	scores = []
	features = []
	prediction = []
	for model, x in zip(target_models, X):
		if x.isnull().any().any():
			raise ValueError('Feature set for model %r contains nulls. Axial sums of nulls:\n%r\n\n%r' %(
				model, x.isnull().sum(), x.isnull().sum(axis=1)))
		splits = splitter.split(y, y)
		score = []
		n = 0
		model_prediction = pd.Series(np.nan, index=Y.index, name=column)
		for train, test in splits:
			try:
				model.fit(x.iloc[train], y.iloc[train])
				ypred = model.predict(x.iloc[test])
			except Exception as e:
				print('error fitting model %r for column %s' %(model, column))
				print('train indices:\n %r\n' %train)
				print('test indices:\n %r\n' %test)
				print('train features: \n%r\n' %x.iloc[train])
				print('test features: \n%r\n' %x.iloc[test])
				print('train column: \n%r\n' %y.iloc[train])
				print('test column: \n%r\n' %y.iloc[test])
				raise e 
			model_prediction.iloc[test] = ypred[:]
			score.append(scoring(y.iloc[test], ypred))
			n += 1
		scores.append(score)
		prediction.append(model_prediction)
		model.fit(x, y)
		try:
			features.append(model.get_feature_series(nfeatures))
		except AttributeError:
			features.append(pd.Series(dtype=np.float))
		gc.collect()
	best_index = np.argmax(np.mean(scores, axis=1))
	if not return_models:
	 target_models = [np.nan for i in range(len(scores))]
	return {'models': target_models, 'best': best_index, 'scores': scores, 'features': features, 'predictions': prediction}



##############################################################
#################### E N S E M B L E #########################
##############################################################

class EnsembleRegressor:    
	def __init__(self, model_types, nfolds=10, scoring=pearson_score, Splitter=KFold):
		'''
		Class for fitting multiple models over a series of folds and storing the predictions (and some summary statistic
		of performance) per fold. Models are finally retrained with all data to derive feature importance.
		Parameters:
			model_types: specified with the format [{'Name': str, 'ModelClass': class,  'kwargs': dict}] 
						Model classes will be initiated with the dict of keyword arguments.
			nfolds: number of folds for cross-validation
			scoring: function that accepts two `pandas.Sequence` objects, ytrue and ypred, and returns a scalar to indicate
					performance
			Splitter: class with interface like `sklearn.model_selection.KFold` to form cross-val splits.

		'''
		self.model_types = model_types
		self.best_indices = {}
		self.trained_models = {}
		self.scores = {}
		self.important_features = {}
		self.nfolds = nfolds
		self.splitter = Splitter(n_splits=nfolds, shuffle=True)
		self.scoring = scoring
		self.columns = None
		self.predictions = None

		
	def check_x(self, X):
		'''
		check list of feature matrices X
		'''
		xerror = ValueError(
			'X must be a list or array with a feature set dataframe of matching indices for each model \
			present in the ensemble, passed\n%r'
			%X
		)
		if not len(X) == len(self.model_types):
			print('X not the same length as models\n')
			raise xerror
		for df in X[1:]:
			if not all(df.index == X[0].index):
				raise xerror

		
	def fit(self, X, Y, columns=None, report_freq=20):
		'''
		Trains models and saves performances as instance attributes. The trained models are stored in `self.trained_models`,
		which model performed best according to `scoring` (averaged over frames) is stored in `self.best_indices` as an integer 
		giving the index in the list of models, the OOS scores for each model in each fold are stored in `self.scores`, 
		and the top ten most important features observed 
		Parameters:
			X (`list` of `{ModelClass: pandas.DataFrame}`): the features to be used for each model
			Y (`pandas.DataFrame`): Dataframe of the labels to be predicted, one column per label. The total number of models that 
							will be fitted is equal to `self.nfolds * len(x) * Y.shape[1]` if `columns` is `None`, 
							otherwise `self.nfolds * len(x) * len(columns)`
			report_freq (`float`): the time in seconds between reports. Whenever the ensemble finishes training all models for all
							folds for a column, if the time since its last report is greater than `report_freq`, it will print
							a report giving the time elapsed and estimated remaining.


		'''
		self.check_x(X)
		assert isinstance(Y, pd.DataFrame)
		if not all(Y.index == X[0].index):
			raise ValueError('Y must be a dataframe with index matching the indices in X')
		if columns is None:
			columns = Y.columns
		self.columns = Y.columns
		n = len(self.model_types)
		outputs = {'models': {}, 'best': {}, 'scores': {}, 'features': {}, 'predictions': {}}
		start_time = time()
		curr_time = start_time
		for i, col in enumerate(columns):
			ind = Y.index[Y[col].notnull()]
			output = _single_fit(column=col, X=[x.loc[ind] for x in X], Y=Y.loc[ind], model_types=self.model_types, splitter=self.splitter, 
					  scoring=self.scoring)
			for key in outputs.keys():
					outputs[key][col] = output[key]
			t = time()
			if t - curr_time > report_freq:
				print(
					'%f elapsed, %i%% complete, %f estimated remaining' %(
						t - start_time, int(100*(i+1)/len(columns)), (t-start_time)*(len(columns)-i-1)*1./(i+1))
					)
				curr_time = t
		self.trained_models.update(outputs['models'])
		self.best_indices.update(outputs['best'])
		self.scores.update(outputs['scores'])
		self.important_features.update(outputs['features'])
		predictions = [{col: val[j] for col, val in outputs['predictions'].items()} for j in range(n)]
		if self.predictions is None:
			self.predictions = [pd.DataFrame(v) for v in predictions]
		else:
			for i in range(len(self.model_types)):
				self.predictions[i] = self.predictions[i].join(outputs['predictions'][i])

	
	def save_results(self, name):
		'''
		Saves a summary of model results, including scores for indvidual folds, top ten feature importances, and 
		whether that model performed best, along with concatenated out of sample predictions for each model across
		samples for each column. 
		Parameters:
			name (`str`): the prefix for the path where results will be saved. Overall summaries will be saved at
			`name + '_summary.csv'`, while each model's out-of-sample predictions for all trained columns will
			be saved at `name + '_[model_name]_predictions.csv'`
		'''
		columns = ['gene', 'model']
		for i in range(self.nfolds):
			columns.append('score%i' %i)
		columns.append('best')
		for i in range(10):
			columns.extend(['feature%i' %i, 'feature%i_importance' %i])
		
		melted = pd.DataFrame(columns=columns)
		for gene in self.trained_models.keys():
			for i in range(len(self.model_types)):
				row = {
					'gene': gene,
					'model': self.model_types[i]['Name'],
					'best': self.best_indices[gene] == i
				}
				for j in range(self.nfolds):
					row['score%i' %j] = self.scores[gene][i][j]
				for j in range(10):
					try:
						row['feature%i' %j] = self.important_features[gene][i].index[j]
						row['feature%i_importance' %j] = self.important_features[gene][i].iloc[j]
					except IndexError:
						row['feature%i' %j] = np.nan
						row['feature%i_importance' %j] = np.nan
				melted = melted.append(row, ignore_index=True)
		melted.to_csv(name + '_summary.csv', index=None)
		for model, pred in zip(self.model_types, self.predictions):
			pred.to_csv('%s_%s_predictions.csv' %(name, model['Name']))



class KFilteredForest(RandomForestRegressor):
	'''
	Selects the top `k` features with highest correlation with the target and variance greater than `var_threshold`
	'''
	def __init__(self, k=1000, var_threshold=1e-10, **kwargs):
		self.k = k
		RandomForestRegressor.__init__(self, **kwargs)
		self.k = k
		self.filter = SelectKBest(score_func=f_regression, k=k)
		self.var_threshold = var_threshold
		
	def fit(self, X, y, **kwargs):
		if self.var_threshold > 0:
			self.mask1 = varfilter(X, self.var_threshold)
			x = X.loc[:, X.columns[self.mask1]]
		else:
			x = X
		if x.shape[1] > self.k:
			zscore_x = (x.values - x.mean().values)/x.std(ddof=0).values
			zscore_y = (y.values - y.mean())/y.std(ddof=0)
			corr = pd.Series(np.abs(np.mean(zscore_x * zscore_y.reshape((-1, 1)))), index=x.columns).sort_values()
			self.feature_names = corr.index[-self.k:].to_list()
		else:
			self.feature_names = x.columns.tolist()
		x = x.loc[:, self.feature_names]
		RandomForestRegressor.fit(self, x.values, y, **kwargs)

		
	def predict(self, X, **kwargs):
		x = X.loc[:, self.feature_names]
		return RandomForestRegressor.predict(self, x.values, **kwargs)
	
	def get_feature_series(self, n_features):
		if n_features is None:
			n_features = len(self.feature_names)
		imp = pd.Series(self.feature_importances_, index=self.feature_names)
		return imp.sort_values(ascending=False)[:n_features]



class KFilteredFeatureTypeForest(KFilteredForest):
	'''
	Selects the top `k` highest correlation features out of a subset of features 
	based on their suffix (text occuring after the final underscore in the feature 
	column name). 
	'''
	def __init__(self, reserved_columns=[], suffixes=[], k=1000, var_threshold=0, **kwargs):
		KFilteredForest.__init__(self, k, var_threshold, **kwargs)
		self.reserved_columns = reserved_columns
		self.suffixes = suffixes
		self.feature_names = []

	def fit(self, X, y, **kwargs):
		mask = X.columns.to_series().apply(lambda x: str(x).split('_')[-1] in self.suffixes)
		if mask.sum() < 1:
			unique_suffixes = sorted(set([str(s).split('_')[-1] for s in X.columns]))
			raise ValueError('None of the given suffixes %r were found in any of the feature suffixes %r' %(
				self.suffixes, unique_suffixes))
		KFilteredForest.fit(self, X[X.columns[mask]], y)



class PandasForest(RandomForestRegressor):
	'''
	A simple wrapper for RandomForestRegressor that plays nice with dataframes and series instead of numpy arrays
	'''    
	def fit(self, X, y, **kwargs):
		self.feature_names = X.columns.tolist()
		RandomForestRegressor.fit(self, X, y, **kwargs)
		
	def get_feature_series(self, n_features):
		if n_features is None:
			n_features = len(self.feature_names)
		imp = pd.Series(self.feature_importances_, index=self.feature_names)
		return imp.sort_values(ascending=False)[:n_features]



class PandasElasticNet(ElasticNet):
	'''
	A simple wrapper for ElasticNet that plays nice with dataframes and series instead of numpy arrays
	'''
	def fit(self, X, y, **kwargs):
		self.feature_names = X.columns.tolist()
		ElasticNet.fit(self, X, y, **kwargs)
		
	def get_feature_series(self, n_features):
		if n_features is None:
			n_features = len(self.feature_names)
		imp = pd.Series(self.feature_importances_, index=self.feature_names)
		return imp.sort_values(ascending=False)[:n_features]



##########################################################
############  P R E D I C T I O N   P I P E   ############
##########################################################
	

def run_subset(X, Y, start_col=0, n_col=None, nfolds=10,
	included_models=None, cfe=None, directory='./'
	):
	'''
	Get ensemble predictions using feature dataframe `X` for all of the `Y` columns from `start_col` to `start_col + n_col`. 
	This is convenient for parallel computing.
	Parameters:
		X (`pandas.DataFrame`): Matrix of features to be used for prediction. Each column name should end in some '_[FeatureType]',
								such as 'BRAF_Exp'. 
		Y ('pandas.DataFrame'): Matrix of viability scores to predicted, with perturbations as the columns. Index must match `X`.
		start_col, n_col (`int`): numerical indices indicating the columns in `Y` to be trained. Useful for parallel computing.
		nfolds (`int`): number of cross validation folds to be passed to `EnsembleRegressor`
		included_models (`list` of `str`): names of specific models to be included, useful for training only a subset of the default
											models. Different named models will select subsets of features based on their final suffix.
		cfe (`pandas.DataFrame`): if included, an additional feature matrix of features which will be supplied only to the CFE model.
		directory (`str'): path where ensemble results will be saved.


	'''
	if included_models is None or included_models == "None":
		included_models = [
		'Histology', 
		'KitchenSink', 'Expression', 'Mutation', 'Fusion',
		'GSEA','CN', 
		'Methylation', 'AllExpression', 'AllGenomics',
		'Mutation+Exp', 'CN+Exp', 
		'Methylation+Exp',
		]

	if not cfe is None:
		included_models = sorted(set(included_models) | set(['CFE']))
	print('Using the following models:\n%r' % included_models)

	if n_col is None:
		n_col = Y.shape[1] - start_col

	print('aligning features')

	shared_lines = list(set(X.index) & set(Y.index))
	if 'CFE' in included_models:
		shared_lines = list(set(shared_lines) & set(cfe.index))
	assert len(shared_lines) > 0, "no shared lines found: \n\n features %r\n\n "
	Y = Y.loc[shared_lines]
	X = X.loc[shared_lines]

	constant_features = [s for s in X.columns if s.endswith('_Cell')]

	if 'CFE' in included_models:
		cfe = cfe.loc[shared_lines]
		cfe = cfe.join(X[constant_features], how='inner')
	#Y.columns = [s.split(' ')[0] for s in Y.columns]
	if len(X) != len(Y):
		raise RuntimeError('length of X and Y do not match (shapes %r and %r)' %(X.shape, Y.shape))

	lineages = [s for s in X.columns if s.endswith("_Histology")]

	print('creating models')
	models = [
		{'Name': 'Histology', 'ModelClass': PandasForest, 'kwargs': dict(max_depth=6, n_estimators=100, min_samples_leaf=5)},
		{'Name': 'KitchenSink','ModelClass': KFilteredForest, 'kwargs': dict(max_depth=6, 
																			 n_estimators=100, min_samples_leaf=5)},
	
		{'Name': 'Expression', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['Exp', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'Mutation', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['Dam', 'Other', 'Hot', 'OtherMut', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'Fusion', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['Fusion', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'GSEA', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['GSEA', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'CN', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['CN', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'Methylation', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['MethTSS', 'MethCpG', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'AllExpression', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['Exp', 'GSEA', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'AllGenomics', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['Dam', 'Hot', 'OtherMut', 'CN', 'MethCpG', 'MethTSS', 'Fusion', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'Mutation+Exp', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['Dam', 'Hot', 'OtherMut', 'Fusion', 'Exp', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'CN+Exp', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['CN', 'Exp', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'Methylation+Exp', 'ModelClass': KFilteredFeatureTypeForest, 'kwargs': dict(
			reserved_columns=constant_features, suffixes=['MethTSS', 'MethCpG', 'Exp', 'Cell'], 
			max_depth=6, n_estimators=100, min_samples_leaf=5
			)},
		{'Name': 'CFE','ModelClass': KFilteredForest, 'kwargs': dict(max_depth=6, 
																			 n_estimators=100, min_samples_leaf=5)},
	]

	Xtrain = []
	if "Histology" in included_models:
		Xtrain.append(X[constant_features])

	Xtrain = Xtrain + [X]*(len(models)-len(Xtrain)-1) 
	if 'CFE' in included_models:
		Xtrain = Xtrain + [cfe]
	if len(set(included_models) - set([v['Name'] for v in models])) > 0:
		raise ValueError("No model defined for the following models listed in included_models:\n%r" 
			%(set(included_models) - set([v['Name'] for v in models])))
	mask = [v['Name'] in included_models for v in models]
	models = [m for i, m in enumerate(models) if mask[i]]
	Xtrain = [x for i, x in enumerate(Xtrain) if mask[i]]
	assert len(Xtrain) == len(models), "number of models %i does not match number of feature sets %i" %(len(models), len(Xtrain))
	for i, x in enumerate(Xtrain):
		assert x.shape[1] > 0, "feature set %i does not have any columns" %i
		assert all(x.index == Y.index), "feature set %i index does not match Y index\n\n%r" %(i, x.iloc[:5, :5]) 

	columns = Y.columns[start_col : (start_col + n_col)]
	print('creating ensemble')
	ensemble = EnsembleRegressor(models, nfolds=nfolds, Splitter=KFold, scoring=pearson_score)

	output = os.path.join(directory, 'save%i_%i' %(start_col, min(start_col+n_col, Y.shape[1])))
	print('fitting ensemble to %i columns' %len(columns))
	ensemble.fit(Xtrain, Y, columns)
	print('saving results to %s' %output)
	ensemble.save_results(output)
