
from sklearn.preprocessing import StandardScaler
from locations import dataDir


def SaveToHDF(train, test, filename):
	train.to_hdf('%s%s.hdf5'%(dataDir,filename), 'train')
	test.to_hdf('%s%s.hdf5'%(dataDir, filename), 'test')


def Scale(train, test, lables): 

	data_train = train.loc[:,lables]
	data_test = test.loc[:,lables]

	scaler = StandardScaler().fit(data_train.values)

	data_train = scaler.transform(data_train.values)
	data_test = scaler.transform(data_test.values)

	train[lables] = data_train
	test[lables] = data_test


	return train, test