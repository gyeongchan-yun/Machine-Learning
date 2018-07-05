from sklearn import datasets

iris = datasets.load_iris()

# == Description
print(iris['DESCR'])

'''
=== iris Data ===

iris['data'][i]: ith iris data
iris['data'][i][j]: jth feature of ith iris data
'''
print("\niris data:\n", iris['data'])

print("\niris['target']:\n", iris['target'])

print("\niris['target_names']:\n", iris['target_names'])

print("\niris['feature_names']:\n", iris['feature_names'])