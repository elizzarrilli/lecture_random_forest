import pandas
import kfold_template

from sklearn import tree


dataset = pandas.read_csv("temperature_data.csv")
#print(dataset)

dataset = pandas.get_dummies(dataset)
#print(dataset)

dataset = dataset.sample(frac=1).reset_index()
#print(dataset)

target = dataset['actual'].values

data = dataset.drop(["actual","level_0"], axis=1)
## keep column names before turning from mframe to matrix to maintain column names
feature_list = data.columns
data = data.values

print(target)
print(data)


## run a decision tree, max_depth = how many levels the tree can take (start max big and go down if unsure)
machine = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=3) #can also use 'entropy'
return_values = kfold_template.run_kfold(machine,data,target,4,True)
print(return_values)


machine = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=3) #can also use 'entropy'
machine.fit(data, target)
feature_importances_raw = machine.feature_importances_
print(feature_importances_raw)
print(feature_list)

feature_zip = zip(feature_list, feature_importances_raw)
print(feature_zip)

feature_importances = [(feature, round(importance,4)) for feature, importance in feature_zip]
sorted(feature_importances, key = lambda x: x[1]) #lambda is a nameless function 
print(feature_importances)
[print('{:13}: {}'.format(*feature_importance)) for feature_importance in feature_importances]
## print but always give first entry 13 spaces, colono, space when printing












