# Wayne Nixalo - 2017-Jun-04 17:02
# testing usage of TPOT -- api for genetic-algorithmic Machine Learning model
# selection and hyperparameter tuning. Basically I want the machine to
# figure out how to learn a thing too.

# From: http://rhiever.github.io/tpot/using/

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                train_size=0.75, test_size=0.25)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
