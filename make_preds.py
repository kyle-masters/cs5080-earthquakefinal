from pyts.multivariate.transformation import WEASELMUSE
from pyts.transformation import BagOfPatterns
from pyts.classification import BOSSVS, SAXVSM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Make predictions using BOP with random forest
def pred_bop(x_train, y_train, x_test, y_test, n_bins, window_size, word_size):
    bop = BagOfPatterns(word_size=word_size, n_bins=n_bins, window_size=window_size, strategy='uniform')
    train_features = bop.fit_transform(x_train)
    test_features = bop.transform(x_test)

    forest = RandomForestClassifier()
    forest.fit(train_features, y_train)
    preds = forest.predict(test_features)

    pos_rate = preds.sum()/preds.shape[0]
    neg_rate = (preds.shape[0] - preds.sum())/preds.shape[0]
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    return [accuracy, recall, precision, f1, pos_rate, neg_rate]

# Make predictions using BOSSVS
def pred_bossvs(x_train, y_train, x_test, y_test, n_bins, window_size, word_size):
    boss = BOSSVS(word_size=word_size,n_bins=n_bins, window_size=window_size, strategy='uniform')

    boss.fit(x_train, y_train)
    preds = boss.predict(x_test)

    pos_rate = preds.sum()/preds.shape[0]
    neg_rate = (preds.shape[0] - preds.sum())/preds.shape[0]
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    return [accuracy, recall, precision, f1, pos_rate, neg_rate]


# Make predictions using SAXVSM
def pred_saxvsm(x_train, y_train, x_test, y_test, n_bins, window_size,  word_size):
    clf = SAXVSM(word_size=word_size,n_bins=n_bins, window_size=window_size, strategy='uniform')

    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    pos_rate = preds.sum()/preds.shape[0]
    neg_rate = (preds.shape[0] - preds.sum())/preds.shape[0]
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    return [accuracy, recall, precision, f1, pos_rate, neg_rate]


def predict(x_train, y_train, x_test, y_test, n_bins, window_size, word_size):

    # If the chosen hyperparameters aren't legal, don't run them.
    if word_size > window_size * x_train.shape[1]:
        return []

    bop = pred_bop(x_train, y_train, x_test, y_test, n_bins, window_size,  word_size)
    boss = pred_bossvs(x_train, y_train, x_test, y_test, n_bins, window_size, word_size)
    saxvsm = pred_saxvsm(x_train, y_train, x_test, y_test, n_bins, window_size,  word_size)

    return bop + boss + saxvsm
