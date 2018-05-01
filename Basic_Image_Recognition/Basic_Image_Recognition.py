
from PIL import Image
import glob
import  numpy as np, time
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

direction_encode = {'right': 0, 'left': 1, 'up': 2, 'straight': 3}
emotion_encode = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3}


def data_prepare(directory):
    X_temp = {} ; y_direction_temp = {} ; y_emotion_temp = {} ; y_eyewear_temp = {}
    c = 0
    #####################################################################

    for image in glob.glob(directory):
        image_name = image.split("_")
        direction = image_name[1]; emotion= image_name[2] ; eyewear = image_name[3]
        image_array = np.array(Image.open(image).convert('L')).flatten()

        X_temp[c] = image_array
        y_direction_temp[c] = direction; y_emotion_temp[c] = emotion ; y_eyewear_temp[c] = eyewear
        c += 1

    ####################################################################

    X = np.zeros((len(X_temp), len(X_temp[0])))
    y_direction = np.zeros(len(y_direction_temp))

    ############################################

    y_emotion = np.zeros(len(y_emotion_temp))
    y_eyewear = np.zeros(len(y_eyewear_temp))

    for key in range(len(X_temp)):
        X[key] = X_temp[key]
        y_direction[key] = direction_encode[y_direction_temp[key]]
        y_emotion[key] = emotion_encode[y_emotion_temp[key]]

    return [X, y_direction, y_emotion]

X_train =data_prepare('TrainingSet\*.jpg')[0]
y_train_direction = data_prepare('TrainingSet\*.jpg')[1]
y_train_emotion = data_prepare('TrainingSet\*.jpg')[2]

X_test = data_prepare('TestSet\*.jpg')[0]
y_test_direction= data_prepare('TestSet\*.jpg')[1]
y_test_emotion = data_prepare('TestSet\*.jpg')[2]


def partA():

    clf = RandomForestClassifier(random_state=0)
    beg = time.time()
    clf.fit(X_train, y_train_direction)
    total_time = time.time()-beg

    preds_direction = clf.predict(X_test)
    accuracy1 = accuracy_score(y_test_direction, preds_direction)
    part_a = [clf, total_time, accuracy1]

    joblib.dump(part_a, "part_a.pkl", protocol = 2)
    # print part_a

###################################################################
def partB():

    pca = PCA(random_state=0, n_components=0.95)
    pca.fit(X_train)

    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)

    clf_reduced = RandomForestClassifier(random_state=0)
    start = time.time()
    clf_reduced.fit(X_train_reduced, y_train_direction)
    total_time_pca = time.time()-start

    preds_direction_reduced = clf_reduced.predict(X_test_reduced)
    accuracy2 = accuracy_score(y_test_direction, preds_direction_reduced)

    part_b = [clf_reduced, total_time_pca, accuracy2]
    joblib.dump(part_b, "part_b.pkl", protocol = 2)
    # print part_b
    return [X_train_reduced,X_test_reduced]

#################################################################
def partC():

    lr = LogisticRegression(random_state=0, multi_class="multinomial", solver="lbfgs")
    beg = time.time()
    lr.fit(X_train, y_train_emotion)
    total_time_lr = time.time() - beg

    preds_emotion = lr.predict(X_test)
    accuracy3 = accuracy_score(y_test_emotion, preds_emotion)

    part_c = [lr, total_time_lr, accuracy3]
    joblib.dump(part_c, "part_c.pkl", protocol = 2)
    # print part_c
###############################################################
def partD():
    X_train_reduced = partB()[0];X_test_reduced = partB()[1]
    lr_reduced = LogisticRegression(random_state=0, multi_class="multinomial", solver="lbfgs")
    beg = time.time()
    lr_reduced.fit(X_train_reduced, y_train_emotion)
    total_time_lr_reduced = time.time() - beg

    preds_emotion_reduced = lr_reduced.predict(X_test_reduced)
    accuracy4 = accuracy_score(y_test_emotion, preds_emotion_reduced)

    part_d = [lr_reduced, total_time_lr_reduced, accuracy4]
    joblib.dump(part_d, "part_d.pkl", protocol = 2)
    # print part_d

partA()
partB()
partC()
partD()