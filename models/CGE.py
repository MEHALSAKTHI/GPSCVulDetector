from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import BatchNormalization

from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn.metrics import classification_report


from sklearn.linear_model import LogisticRegression 


tf.compat.v1.set_random_seed(9906)
args = parameter_parser()

"""
The merged features (graph feature and pattern feature) are fed into the CGE model (Conv, Maxpooling, Dense)
"""


class CGEConv:
    def __init__(self, graph_train, graph_test, pattern_train, pattern_test, y_train, y_test, 
                 batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
#         print("Added Validation")
#         print("graph train: ")
#         print(type(graph_train))
#         print(graph_train)
#         print("graph test: ")
#         print(graph_test)
#         print("pattern train: ")
#         print(pattern_train)
#         print("pattern test: ")
#         print(pattern_test)
#         np.savetxt("graph_train.csv", graph_train,delimiter = ",", fmt ='%s',comments='')
#         np.savetxt("graph_test.csv", graph_test,delimiter = ",", fmt ='%s',comments='')
#         np.savetxt("pattern_train.csv", pattern_train,delimiter = ",", fmt ='%s',comments='')
#         np.savetxt("pattern_test.csv", pattern_test,delimiter = ",", fmt ='%s',comments='')
#         input1 = tf.keras.Input(shape=(1, 250), name='input1')
#         input2 = tf.keras.Input(shape=(3, 250), name='input2')        
        self.graph_train = graph_train
        self.graph_test = graph_test
        self.pattern_train = pattern_train
        self.pattern_test = pattern_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
        

        
        
#         # decode the results into a list of tuples (class, description, probability)
#         # (one such list for each sample in the batch)
#         print('Predicted:', decode_predictions(preds, top=3)[0])
# #         print(self.class_weight)
#         graph_train = tf.keras.layers.Conv1D(200, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')(
#             input1)
#         graph_train = tf.keras.layers.MaxPooling1D(pool_size=1, strides=1)(graph_train)
#         graph_train = tf.keras.layers.BatchNormalization()(graph_train)

#         pattern_train = tf.keras.layers.Conv1D(200, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')(
#             input2)
#         pattern_train = tf.keras.layers.MaxPooling1D(pool_size=3, strides=3)(pattern_train)
#         pattern_train = tf.keras.layers.BatchNormalization()(pattern_train)

#         mergevec = tf.keras.layers.Concatenate()([graph_train, pattern_train])
#         Dense1 = tf.keras.layers.Dense(100, activation='relu')(mergevec)
#         Dense2 = tf.keras.layers.Dense(50, activation='relu')(Dense1)
#         Dense3 = tf.keras.layers.Dense(10, activation='relu')(Dense2)
#         prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(Dense3)

#         model = tf.keras.Model(inputs=[input1, input2], outputs=[prediction])

#         model.summary()
#         adama = tf.keras.optimizers.Adam(0.0005)
# #         from tensorflow.keras.optimizers import Adam,Nadam, SGD
# #         optimizer = SGD(lr=0.1, decay=1e-6, momentum=0, nesterov=False)
#         model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
#         self.model = model

    """
    Training model
    """

    @staticmethod
    def unlevel(obj):
        while isinstance(obj, list) and len(obj) == 1:
            obj = obj[0]
        if isinstance(obj, list):
            return [unlevel(item) for item in obj]
        else:
            return obj

    def train(self):

        X_train1, X_val1, y_train, y_val = train_test_split(self.graph_train, self.y_train, test_size=0.10, random_state = np.random.randint(1,1000, 1)[0])
#         X_train2, X_val2, y_train, y_val = train_test_split(self.pattern_train, self.y_train, test_size=0.10, random_state = np.random.randint(1,1000, 1)[0])
        
        print("Log Reg")
        print("X_train1")
        X_train1=self.unlevel(X_train1)
        
        
        print(X_train1)
        print("y_train")
        print(y_train)
        
        logreg = LogisticRegression()
        logreg.fit(X_train1, y_train)
        score(logreg)
        
#         print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))
       
#         n = len(self.graph_train)
#         t_cnt = int(n*0.8)
#         X_val1 = self.graph_train[t_cnt:]
#         X_val2 = self.pattern_train[t_cnt:]
#         y_val = self.y_train[t_cnt:]
#         print("G Train size: ")
#         print(self.graph_train)
#         print(len(self.graph_train))
#         print("P Train size: ")
#         print(self.pattern_train)
#         print(len(self.pattern_train))
#         print([self.graph_train, self.pattern_train])
#         print("Val: ")
#         print([X_val1,X_val2])
#         self.model.fit([self.graph_train, self.pattern_train], self.y_train, validation_data= ([X_val1,X_val2],y_val), batch_size=self.batch_size,
#                        epochs=100)
                       
            #class_weight=self.class_weight)
        # self.model.save_weights("model.pkl")

    """
    Testing model
    """

    def test(self):
        # self.model.load_weights("_model.pkl")
        values = self.model.evaluate([self.graph_test, self.pattern_test], self.y_test, batch_size=self.batch_size,
                                     verbose=1)
        print("Loss: ", values[0], "Accuracy: ", values[1])
        predictions = (self.model.predict([self.graph_test, self.pattern_test], batch_size=self.batch_size).round())
        predictions = predictions.flatten()

        print(classification_report(self.y_test, predictions))
        
        
        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
        print('False positive rate(FPR): ', fp / (fp + tn))
        print('False negative rate(FNR): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall(TPR): ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
