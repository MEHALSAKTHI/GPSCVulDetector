from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix

tf.compat.v1.set_random_seed(9906)
args = parameter_parser()

"""
The merged features (graph feature and pattern feature) are fed into the CGE model (Conv, Maxpooling, Dense)
"""


class CGEConv:
    def __init__(self, graph_train, graph_test, pattern_train, pattern_test, y_train, y_test,
                 batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
        input1 = tf.keras.Input(shape=(1, 250), name='input1')
        input2 = tf.keras.Input(shape=(3, 250), name='input2')

        self.graph_train = graph_train
        self.graph_test = graph_test
        self.pattern_train = pattern_train
        self.pattern_test = pattern_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
        print(self.class_weight)
        graph_train = tf.keras.layers.Conv1D(200, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')(
            input1)
        graph_train = tf.keras.layers.MaxPooling1D(pool_size=1, strides=1)(graph_train)

        pattern_train = tf.keras.layers.Conv1D(200, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')(
            input2)
        pattern_train = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1)(pattern_train)

        mergevec = tf.keras.layers.Concatenate()([graph_train, pattern_train])
        Dense1 = tf.keras.layers.Dense(100, activation='relu')(mergevec)
        Dense2 = tf.keras.layers.Dense(50, activation='relu')(Dense1)
        Dense3 = tf.keras.layers.Dense(10, activation='relu')(Dense2)
        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(Dense3)

        model = tf.keras.Model(inputs=[input1, input2], outputs=[prediction])

        model.summary()
        adama = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Training model
    """

    def train(self):
        self.model.fit([self.graph_train, self.pattern_train], self.y_train, batch_size=self.batch_size,
                       epochs=self.epochs)
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

        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
        print('False positive rate(FPR): ', fp / (fp + tn))
        print('False negative rate(FNR): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall(TPR): ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
