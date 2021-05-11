from numpy.core.numeric import False_
import tensorflow as tf
import tensorflow_text  # Registers the ops for preprocessing
import tensorflow_hub as hub

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Dropout, Concatenate, Reshape


class TaggerModel(tf.keras.Model):
    def __init__(
        self, num_labels, seq_length, mode="concat_before"
    ):
        super(TaggerModel, self).__init__()
        self.mode = mode
        # Sentence encoders
        self.encoder1 = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/2", trainable=False, name="universal_sentence_encoder")
        self.encoder2 = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2", trainable=False, name="nnlm")

        # Hidden layers
        self.hidden1 = Dense(512, activation="elu")
        self.drop1 = Dropout(0.1)
        # self.hidden2 = Dense(256, activation="relu")
        # self.drop2 = Dropout(0.2)
        # Output layer
        self.output_layer = Dense(
            num_labels,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-2.0), # for focal loss
        )

    def call(self, inputs, training=False):
        text_input = inputs["synopsis"]
        keyword_input = inputs["keywords"]
        # Squeeze the extra dim
        text_input = tf.squeeze(text_input)
        keyword_input = tf.squeeze(keyword_input)
        # Feeding into encoder
        synopsis_net1 = self.encoder1(text_input)["outputs"]
        synopsis_net2 = self.encoder2(text_input)
        keyword_net1 = self.encoder1(keyword_input)["outputs"]
        keyword_net2 = self.encoder2(keyword_input)

        # Outputs
        output = Concatenate(axis=1)([synopsis_net1, synopsis_net2, 
                                      keyword_net1, keyword_net2])
        output = self.hidden1(output)
        output = self.drop1(output, training=training)
        output = self.output_layer(output)

        if self.mode == "concat_before":
            return output
        elif self.mode == "concat_after":
            if training:
                return output
            else:
                return Concatenate(axis=1)([output, keyword_net2])
        else:
            raise(NotImplementedError(f"Unrecognized mode: {self.mode}"))


    def model(self, inputs):
        return tf.keras.Model(inputs, self.call(inputs))