from numpy.core.numeric import False_
import tensorflow as tf
import tensorflow_text  # Registers the ops for preprocessing
import tensorflow_hub as hub

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Dropout, Concatenate, Reshape


class TaggerModel(tf.keras.Model):
    def __init__(
        self, preprocessor_url, encoder_url, title_embed_url, num_labels, seq_length
    ):
        super(TaggerModel, self).__init__()
        # Preprocessing unit
        self.preprocessor = hub.load(preprocessor_url)
        self.preprocessing_layer = hub.KerasLayer(
            self.preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=seq_length),
            name="preprocessing",
        )
        # Encoder
        self.encoder = hub.KerasLayer(encoder_url, trainable=False, name="BERT_encoder")
        self.tokenize = hub.KerasLayer(self.preprocessor.tokenize, name="tokenize")

        # Title embedding
        self.title_embed = tf.keras.models.load_model(title_embed_url).get_layer(
            "Embedding"
        )
        self.embed_pool = Lambda(
            lambda x: K.mean(x, axis=1, keepdims=False), name="embed_avg_pooling"
        )
        # Hidden layers
        self.hidden1 = Dense(512, activation="relu")
        self.drop1 = Dropout(0.2)
        self.hidden2 = Dense(256, activation="relu")
        self.drop2 = Dropout(0.2)
        # Output layer
        self.output_layer = Dense(
            num_labels,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-2.0),
        )

    def call(self, inputs, training=False):
        text_input = inputs["synopsis"]
        # Squeeze the extra dim
        text_input = tf.squeeze(text_input)
        # print("text input: ", text_input.shape)
        # print(text_input)
        # Synopsis
        tokenized_inputs = [self.tokenize(text_input)]
        encoder_inputs = self.preprocessing_layer(tokenized_inputs)
        synopsis_outputs = self.encoder(encoder_inputs)
        synopsis_net = synopsis_outputs["pooled_output"]
        
        ######################################################################
        t_embed = self.title_embed(inputs["title"])
        t_embed = self.embed_pool(t_embed)
        output = Concatenate(axis=1)([synopsis_net, t_embed])
        output = self.output_layer(output)
        return output
        ######################################################################

        ######################################################################
        # output = self.output_layer(synopsis_net)
        # if training:
        #     return output
        # else:
        #     # Title
        #     t_embed = self.title_embed(inputs["title"])
        #     t_embed = self.embed_pool(t_embed)
        #     return Concatenate(axis=1)([output, t_embed])
        ######################################################################

    def model(self, inputs):
        return tf.keras.Model(inputs, self.call(inputs))