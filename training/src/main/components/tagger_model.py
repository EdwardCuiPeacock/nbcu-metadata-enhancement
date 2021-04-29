from numpy.core.numeric import False_
import tensorflow as tf
import tensorflow_text  # Registers the ops for preprocessing
import tensorflow_hub as hub

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Dropout, Concatenate, Reshape


class TaggerModel(tf.keras.Model):
    def __init__(
        self, preprocessor_url, encoder_url, token_embed_url, num_labels, seq_length
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

        # Token embedding
        self.token_embed = tf.keras.models.load_model(token_embed_url).get_layer(
            "Embedding"
        )
        # TODO: does this work with variable length inputs?
        self.embed_pool = Lambda(
            lambda x: K.mean(x, axis=1, keepdims=False), name="embed_avg_pooling"
        )
        # Outputs
        self.hidden1 = Dense(512, activation="relu")
        self.drop1 = Dropout(0.2)
        self.hidden2 = Dense(256, activation="relu")
        self.drop2 = Dropout(0.2)
        self.output_layer = Dense(num_labels, activation="sigmoid")

    def call(self, inputs, training=False):
        text_input, tokens = inputs["synopsis"], inputs["tokens"]
        keywords = inputs["keywords"]
        # Squeeze the extra dim
        text_input = tf.squeeze(text_input)
        #print("text input: ", text_input.shape)
        #print(text_input)
        # Convert tokens to ragged tensor
        tokens = tf.ragged.boolean_mask(tokens, tokens > -1)
        keywords = tf.ragged.boolean_mask(keywords, tokens > -1)
        # Synopsis
        tokenized_inputs = [self.tokenize(text_input)]
        encoder_inputs = self.preprocessing_layer(tokenized_inputs)
        synopsis_outputs = self.encoder(encoder_inputs)
        synopsis_net = synopsis_outputs["pooled_output"]
        # Tokens
        t_embed = self.token_embed(tokens)
        # Pool the embed
        t_embed = self.embed_pool(t_embed)
        # Keywords
        k_embed = self.token_embed(keywords)
        k_embed = self.embed_pool(k_embed)
        # Concatenate
        output = Concatenate(axis=1)([synopsis_net, t_embed, k_embed]) # 
        # Pass through the dense layers
        # output = self.hidden1(output)
        # if training:
        #     output = self.drop1(output, training=training)
        # output = self.hidden2(output)
        # if training:
        #     output = self.drop2(output, training=training)
        output = self.output_layer(output)

        return output

    def model(self, inputs):
        return tf.keras.Model(inputs, self.call(inputs))