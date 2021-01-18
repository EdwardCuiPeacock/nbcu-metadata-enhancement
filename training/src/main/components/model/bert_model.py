import os
import json

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Input,
    LSTM,
    Dropout,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.models import Model, Sequential

import tensorflow_addons as tfa

import transformers as hf

from model.utils import smooth_curve, get_model_path


class EmbeddingModel:
    """This is the base class encapsulating TF models with extra functionality
    to save and load models and plot training history
    Should not be instantiated directly, embedding model classes specific to
    each type of modelization should be used instead.

    Subclasses should implement the `setup_text_part` method (see below for
    examples)
    """

    def __init__(
        self,
        n_tags=2,
        optimizer_info={},
        metric_info={},
        data_name="",
        tag_index={},
        name="",
    ):
        """
        Arguments:
            n_tags - int - the number of distinct tags
            optimizer_info - dict - optimizer parameters
            metric_info - dict - metric parameters
            data_name - string - the data the model was trained on (Merlin,
                TMdb...)
            name - string - model name
        """
        self.n_tags = n_tags
        self.optimizer_info = optimizer_info
        self.metric_info = metric_info
        self.data_name = data_name
        self.tag_index = tag_index
        self.name = name

    def from_config(self, config):
        """Refreshes the object's initialization attributes based on a
        configuration dictionary

        Argument:
            config - dict - configuration dictionary
        """
        self.n_tags = config["n_tags"]
        self.optimizer_info = config["optimizer_info"]
        self.metric_info = config["metric_info"]
        self.data_name = config["data_name"]
        self.tag_index = config["tag_index"]

    def get_config(self):
        """Generates a configuration dictionary based on the instance's
        attributes

        Returns:
             - dict - the configuration dictionary
        """
        return {
            "name": self.name,
            "n_tags": self.n_tags,
            "optimizer_info": self.optimizer_info,
            "metric_info": self.metric_info,
            "data_name": self.data_name,
            "tag_index": self.tag_index,
        }

    def setup(self):
        """This setups the model architecture with
            - a branch that handles text data
            - the final dense layer with a sigmoid activation for multi-label
            classification
        and assigns the resulting tf.keras.Model to `self.model`
        """
        inputs, intermediate = self.setup_text_part()
        output = Dense(self.n_tags, activation="sigmoid")(intermediate)
        model = Model(inputs=inputs, outputs=output)
        self.model = model

    def summary(self):
        """Encapsulates `self.model`'s summary method"""
        self.model.summary()

    def compile(self):
        """Creates the optimizer, metric from the optimizer and metric info and
        compiles the model using them
        """
        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_info)
        if self.metric_info["name"] == "macro_f1":
            self.metric = tfa.metrics.F1Score(
                num_classes=self.n_tags, **self.metric_info
            )
        else:
            self.metric = "accuracy"
        self.model.compile(
            optimizer=self.optimizer, loss="binary_crossentropy", metrics=[self.metric]
        )

    def fit(self, train_ds, test_ds, epochs=10, steps_per_epoch=None, callbacks=None):
        """Fits the encapsulated TF model on training data, validating on
        `test_ds` and stores the training history as an object attribute

        Arguments:
            train_ds - tf.data.Dataset - the training dataset
            test_ds - tf.data.Dataset - the testing dataset
            epochs - int - number of epochs to train on
            steps_per_epoch - None or int - the number of batchs to go through
                in an epoch, if None, will go through all batches in `train_ds`
                each epoch

        Returns:
            history - tf.keras.callbacks.Callback - training history
        """
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=test_ds,
        )
        self.history = history
        return history

    def plot_history(self, smooth=False):
        """Plots the history of a fitted model (loss and metric) with optional
        smoothing

        Argument:
            smooth - boolean - whether to smooth the metric and loss
        """
        loss_history = self.history.history["loss"]
        metric_history = self.history.history[self.metric_info["name"]]
        val_loss_history = self.history.history["val_loss"]
        val_metric_history = self.history.history[
            "val_{}".format(self.metric_info["name"])
        ]

        if smooth:
            loss_history = smooth_curve(loss_history)
            metric_history = smooth_curve(metric_history)
            val_loss_history = smooth_curve(val_loss_history)
            val_metric_history = smooth_curve(val_metric_history)

        # Plot training & validation macro F1
        plt.plot(metric_history)
        plt.plot(val_metric_history)
        plt.title("Model {}".format(self.metric_info["name"]))
        plt.ylabel(self.metric_info["name"])
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()

        # Plot training & validation loss values
        plt.plot(loss_history)
        plt.plot(val_loss_history)
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()

    def predict(self, texts, use_cache=False, train=False, most_recent=True):
        """Predicts the `texts`' tags using the encapsulated model

        Argument:
            texts - tf.tensor - pre-processed text tokens
            use_cache - boolean - whether to try and fetch test predictions from
                the cache and save there if no file is found

        Returns:
             - tf.tensor - model predictions
        """
        if use_cache:
            model_path = self._get_model_path(most_recent)
            cache_file = os.path.join(model_path, "predictions")
            if not os.path.isdir(cache_file):
                os.mkdir(cache_file)
            cache_file = os.path.join(cache_file, "train" if train else "test")
            if not os.path.isdir(cache_file):
                os.mkdir(cache_file)
            cache_file = os.path.join(cache_file, "preds.npy")
            if os.path.exists(cache_file):
                return tf.constant(np.load(cache_file))
            predictions = self.model.predict(texts)
            np.save(cache_file, predictions)
            return predictions
        return self.model.predict(texts)

    def evaluate(self, texts, tags, verbose=1):
        """Evaluates the `texts`' predictions using the encapsulated model

        Argument:
            texts - tf.tensor - pre-processed text tokens
            tags - np.array - one-hot-encoded tags
            verbose - int - if 0 silent, if 1 progress bar

        Returns:
            - list - values of loss and metric computed on predictions for
                `texts` based on their ground-truth `tags`
        """
        return self.model.evaluate(texts, tags, verbose=verbose)

    def _get_model_path(self, most_recent):
        root = os.path.join("..", "models", self.data_name)
        if not os.path.isdir(root):
            os.mkdir(root)
        if not os.path.isdir(os.path.join(root, self.name)):
            os.mkdir(os.path.join(root, self.name))
        return get_model_path(
            os.path.join("..", "models", self.data_name, self.name), most_recent
        )

    def save(self, most_recent):
        """Saves a model in two files: one TF checkpoint file containing the
        encapsulated model's weights and a json file with the model
        configuration obtained by calling `self.get_config()`

        Argument:
            model_path - string - the root path where to save the model files
                the weights will be saved in the `weights` subdirectory and the
                configuration file will be saved in the `config` subdirectory
        """
        model_path = self._get_model_path(most_recent)
        config_dir = os.path.join(model_path, "config")
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        if not os.path.isdir(config_dir):
            os.mkdir(config_dir)
        self.model.save_weights(os.path.join(model_path, "weights", "checkpoint"))
        with open(os.path.join(config_dir, "model_config.json"), "w") as file:
            json.dump(self.get_config(), file)

    def save_to_path(self, model_path):
        """Saves a model in two files: one TF checkpoint file containing the
        encapsulated model's weights and a json file with the model
        configuration obtained by calling `self.get_config()` with an explicit path

        Argument:
            model_path - string - the root path where to save the model files
                the weights will be saved in the `weights` subdirectory and the
                configuration file will be saved in the `config` subdirectory
        """

        self.model.save_weights(os.path.join(model_path, "weights", "checkpoint"))

        if not os.path.isdir("temp"):
            os.mkdir("temp")
        with open(os.path.join("temp", "model_config.json"), "w") as file:
            json.dump(self.get_config(), file)

        tf.io.gfile.copy(
            os.path.join("temp", "model_config.json"),
            os.path.join(model_path, "config", "model_config.json"),
        )

    def load(self, most_recent):
        """Load a model from its root folder by loading the configuration file
        and refreshing the instances attributes based on this file, setting up
        the model, compiling the model and finally loading the weights from
        the `weights` subdirectory

        Argument:
            model_path - string - the root path where the model files have been
                written (the weights in the `weights` subdirectory and the
                configuration file in the `config` subdirectory

        Returns:
            self - EmbeddingModel
        """
        model_path = self._get_model_path(most_recent)
        config_path = os.path.join(model_path, "config", "model_config.json")
        with open(config_path) as file:
            config = json.load(file)
        self.from_config(config)
        self.setup()
        # self.compile()
        self.model.load_weights(os.path.join(model_path, "weights", "checkpoint"))
        return self

    def load_from_path(self, model_path):
        """Load a model from its root folder by loading the configuration file
        and refreshing the instances attributes based on this file, setting up
        the model, compiling the model and finally loading the weights from
        the `weights` subdirectory

        Argument:
            model_path - string - the root path where the model files have been
                written (the weights in the `weights` subdirectory and the
                configuration file in the `config` subdirectory

        Returns:
            self - EmbeddingModel
        """

        if not os.path.isdir("temp"):
            os.mkdir("temp")
        tf.io.gfile.copy(
            os.path.join(model_path, "config", "model_config.json"),
            os.path.join("temp", "model_config.json"),
        )
        with open("temp/model_config.json") as file:
            print("Loading from config...")
            config = json.load(file)
        self.from_config(config)
        self.setup()
        # self.compile()
        print("Loading weights...")
        self.model.load_weights(os.path.join(model_path, "weights", "checkpoint"))

        return self


class EmbeddingModelWithKeywords(EmbeddingModel):
    """This is the base class encapsulating TF models with extra functionality
    to save and load models and plot training history
    Should not be instantiated directly, embedding model classes specific to
    each type of modelization should be used instead.

    Subclasses should implement the `setup_text_part` and `setup_wide_part`
    methods (see below for examples)
    """

    def __init__(
        self,
        n_tags=2,
        optimizer_info={},
        metric_info={},
        data_name="",
        tag_index={},
        keyword_index={},
        n_keywords=0,
        name="",
    ):
        """
        Arguments:
            n_tags - int - the number of distinct tags
            optimizer_info - dict - optimizer parameters
            metric_info - dict - metric parameters
            data_name - string - the data the model was trained on (Merlin,
                TMdb...)
            n_keywords - int - if > 0 number of keywords used to build a wide
                part to the model to take in one-hot-encoded features along the
                text data
            name - string - model name
        """
        EmbeddingModel.__init__(
            self, n_tags, optimizer_info, metric_info, data_name, tag_index, name
        )
        self.keyword_index = keyword_index
        self.n_keywords = n_keywords

    def from_config(self, config):
        """Refreshes the object's initialization attributes based on a
        configuration dictionary

        Argument:
            config - dict - configuration dictionary
        """
        super(EmbeddingModelWithKeywords, self).from_config(config)
        self.n_keywords = config["n_keywords"]
        self.keyword_index = config["keyword_index"]

    def get_config(self):
        """Generates a configuration dictionary based on the instance's
        attributes

        Returns:
             - dict - the configuration dictionary
        """
        return {
            **super(EmbeddingModelWithKeywords, self).get_config(),
            "n_keywords": self.n_keywords,
            "keyword_index": self.keyword_index,
        }

    def setup(self):
        """This setups the model architecture with
            - a branch that handles text data
            - a branch that handles wide one-hot-encoded input
            - the concatenation of the outputs of both branches
            - the final dense layer with a sigmoid activation for multi-label
            classification
        and assigns the resulting tf.keras.Model to `self.model`
        """
        text_inputs, text_intermediate = self.setup_text_part()
        wide_inputs, wide_intermediate = self.setup_wide_part()
        intermediate = tf.keras.layers.concatenate(
            [text_intermediate, wide_intermediate]
        )
        output = Dense(self.n_tags, activation="sigmoid")(intermediate)
        model = Model(inputs=[text_inputs, wide_inputs], outputs=output)
        self.model = model

    def predict(self, texts, keywords, use_cache=False, train=False, most_recent=True):
        """Predicts the `texts`' tags using the encapsulated model

        Argument:
            texts - tf.tensor - pre-processed text tokens
            keywords - tf.tensor - one-hot-encoded keywords
            use_cache - boolean - whether to try and fetch test predictions from
                the cache and save there if no file is found

        Returns:
             - tf.tensor - model predictions
        """
        if use_cache:
            model_path = self._get_model_path(most_recent)
            cache_file = os.path.join(model_path, "predictions")
            if not os.path.isdir(cache_file):
                os.mkdir(cache_file)
            cache_file = os.path.join(cache_file, "train" if train else "test")
            if not os.path.isdir(cache_file):
                os.mkdir(cache_file)
            cache_file = os.path.join(cache_file, "preds.npy")
            if os.path.exists(cache_file):
                return np.load(cache_file)
            predictions = self.model.predict([texts, keywords])
            np.save(cache_file, predictions)
            return predictions
        return self.model.predict([texts, keywords])

    def evaluate(self, texts, tags, keywords, verbose=1):
        """Evaluates the `texts`' predictions using the encapsulated model

        Argument:
            texts - tf.tensor - pre-processed text tokens
            tags - np.array - one-hot-encoded tags
            keywords - tf.tensor - one-hot-encoded keywords
            verbose - int - if 0 silent, if 1 progress bar

        Returns:
            - list - values of loss and metric computed on predictions for
                `texts` based on their ground-truth `tags`
        """
        return self.model.evaluate([texts, keywords], tags, verbose=verbose)


class LSTMEmbeddingModel(EmbeddingModel):
    """This is the EmbeddingModel class for the custom recurrent model"""

    def __init__(
        self,
        max_seq_len=10,
        vocab_size=2000,
        n_tags=2,
        optimizer_info={},
        metric_info={},
        data_name="",
        tag_index={},
    ):
        """
        Arguments:
            max_seq_len - int - the fixed length of token lists to be passed to
                the model after padding/truncating
            vocab_size - int - the number of different tokenized words
            n_tags - int - the number of distinct tags
            optimizer_info - dict - optimizer parameters
            metric_info - dict - metric parameters
            data_name - string - the data the model was trained on (Merlin,
                TMdb...)
        """
        super(LSTMEmbeddingModel, self).__init__(
            n_tags, optimizer_info, metric_info, data_name, tag_index, "lstm"
        )
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def from_config(self, config):
        """Refreshes the object's initialization attributes based on a
        configuration dictionary

        Argument:
            config - dict - configuration dictionary
        """
        super(LSTMEmbeddingModel, self).from_config(config)
        self.max_seq_len = config["max_seq_len"]
        self.vocab_size = config["vocab_size"]

    def get_config(self):
        """Generates a configuration dictionary based on the instance's
        attributes

        Returns:
             - dict - the configuration dictionary
        """
        return {
            **super(LSTMEmbeddingModel, self).get_config(),
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
        }

    def setup_text_part(self):
        """Sets up the encapsulated TF model pipeline for the branch that
        handles the text data and returns the inpus and output of the pipeline

        Returns:
            synopsis_input - tf.keras.Input - the text input
            encoded_synopsis - tf.Tensor - the output of the text layers
        """
        synopsis_input = Input(shape=(self.max_seq_len,), dtype="int32")
        embedded_synopsis = Embedding(
            input_dim=self.vocab_size + 1,
            output_dim=32,
            input_length=self.max_seq_len,
            mask_zero=True,
            embeddings_initializer="glorot_normal",
        )(synopsis_input)
        encoded_synopsis = LSTM(16, dropout=0.3)(embedded_synopsis)
        return synopsis_input, encoded_synopsis


class BertEmbeddingModel(EmbeddingModel):
    """This is the EmbeddingModel class for the pretrained fine_tuned BERT
    model
    """

    def __init__(
        self,
        max_seq_len=10,
        n_tags=2,
        optimizer_info={},
        metric_info={},
        data_name="",
        tag_index={},
    ):
        """
        Arguments:
            max_seq_len - int - the fixed length of token lists to be passed to
                the model after padding/truncating
            n_tags - int - the number of distinct tags
            optimizer_info - dict - optimizer parameters
            metric_info - dict - metric parameters
            data_name - string - the data the model was trained on (Merlin,
                TMdb...)
        """
        super(BertEmbeddingModel, self).__init__(
            n_tags, optimizer_info, metric_info, data_name, tag_index, "bert_full"
        )
        self.max_seq_len = max_seq_len

    def from_config(self, config):
        """Refreshes the object's initialization attributes based on a
        configuration dictionary

        Argument:
            config - dict - configuration dictionary
        """
        super(BertEmbeddingModel, self).from_config(config)
        self.max_seq_len = config["max_seq_len"]

    def get_config(self):
        """Generates a configuration dictionary based on the instance's
        attributes

        Returns:
             - dict - the configuration dictionary
        """
        return {
            **super(BertEmbeddingModel, self).get_config(),
            "max_seq_len": self.max_seq_len,
        }

    def setup_text_part(self):
        """Sets up the encapsulated TF model pipeline for the branch that
        handles the text data and returns the inpus and output of the pipeline

        Returns:
            synopsis_input - tf.keras.Input - the text input
            encoded_synopsis - tf.Tensor - the output of the text layers
        """
        synopsis_input = Input(self.max_seq_len, dtype="int32", name="tokens")
        encoded_synopsis = hf.TFDistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        ).distilbert(synopsis_input)[0][:, 0, :]
        return synopsis_input, encoded_synopsis


class DenseHead(EmbeddingModel):
    """This is the EmbeddingModel class for the dense head used on top of the
    frozen pretrained Bert model
    """

    def __init__(
        self,
        embedding_size=768,
        n_parts=128,
        n_tags=2,
        optimizer_info={},
        metric_info={},
        data_name="",
        tag_index={},
    ):
        """
        Arguments:
            embeddings_size - int - the input size of the dense head
            n_parts - int - how many parts to break the data into whenever
                passing in through `self.bert` (for resource allocation)
            n_tags - int - the number of distinct tags
            optimizer_info - dict - optimizer parameters
            metric_info - dict - metric parameters
            data_name - string - the data the model was trained on (Merlin,
                TMdb...)
        """
        super(DenseHead, self).__init__(
            n_tags, optimizer_info, metric_info, data_name, tag_index, "bert_frozen"
        )
        self.embedding_size = embedding_size
        self.bert = hf.TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        self.n_parts = n_parts

    def from_config(self, config):
        """Refreshes the object's initialization attributes based on a
        configuration dictionary

        Argument:
            config - dict - configuration dictionary
        """
        super(DenseHead, self).from_config(config)
        self.embedding_size = config["embedding_size"]

    def get_config(self):
        """Generates a configuration dictionary based on the instance's
        attributes

        Returns:
             - dict - the configuration dictionary
        """
        return {
            **super(DenseHead, self).get_config(),
            "embedding_size": self.embedding_size,
        }

    def setup_text_part(self):
        """Sets up the encapsulated TF model pipeline for the branch that
        handles the text data and returns the inpus and output of the pipeline

        Returns:
            synopsis_input - tf.keras.Input - the text input
            encoded_synopsis - tf.Tensor - the output of the text layers
        """
        synopsis_input = Input(shape=(self.embedding_size,))
        hidden = Dense(32, activation="relu")(synopsis_input)
        normed = BatchNormalization()(hidden)
        encoded_synopsis = Dropout(0.5)(normed)
        return synopsis_input, encoded_synopsis

    def _generate_bert_embeddings(self, tokens):
        """Passes encoded tokens through the pretrained Distilbert model and
        returns the embedded tokens

        Arguments:
            tokens - list - list of encoded tokens

        Returns:
            embedded_tokens - tf.Tensor - tensor of embedded tokens
        """
        n = tokens.shape[0]

        embedded_tokens_parts = []
        for i in range(self.n_parts):
            # we retrieve only DistilBert's output for the [CLS] token
            embedded_tokens_parts.append(
                self.bert(
                    tokens[i * (n // self.n_parts) : (i + 1) * (n // self.n_parts)]
                )[0][:, 0, :]
            )
        # append the remaining part (not exact divisibility)
        embedded_tokens_parts.append(
            self.bert(tokens[self.n_parts * (n // self.n_parts) :])[0][:, 0, :]
        )

        embedded_tokens = tf.concat(embedded_tokens_parts, axis=0)

        return embedded_tokens

    def generate_bert_embeddings(
        self, tokens, use_cache=False, train=True, most_recent=True
    ):
        """Passes encoded tokens through the pretrained Distilbert model and
        returns the embedded tokens

        Arguments:
            tokens - list - list of encoded tokens
            use_cache - boolean - whether to use the cache to fetch or save
                the embeddings
            train - boolean - if True will fetch data from the train folder in
                the cache, if False will fetch test data

        Returns:
            embedded_tokens - tf.Tensor - tensor of embedded tokens
        """
        if use_cache:
            model_path = self._get_model_path(most_recent)
            if not os.path.isdir(model_path):
                os.mkdir(model_path)
            cache_file = os.path.join(model_path, "bert_embeddings")
            if not os.path.isdir(cache_file):
                os.mkdir(cache_file)
            cache_file = os.path.join(cache_file, "train" if train else "test")
            if not os.path.isdir(cache_file):
                os.mkdir(cache_file)
            cache_file = os.path.join(cache_file, "embeddings.npy")
            if os.path.exists(cache_file):
                return tf.constant(np.load(cache_file))
            embeddings = self._generate_bert_embeddings(tokens)
            np.save(cache_file, embeddings.numpy())
            return embeddings
        return self._generate_bert_embeddings(tokens)

    def predict(self, texts):
        """Predicts the `texts`' tags using the encapsulated model by passing
        them through `self.bert` first

        Argument:
            texts - tf.tensor - pre-processed text tokens

        Returns:
             - tf.tensor - model predictions
        """
        embedded_texts = self.generate_bert_embeddings(texts, use_cache=False)
        return super(DenseHead, self).predict(embedded_texts)

    def evaluate(self, texts, tags, verbose=1):
        """Evaluates the `texts`' predictions using the encapsulated model

        Argument:
            texts - tf.tensor - pre-processed text tokens
            tags - np.array - one-hot-encoded tags
            verbose - int - if 0 silent, if 1 progress bar

        Returns:
            - list - values of loss and metric computed on predictions for
                `texts` based on their ground-truth `tags`
        """
        embedded_texts = self.generate_bert_embeddings(texts, use_cache=False)
        return super(DenseHead, self).evaluate(embedded_texts, tags, verbose)

    def load(self, most_recent):
        """Load a model from its root folder by loading the configuration file
        and refreshing the instances attributes based on this file, setting up
        the model, compiling the model and finally loading the weights from
        the `weights` subdirectory

        Argument:
            model_path - string - the root path where the model files have been
                written (the weights in the `weights` subdirectory and the
                configuration file in the `config` subdirectory)

        Returns:
            self - EmbeddingModel
        """
        self.bert = hf.TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        return super(DenseHead, self).load(most_recent)


class BertEmbeddingModelWithKeywords(EmbeddingModelWithKeywords, BertEmbeddingModel):
    """This is the EmbeddingModel class for the pretrained fine_tuned BERT
    model using keywords
    """

    def __init__(
        self,
        max_seq_len=10,
        n_tags=2,
        optimizer_info={},
        metric_info={},
        data_name="",
        tag_index={},
        keyword_index={},
        n_keywords=0,
    ):
        """
        Arguments:
            max_seq_len - int - the fixed length of token lists to be passed to
                the model after padding/truncating
            n_tags - int - the number of distinct tags
            optimizer_info - dict - optimizer parameters
            metric_info - dict - metric parameters
            data_name - string - the data the model was trained on (Merlin,
                TMdb...)
            n_keywords - int - if > 0 number of keywords used to build a wide
                part to the model to take in one-hot-encoded features along the
                text data
        """
        EmbeddingModelWithKeywords.__init__(
            self,
            n_tags,
            optimizer_info,
            metric_info,
            data_name,
            tag_index,
            keyword_index,
            n_keywords,
            "bert_kws",
        )
        self.max_seq_len = max_seq_len

    def from_config(self, config):
        """Refreshes the object's initialization attributes based on a
        configuration dictionary

        Argument:
            config - dict - configuration dictionary
        """
        EmbeddingModelWithKeywords.from_config(self, config)
        self.max_seq_len = config["max_seq_len"]

    def get_config(self):
        """Generates a configuration dictionary based on the instance's
        attributes

        Returns:
             - dict - the configuration dictionary
        """
        return {
            **EmbeddingModelWithKeywords.get_config(self),
            "max_seq_len": self.max_seq_len,
        }

    def setup_wide_part(self):
        """Sets up the wide branch of the network

        Returns:
            wide_input - tf.keras.Input - the wide input
            wide_output - tf.Tensor - the output of the wide layers
        """
        wide_input = Input(self.n_keywords, dtype="float32", name="tags")
        wide_output = Dropout(0.5)(wide_input)
        return wide_input, wide_output

    def setup(self):
        """This setups the model architecture with
            - a branch that handles text data
            - a branch that handles wide one-hot-encoded input
            - the concatenation of the outputs of both branches
            - the final dense layer with a sigmoid activation for multi-label
            classification
        and assigns the resulting tf.keras.Model to `self.model`
        """
        EmbeddingModelWithKeywords.setup(self)

    def predict(self, texts, keywords, use_cache=False):
        """Predicts the `texts`' tags using the encapsulated model

        Argument:
            texts - tf.tensor - pre-processed text tokens
            keywords - tf.tensor - one-hot-encoded keywords
            use_cache - boolean - whether to try and fetch test predictions from
                the cache and save there if no file is found

        Returns:
             - tf.tensor - model predictions
        """
        return EmbeddingModelWithKeywords.predict(self, texts, keywords, use_cache)

    def evaluate(self, texts, tags, keywords, verbose=1):
        """Evaluates the `texts`' predictions using the encapsulated model

        Argument:
            texts - tf.tensor - pre-processed text tokens
            tags - np.array - one-hot-encoded tags
            keywords - tf.tensor - one-hot-encoded keywords
            verbose - int - if 0 silent, if 1 progress bar

        Returns:
            - list - values of loss and metric computed on predictions for
                `texts` based on their ground-truth `tags`
        """
        return EmbeddingModelWithKeywords.evaluate(self, texts, tags, keywords, verbose)
