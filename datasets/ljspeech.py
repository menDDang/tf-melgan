import os
import argparse

import tensorflow as tf

from .audio_dataset import AudioDataset

def NormalizeText(text : str):
    text = text.lower()
    normalized = ""
    for c in text:
        if c == '"':
            continue
        if c == "'":
            continue
        if c == "-":
            c = " "
        normalized += c
    return normalized

class LJSpeechDataset(AudioDataset):
    def __init__(self, hp):
        super(LJSpeechDataset, self).__init__(hp)

    def Create(self, data_dir, train_ratio=0.8):
        file_list = self._GetFileList(data_dir)
        num_train = int(len(file_list) * train_ratio)

        self._CreateTokenizer()
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(file_list[0:num_train])
        )
        self.valid_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(file_list[num_train:])
        )

        self.train_dataset = self.train_dataset.map(
            lambda line: self._ParseLine(line[0], line[1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.valid_dataset = self.valid_dataset.map(
            lambda line: self._ParseLine(line[0], line[1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        

    def Write(self, train_tfrecord_path, valid_tfrecord_path):
        # Serialize dataset to save as tfrecord format
        self.train_dataset = self.train_dataset.map(
            self._SerializeExample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        writer = tf.data.experimental.TFRecordWriter(train_tfrecord_path)
        writer.write(self.train_dataset)

        self.valid_dataset = self.valid_dataset.map(
            self._SerializeExample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        writer = tf.data.experimental.TFRecordWriter(valid_tfrecord_path)
        writer.write(self.valid_dataset)

    def Load(self, train_tfrecord_path, valid_tfrecord_path):
        self.train_dataset = tf.data.TFRecordDataset(train_tfrecord_path)
        self.train_dataset = self.train_dataset.map(
            self._ParseExample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.valid_dataset = tf.data.TFRecordDataset(valid_tfrecord_path)
        self.valid_dataset = self.valid_dataset.map(
            self._ParseExample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    def _CreateTokenizer(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        space = ' '
        start = 'S'
        end = 'E'
        
        keys = [space, start, end] + [c for c in alphabet]
        values = range(len(keys))

        init_key_values = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)
        hash_table = tf.lookup.StaticHashTable(init_key_values, default_value=0)
        self.tokenizer = lambda text: hash_table.lookup(tf.strings.bytes_split(text))
        self.vocab_size = len(keys)

    def _GetFileList(self, data_dir):
        transcript_path = os.path.join(data_dir, "metadata.csv")
        output = []
        with open(transcript_path, "r") as f:
            for line in f.readlines():
                line = line.split("\n")[0]
                uttid, _, text = line.split("|")
                file_path = os.path.join(data_dir, "wavs", uttid + ".wav")
                text = NormalizeText(text)
                output.append([file_path, text])
        return output

    def _ParseLine(self, audio_file_path, text):
        # load audio files
        audio, sr = tf.py_function(
            lambda path: self.LoadAudio(path.numpy()),
            inp=[audio_file_path],
            Tout=[tf.float32, tf.int32]
        )

        # extract features
        magnitudes = self.GetMagnitudes(audio, sr)
        mel = self.GetMel(magnitudes)

        text = 'S' + text + 'E'
        tokens = self.tokenizer(tf.reshape(text, ()))

        return audio, tf.shape(audio)[0], mel, tf.shape(mel)[0], tokens, tf.shape(tokens)[0]

    def _SerializeExample(self, audio, audio_length, mel, mel_length, tokens, token_length):

        def _bytes_features(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _serialize(_audio, _audio_length, _mel, _mel_length, _tokens, _token_length):
            serialized_audio = tf.io.serialize_tensor(_audio)
            serialized_audio_length = tf.io.serialize_tensor(_audio_length)
            serialized_mel = tf.io.serialize_tensor(_mel)
            serialized_mel_length = tf.io.serialize_tensor(_mel_length)
            serialized_tokens = tf.io.serialize_tensor(_tokens)
            serialized_token_length = tf.io.serialize_tensor(_token_length)

            feature = {
                'audio': _bytes_features(serialized_audio),
                'audio_length': _bytes_features(serialized_audio_length),
                'mel': _bytes_features(serialized_mel),
                'mel_length': _bytes_features(serialized_mel_length),
                'tokens': _bytes_features(serialized_tokens),
                'token_length': _bytes_features(serialized_token_length)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            example = example.SerializeToString()
            return example

        output = tf.py_function(
            _serialize,
            inp=[audio, audio_length, mel, mel_length, tokens, token_length],
            Tout=[tf.string])

        return tf.reshape(output, ())

    def _ParseExample(self, serialized_example):

        parse_dict = {
            'audio': tf.io.FixedLenFeature([], tf.string),
            'audio_length': tf.io.FixedLenFeature([], tf.string),
            'mel': tf.io.FixedLenFeature([], tf.string),
            'mel_length': tf.io.FixedLenFeature([], tf.string),
            'tokens': tf.io.FixedLenFeature([], tf.string),
            'token_length': tf.io.FixedLenFeature([], tf.string)
        }

        example = tf.io.parse_single_example(serialized_example, parse_dict)

        audio = tf.io.parse_tensor(example['audio'], out_type=tf.float32)
        audio_length = tf.io.parse_tensor(example['audio_length'], out_type=tf.int32)
        mel = tf.io.parse_tensor(example['mel'], out_type=tf.float32)
        mel_length = tf.io.parse_tensor(example['mel_length'], out_type=tf.int32)
        tokens = tf.io.parse_tensor(example['tokens'], out_type=tf.int32)
        token_length = tf.io.parse_tensor(example['token_length'], out_type=tf.int32)
        return (audio, audio_length, mel, mel_length, tokens, token_length)

    @staticmethod
    def ParseArgument(parser: argparse.ArgumentParser):
        pass
