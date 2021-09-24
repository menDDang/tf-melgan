import os
import argparse

import soundfile as sf
import tensorflow as tf


class AudioDataset:
    def __init__(self, hp):
        self.frame_length_in_sec = hp["frame_length_in_sec"]
        self.step_length_in_sec = hp["step_length_in_sec"]
        self.num_fft_point = hp["num_fft_point"]
        
        self.mel_bins = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=hp["num_mels"],
            num_spectrogram_bins=int(self.num_fft_point / 2) + 1,
            sample_rate=hp["sampling_rate"],
            lower_edge_hertz=hp["hertz_low"],
            upper_edge_hertz=hp["hertz_high"]
        )

    def CheckFileExist(self, filepath):
        return tf.py_function(
            lambda x: os.path.exists(x.numpy()),
            inp=[filepath],
            Tout=tf.bool)

    def LoadAudio(self, file_path):
        audio, sr = sf.read(file_path)
        return audio, sr
    
    def GetDuration(self, file_path):
        audio, sr = sf.read(file_path)
        duration = audio.shape[0] / float(sr)
        return duration
        
    def GetMagnitudes(self, audio, sr):
        frame_length = tf.cast(
            tf.round(float(sr) * self.frame_length_in_sec),
            tf.int32
        )
        step_length = tf.cast(
            tf.round(float(sr) * self.step_length_in_sec),
            tf.int32
        )
        stft = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=step_length,
            fft_length=self.num_fft_point
        )
        magnitudes = tf.abs(stft)
        return magnitudes

    def GetMel(self, magnitudes, normalize=False):
        mel = tf.linalg.matmul(magnitudes, self.mel_bins)
        log_mel = tf.math.log(mel + 1e-6)
        if normalize:
            log_mel -= tf.reduce_mean(log_mel, axis=0)
        return log_mel

    
    @staticmethod
    def ParseArgument(parser : argparse.ArgumentParser):
        parser.add_argument("--sampling_rate", type=int, default=16000)
        parser.add_argument("--frame_length_in_sec", type=float, default=0.02)
        parser.add_argument("--step_length_in_sec", type=float, default=0.01)
        parser.add_argument("--num_fft_point", type=int, default=512)
        parser.add_argument("--num_mels", type=int, default=80)
        parser.add_argument("--hertz_low", type=int, default=0)
        parser.add_argument("--hertz_high", type=int, default=8000)

    @staticmethod
    def CreateHparamDict(hp : dict, args):

        hp["sampling_rate"] = args.sampling_rate
        hp["frame_length_in_sec"] = args.frame_length_in_sec
        hp["step_length_in_sec"] = args.step_length_in_sec
        hp["num_fft_point"] = args.num_fft_point
        hp["num_mels"] = args.num_mels
        hp["hertz_low"] = args.hertz_low
        hp["hertz_high"] = args.hertz_high       
        return hp