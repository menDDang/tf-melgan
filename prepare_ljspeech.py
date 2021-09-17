import os
import argparse

import tensorflow as tf

import datasets

def ParseArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_tfrecord", type=str, required=True)
    parser.add_argument("--valid_tfrecord", type=str, required=True)

    # Parse arguments for audio dataset
    datasets.AudioDataset.ParseArgument(parser)
    # Parse arguments for ljspeech
    datasets.LJSpeechDataset.ParseArgument(parser)

    args = parser.parse_args()
    return args

def CreateParamDict(args):
    hp = dict()
    hp = datasets.AudioDataset.CreateHparamDict(hp, args)
    return hp

if __name__ == "__main__":
    
    args = ParseArgument()
    hp = CreateParamDict(args)

    ljspeech = datasets.LJSpeechDataset(hp)
    ljspeech.Create(args.data_dir)
    ljspeech.Write(args.train_tfrecord, args.valid_tfrecord)
    