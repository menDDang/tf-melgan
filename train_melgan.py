import argparse

import tensorflow as tf

import datasets
from models import MelGanDiscriminator, MelGanGenerator

def ParseArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--chkpt_dir", type=str, default="./chkpt")

    parser.add_argument("--train_tfrecord", type=str, required=True)
    parser.add_argument("--valid_tfrecord", type=str, required=True)

    parser.add_argument("--init_learning_rate", type=float, default=0.01)
    parser.add_argument("--end_learning_rate", type=float, default=0.0001)
    parser.add_argument("--learning_rate_decay_steps", type=int, default=1000)
    parser.add_argument("--learning_rate_decay_rate", type=float, default=0.96)
    
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_iter", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default='sgd', help='one of {"sgd", "adam"}')
    
    # Parse arguments for audio dataset
    datasets.AudioDataset.ParseArgument(parser)
    # Parse arguments for ljspeech
    datasets.LJSpeechDataset.ParseArgument(parser)
    # Parse arguments for discriminator
    MelGanDiscriminator.ParseArgument(parser)
    
    args = parser.parse_args()
    return args

def CreateParamDict(args):
    hp = dict()
    hp["init_learning_rate"] = args.init_learning_rate
    hp["end_learning_rate"] = args.end_learning_rate
    hp["learning_rate_decay_steps"] = args.learning_rate_decay_steps
    hp["learning_rate_decay_rate"] = args.learning_rate_decay_rate

    hp["batch_size"] = args.batch_size
    hp["num_iter"] = args.num_iter
    hp["optimizer"] = args.optimizer

    hp = datasets.AudioDataset.CreateHparamDict(hp, args)
    hp = MelGanDiscriminator.CreateHparamDict(hp, args)
    return hp

def BuildOptimizer(hp):
    lr = tf.optimizers.schedules.PolynomialDecay(
        hp["init_learning_rate"],
        hp["learning_rate_decay_steps"],
        hp["end_learning_rate"])

    if hp["optimizer"] == "sgd":
        optimizer = tf.optimizers.SGD(lr)
        return optimizer
    elif hp["optimizer"] == "adam":
        optimizer = tf.optimizers.Adam(lr)
        return optimizer
    elif hp["optimizer"] == "adadelta":
        optimizer = tf.optimizers.Adadelta(lr, rho=0.95, epsilon=1e-10)
        return optimizer
    else:
        raise ValueError("Invalid type of optimizer")

#@tf.function
#def ComputeLoss(d_real, d_real_maps, d_fake, d_fake_maps):


@tf.function
def train_step(audio_real, audio_length,
               mel, mel_length,
               generator, discrinator, optimizer):
    
    audio_fake = generator(mel)
    #with tf.GradientTape() as tape:
    #    d_
if __name__ == "__main__":

    args = ParseArgument()
    hp = CreateParamDict(args)
    
    ljspeech = datasets.LJSpeechDataset(hp)
    ljspeech.Load(args.train_tfrecord, args.valid_tfrecord)
    train_dataset = ljspeech.train_dataset
    valid_dataset = ljspeech.valid_dataset

    generator = MelGanGenerator()
    discriminator = MelGanDiscriminator(hp)
    optimizer = BuildOptimizer(hp)

    for step, (audio_real, audio_length, mel, mel_length, _, _) in enumerate(train_dataset):
        mel = tf.expand_dims(mel, 0)
        print(mel.shape)
        
        audio_fake = generator(mel)
        print(audio_fake.shape)

        d_fake_probs, d_fake_feature_maps = discriminator(audio_fake)
        for prob in d_fake_probs:
            print(prob.shape)
            #(1, 962, 1)
            #(1, 241, 1)
            #(1, 61, 1)
        break