import argparse
import os
import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api

import datasets
from models import MelGanDiscriminator, MelGanGenerator

SAMPLING_RATE = 22050

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
    
    parser.add_argument("--max_duration", type=float, default=10)
    parser.add_argument("--feat_matching_weight", type=float, default=10)
    parser.add_argument("--use_hinge_loss", type=bool, default=True)

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

    hp["max_duration"] = args.max_duration
    hp["feat_matching_weight"] = args.feat_matching_weight
    hp["use_hinge_loss"] = args.use_hinge_loss

    hp = datasets.AudioDataset.CreateHparamDict(hp, args)
    hp = MelGanDiscriminator.CreateHparamDict(hp, args)
    return hp

def BuildOptimizer(hp):
    d_lr = tf.optimizers.schedules.PolynomialDecay(
        hp["init_learning_rate"],
        hp["learning_rate_decay_steps"],
        hp["end_learning_rate"])
    g_lr = tf.optimizers.schedules.PolynomialDecay(
        hp["init_learning_rate"],
        hp["learning_rate_decay_steps"],
        hp["end_learning_rate"])

    if hp["optimizer"] == "sgd":
        d_optimizer = tf.optimizers.SGD(d_lr)
        g_optimizer = tf.optimizers.SGD(g_lr)
        return d_optimizer, g_optimizer
    elif hp["optimizer"] == "adam":
        d_optimizer = tf.optimizers.Adam(d_lr, beta_1=0.5, beta_2=0.9)
        g_optimizer = tf.optimizers.Adam(g_lr, beta_1=0.5, beta_2=0.9)
        return d_optimizer, g_optimizer
    elif hp["optimizer"] == "adadelta":
        d_optimizer = tf.optimizers.Adadelta(d_lr, rho=0.95, epsilon=1e-10)
        g_optimizer = tf.optimizers.Adadelta(g_lr, rho=0.95, epsilon=1e-10)
        return d_optimizer, g_optimizer
    else:
        raise ValueError("Invalid type of optimizer")

@tf.function
def compute_d_loss(d_reals, d_fakes, use_hinge_loss=True):
    if use_hinge_loss:
        '''
        * Use "hinge loss" for discriminator
        * Paper - Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. 
        "Spectral normalization for generative adversarial networks.", 
        arXiv preprint arXiv:1802.05957, 2018.
        * Link - https://arxiv.org/pdf/1802.05957.pdf
        
        # d_loss_real = mean(min(0, 1 - d_real))
        # d_loss_fake = mean(min(0, 1 + d_fake))
        # d_loss = d_loss_real + d_loss_fake
        '''
        d_loss = 0
        for k in range(len(d_reals)):
            d_loss_real_k = -1.0 * tf.reduce_mean(
                tf.minimum(0.0, -1 + d_reals[k]))
            d_loss_fake_k = -1.0 * tf.reduce_mean(
                tf.minimum(0.0, -1 - d_fakes[k]))
            d_loss += d_loss_real_k + d_loss_fake_k
    else:
        '''
        * Use original loss for discriminator
        '''
        d_loss = 0
        for k in range(len(d_reals)):
            d_loss_real_k = -1.0 * tf.reduce_mean(
                tf.math.log(tf.nn.sigmoid(d_reals[k])))
            d_loss_fake_k = -1.0 * tf.reduce_mean(
                tf.math.log(1 - tf.nn.sigmoid(d_fakes[k])))
            d_loss += d_loss_real_k + d_loss_fake_k
    return d_loss

@tf.function
def compute_g_loss(d_real_maps, d_fakes, d_fake_maps, feat_matching_weight=10):
    # g_adv_loss = mean(sum(-d_fake))
    # g_feat_matching_loss = mean(
    #   rms(d_real_maps[i] - d_fake_maps[i]))
    # g_loss = g_adv_loss + feat_matching_weight * g_feat_matching_loss
    g_adv_loss = 0
    for k in range(len(d_fakes)):
        g_adv_loss_k = tf.reduce_mean(-1 * d_fakes[k])
        g_adv_loss += g_adv_loss_k
    g_feat_matching_loss = 0
    for k in range(len(d_real_maps)):
        g_feat_matching_loss += \
            tf.reduce_mean(tf.abs(d_real_maps[k] - d_fake_maps[k]))
    g_loss = tf.reduce_mean(g_adv_loss) \
        + feat_matching_weight * g_feat_matching_loss
    return g_loss
    
@tf.function
def train_step(audio_real, mel,
               generator, discriminator, 
               g_optimizer, d_optimizer,
               max_audio_length, 
               feat_matching_weight=10,
               use_hinge_loss=True):
    audio_fake = generator(mel)
    audio_fake = audio_fake[:, :max_audio_length, :]    
    with tf.GradientTape() as tape:
        d_reals, d_real_maps = discriminator(audio_real)
        d_fakes, d_fake_maps = discriminator(audio_fake)

        d_loss = compute_d_loss(d_reals, d_fakes, use_hinge_loss)
        
        d_vars = discriminator.trainable_variables
        d_grads = tape.gradient(d_loss, d_vars)
        d_optimizer.apply_gradients(zip(d_grads, d_vars))

    with tf.GradientTape() as tape:
        audio_fake = generator(mel)
        audio_fake = audio_fake[:, :max_audio_length, :]
        d_reals, d_real_maps = discriminator(audio_real)
        d_fakes, d_fake_maps = discriminator(audio_fake)

        g_loss = compute_g_loss(d_real_maps, 
            d_fakes, d_fake_maps, 
            feat_matching_weight)

        g_vars = generator.trainable_variables
        g_grads = tape.gradient(g_loss, g_vars)
        g_optimizer.apply_gradients(zip(g_grads, g_vars))
        
    return d_loss, g_loss

@tf.function
def valid_step(audio_real, mel,
               generator, discriminator, 
               max_audio_length, 
               feat_matching_weight=10,
               use_hinge_loss=True):
    audio_fake = generator(mel)
    audio_fake = audio_fake[:, :max_audio_length, :]    
    
    d_reals, d_real_maps = discriminator(audio_real)
    d_fakes, d_fake_maps = discriminator(audio_fake)

    d_loss = compute_d_loss(d_reals, d_fakes, use_hinge_loss)
    g_loss = compute_g_loss(d_real_maps, 
        d_fakes, d_fake_maps, 
        feat_matching_weight)
        
    return d_loss, g_loss


if __name__ == "__main__":

    args = ParseArgument()
    hp = CreateParamDict(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    log_dir = os.path.join(args.log_dir, current_time)
    chkpt_dir = os.path.join(args.chkpt_dir, current_time)

    #with tf.summary.create_file_writer(log_dir).as_default():
    #    api.hparams(hp)
    
    SAMPLING_RATE = hp["audio"]["sampling_rate"]
    window_size = int(hp["audio"]["frame_length_in_sec"] * SAMPLING_RATE)
    hop_size = int(hp["audio"]["step_length_in_sec"] * SAMPLING_RATE)
    
    ljspeech = datasets.LJSpeechDataset(hp)
    ljspeech.Load(args.train_tfrecord, args.valid_tfrecord)
    train_dataset = ljspeech.train_dataset
    valid_dataset = ljspeech.valid_dataset

    # Erase un-used variables
    train_dataset = train_dataset.map(
        lambda audio, audio_length, mel, mel_length, tokens, token_length :
        (tf.expand_dims(audio, axis=1), mel))
    valid_dataset = valid_dataset.map(
        lambda audio, audio_length, mel, mel_length, tokens, token_length :
        (tf.expand_dims(audio, axis=1), mel))

    # Filter by audio length
    max_audio_length = int(hp["max_duration"] * SAMPLING_RATE)
    max_mel_length = int((max_audio_length - window_size) / hop_size)
    train_dataset = train_dataset.filter(
        lambda audio, mel : 
        (tf.shape(audio)[0] < max_audio_length))
    valid_dataset = valid_dataset.filter(
        lambda audio, mel : 
        (tf.shape(audio)[0] < max_audio_length))
    
    # Make mini batch
    train_dataset = train_dataset.padded_batch(
        batch_size=hp["batch_size"],
        padded_shapes=([max_audio_length, 1], [max_mel_length, 80]))
    valid_dataset = valid_dataset.padded_batch(
        batch_size=hp["batch_size"],
        padded_shapes=([max_audio_length, 1], [max_mel_length, 80]))
        
    generator = MelGanGenerator()
    discriminator = MelGanDiscriminator(hp)
    d_optimizer, g_optimizer = BuildOptimizer(hp)

    for iter in range(1, hp["num_iter"] + 1):
        #
        # Training
        #
        train_d_loss_object = tf.keras.metrics.Mean()
        train_g_loss_object = tf.keras.metrics.Mean()

        for step, (audio_real, mel) in enumerate(train_dataset):
            d_loss, g_loss = train_step(
                audio_real, mel,
                generator, discriminator, 
                d_optimizer, g_optimizer,
                max_audio_length, 
                feat_matching_weight=hp["feat_matching_weight"],
                use_hinge_loss=hp["use_hinge_loss"]
            )
            train_d_loss_object.update_state(d_loss)
            train_g_loss_object.update_state(g_loss)
            
        train_d_loss = train_d_loss_object.result()
        train_g_loss = train_g_loss_object.result()
        
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar("Train D Loss", train_d_loss, step=iter)
            tf.summary.scalar("Train G Loss", train_g_loss, step=iter)
        
        #
        # Validation
        #
        valid_d_loss_object = tf.keras.metrics.Mean()
        valid_g_loss_object = tf.keras.metrics.Mean()

        for step, (audio_real, mel) in enumerate(valid_dataset):
            d_loss, g_loss = valid_step(
                audio_real, mel,
                generator, discriminator,
                max_audio_length, 
                feat_matching_weight=hp["feat_matching_weight"],
                use_hinge_loss=hp["use_hinge_loss"]
            )
            valid_d_loss_object.update_state(d_loss)
            valid_g_loss_object.update_state(g_loss)
            
        valid_d_loss = valid_d_loss_object.result()
        valid_g_loss = valid_g_loss_object.result()
        
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar("Valid D Loss", valid_d_loss, step=iter)
            tf.summary.scalar("Valid G Loss", valid_g_loss, step=iter)

        print("Iter : %d, Train D Loss : %f, Train G Loss : %f, " \
              "Valid D Loss : %f, Valid G Loss : %f" % \
            (iter, float(train_d_loss), float(train_g_loss), \
            float(valid_d_loss), float(valid_g_loss)))

        # Save checkpoint
        chkpt_file_path = os.path.join(chkpt_dir, "discriminator", "chkpt_step:{}_loss:{:.4f}".format(iter, float(train_d_loss)))
        discriminator.save_weights(chkpt_file_path)
        chkpt_file_path = os.path.join(chkpt_dir, "generator", "chkpt_step:{}_loss:{:.4f}".format(iter, float(train_g_loss)))
        generator.save_weights(chkpt_file_path)    
        