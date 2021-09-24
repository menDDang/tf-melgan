import argparse

import tensorflow as tf


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, alpha=0.3):
        super(DiscriminatorBlock, self).__init__()
        self.alpha = alpha

        self.conv1 = tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=15,
            strides=1,
            padding='same'
        )
        self.downsampling1 = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=41,
            strides=4,
            groups=4,
            padding='same'
        )
        self.downsampling2 = tf.keras.layers.Conv1D(
            filters=256,
            kernel_size=41,
            strides=4,
            groups=16,
            padding='same'
        )
        self.downsampling3 = tf.keras.layers.Conv1D(
            filters=1024,
            kernel_size=41,
            strides=4,
            groups=64,
            padding='same'
        )
        self.downsampling4 = tf.keras.layers.Conv1D(
            filters=1024,
            kernel_size=41,
            strides=4,
            groups=256,
            padding='same'
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=1024,
            kernel_size=5,
            strides=1,
            padding='same'
        )
        self.conv3 = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same'
        )

    @tf.function(experimental_compile=True)
    def call(self, x):
        feature_map1 = self.conv1(x)
        feature_map1 = tf.nn.leaky_relu(feature_map1, self.alpha)

        feature_map2_1 = self.downsampling1(feature_map1)
        feature_map2_1 = tf.nn.leaky_relu(feature_map2_1, self.alpha)
        
        feature_map2_2 = self.downsampling2(feature_map2_1)
        feature_map2_2 = tf.nn.leaky_relu(feature_map2_2, self.alpha)

        feature_map2_3 = self.downsampling3(feature_map2_2)
        feature_map2_3 = tf.nn.leaky_relu(feature_map2_3, self.alpha)

        feature_map2_4 = self.downsampling4(feature_map2_3)
        feature_map2_4 = tf.nn.leaky_relu(feature_map2_4, self.alpha)

        feature_map3 = self.conv2(feature_map2_4)
        feature_map3 = tf.nn.leaky_relu(feature_map3, self.alpha)
        output = self.conv3(feature_map3)
        #output = tf.nn.leaky_relu(output, self.alpha)

        feature_maps = [
            feature_map1,
            feature_map2_1, feature_map2_2, feature_map2_3, feature_map2_3,
            feature_map3
        ]

        return output, feature_maps


class MelGanDiscriminator(tf.keras.Model):
    def __init__(self, hp : dict):
        super(MelGanDiscriminator, self).__init__()

        self.num_blocks = hp["discriminator_num_blocks"]
        alpha = hp["discriminator_leaky_relu_alpha"]
        
        self.blocks = []
        for n in range(self.num_blocks):
            self.blocks.append(DiscriminatorBlock(alpha=alpha))

        self.avg_pools = []
        for n in range(self.num_blocks - 1):
            self.avg_pools.append(
                tf.keras.layers.AveragePooling1D(pool_size=4, padding='same')
            )

    @tf.function(experimental_compile=True)
    def call(self, x):
        outputs = []
        feature_maps = []
        for n in range(self.num_blocks):
            out, map = self.blocks[n](x)
            outputs.append(out)
            for k in range(len(map)):
                feature_maps.append(map[k])
            if n < self.num_blocks - 1:
                x = self.avg_pools[n](x)
            
        return outputs, feature_maps

    @staticmethod
    def ParseArgument(parser: argparse.ArgumentParser):
        parser.add_argument("--discriminator_num_blocks", type=int, default=1)
        parser.add_argument("--discriminator_leaky_relu_alpha", type=float, default=0.2)

    @staticmethod
    def CreateHparamDict(hp : dict, args):
        hp["discriminator_num_blocks"] = args.discriminator_num_blocks
        hp["discriminator_leaky_relu_alpha"] = args.discriminator_leaky_relu_alpha
        return hp