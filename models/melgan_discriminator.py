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
        output = tf.nn.leaky_relu(output, self.alpha)

        feature_maps = [
            feature_map1,
            feature_map2_1, feature_map2_2, feature_map2_3, feature_map2_3,
            feature_map3
        ]

        return output, feature_maps


class MelGanDiscriminator(tf.keras.Model):
    def __init__(self, alpha=0.3):
        super(MelGanDiscriminator, self).__init__()

        self.block1 = DiscriminatorBlock(alpha=alpha)
        self.avg_pool1 = tf.keras.layers.AveragePooling1D(pool_size=4, padding='same')
        self.block2 = DiscriminatorBlock(alpha=alpha)
        self.avg_pool2 = tf.keras.layers.AveragePooling1D(pool_size=4, padding='same')
        self.block3 = DiscriminatorBlock(alpha=alpha)


    @tf.function(experimental_compile=True)
    def call(self, x):

        output1, feature_maps1 = self.block1(x)
        x = self.avg_pool1(x)
        output2, feature_maps2 = self.block2(x)
        x = self.avg_pool2(x)
        output3, feature_maps3 = self.block3(x)

        outputs = [output1, output2, output3]
        feature_maps = [] + feature_maps1 + feature_maps2 + feature_maps3
        return outputs, feature_maps

if __name__ == "__main__":
    batch_size = 1
    input_time_length = 2048 * 4
    input_dimension = 1

    mel_gan_discriminator = MelGanDiscriminator()

    x = tf.zeros(shape=[batch_size, input_time_length, input_dimension], dtype=tf.float32)
    outputs, feature_maps = mel_gan_discriminator(x)

    print("outputs : ")
    for output in outputs:
        print(output.shape)
    
    print("feature maps : ")
    for features in feature_maps:
        print(features.shape)