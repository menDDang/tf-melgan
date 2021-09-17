import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, alpha=0.3):
        super(ResidualBlock, self).__init__()
        self.alpha = alpha

        self.conv1 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            dilation_rate=dilation_rate,
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            dilation_rate=dilation_rate,
        )

    def call(self, x):
        conv1_out = self.conv1(x)
        conv1_out = tf.nn.leaky_relu(conv1_out, alpha=self.alpha)
        conv2_out = self.conv2(conv1_out)
        conv2_out = tf.nn.leaky_relu(conv2_out, alpha=self.alpha)
        return conv2_out + x

class ResidualStack(tf.keras.layers.Layer):
    def __init__(self, filters, alpha=0.3):
        super(ResidualStack, self).__init__()
    
        self.block1 = ResidualBlock(
            filters=filters,
            kernel_size=3,
            dilation_rate=1,
            alpha=alpha
        )
        self.block2 = ResidualBlock(
            filters=filters,
            kernel_size=3,
            dilation_rate=1,
            alpha=alpha
        )

        self.block3 = ResidualBlock(
            filters=filters,
            kernel_size=3,
            dilation_rate=3,
            alpha=alpha
        )
        self.block4 = ResidualBlock(
            filters=filters,
            kernel_size=3,
            dilation_rate=1,
            alpha=alpha
        )

        self.block5 = ResidualBlock(
            filters=filters,
            kernel_size=3,
            dilation_rate=9,
            alpha=alpha
        )
        self.block6 = ResidualBlock(
            filters=filters,
            kernel_size=3,
            dilation_rate=1,
            alpha=alpha
        )
    
    def call(self, x):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        x = x + block2_out

        block3_out = self.block3(x)
        block4_out = self.block4(block3_out)
        x = x + block4_out

        block5_out = self.block5(x)
        block6_out = self.block6(block5_out)
        x = x + block6_out
        return x

class MelGanGenerator(tf.keras.Model):

    def __init__(self, alpha=0.3):
        super(MelGanGenerator, self).__init__()
        self.alpha = alpha

        self.conv1 = tf.keras.layers.Conv1D(
            filters=512,
            kernel_size=[7],
            strides=1,
            padding='valid'
        )

        self.upsampling1 = tf.keras.layers.Conv1DTranspose(
            filters=256, 
            kernel_size=[16],
            strides=8,
            padding='valid'
        )
        self.residual_stack1 = ResidualStack(256, alpha)

        self.upsampling2 = tf.keras.layers.Conv1DTranspose(
            filters=128,
            kernel_size=16,
            strides=8,
            padding='valid'
        )
        self.residual_stack2 = ResidualStack(128, alpha)

        self.upsampling3 = tf.keras.layers.Conv1DTranspose(
            filters=64,
            kernel_size=4,
            strides=2,
            padding='valid'
        )
        self.residual_stack3 = ResidualStack(64, alpha)

        self.upsampling4 = tf.keras.layers.Conv1DTranspose(
            filters=32,
            kernel_size=4,
            strides=2,
            padding='valid'
        )
        self.residual_stack4 = ResidualStack(32, alpha)

        self.conv2 = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=7,
            strides=1,
            padding='valid',
            activation='tanh'
        )

    def call(self, x):
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, alpha=self.alpha)

        x = self.upsampling1(x)
        x = tf.nn.leaky_relu(x, alpha=self.alpha)
        x = self.residual_stack1(x)

        x = self.upsampling2(x)
        x = tf.nn.leaky_relu(x, alpha=self.alpha)
        x = self.residual_stack2(x)

        x = self.upsampling3(x)
        x = tf.nn.leaky_relu(x, alpha=self.alpha)
        x = self.residual_stack3(x)

        x = self.upsampling4(x)
        x = tf.nn.leaky_relu(x, alpha=self.alpha)
        x = self.residual_stack4(x)

        x = self.conv2(x)
        return x

if __name__ == "__main__":
    batch_size = 1
    input_time_length = 98
    input_dimension = 80

    mel_gan_generator = MelGanGenerator()

    x = tf.zeros(shape=[batch_size, input_time_length, input_dimension], dtype=tf.float32)
    x = mel_gan_generator(x)
    print(x.shape)
    