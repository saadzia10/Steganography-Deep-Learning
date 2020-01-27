from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input

class DecoderNetwork:
    def __init__(self, target_image_shape = (32, 32, 1)):
#         super(DecoderModel, self).__init__()
        self.target_image_shape = target_image_shape
        
    def _init_network(self, input_):

        self.conv_1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(input_)
        self.conv_2 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.conv_1)
        self.concat_1 = Concatenate()([self.conv_1, self.conv_2])
        self.conv_3 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.concat_1)
        
        self.branch_1_conv_1 = Conv2D(8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.conv_3)
        self.branch_1_conv_2 = Conv2D(8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_1_conv_1)
        self.branch_1_concat_1 = Concatenate()([self.branch_1_conv_1, self.conv_3])
        self.branch_1_conv_3 = Conv2D(8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_1_concat_1)
        self.branch_1_conv_4 = Conv2D(3, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_1_conv_3)
        self.branch_1_concat_2 = Concatenate()([self.branch_1_conv_3, self.branch_1_concat_1])
        self.branch_1_conv_5 = Conv2D(3, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_1_concat_2)
        self.decoded_output = Conv2D(self.target_image_shape[-1], self.target_image_shape[-1], padding='same', activation='relu', kernel_initializer='he_normal', name='decoded_output')(self.branch_1_conv_5)
        
        
        self.branch_2_conv_1 = Conv2D(8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.conv_3)
        self.branch_2_conv_2 = Conv2D(8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_2_conv_1)
        self.branch_2_conv_3 = Conv2D(8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_2_conv_2)
        self.branch_2_conv_4 = Conv2D(3, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_2_conv_3)
        self.branch_2_conv_5 = Conv2D(3, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_2_conv_4)
        self.decoded_host_output = Conv2D(self.target_image_shape[-1], self.target_image_shape[-1], padding='same', activation='relu', kernel_initializer='he_normal', name='decoded_host_output')(self.branch_2_conv_5)
        
        
    def get_network(self, encoder_output):
        
        self._init_network(encoder_output)

        return self.decoded_output, self.decoded_host_output