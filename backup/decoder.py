from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input

class DecoderNetwork:
    def __init__(self):

        self._init_network()
        
    def _init_network(self):

        self.conv_1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_2 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_3 = Conv2D(8, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_4 = Conv2D(8, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_5 = Conv2D(3, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_6 = Conv2D(3, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.decoded_output = Conv2D(1, self.target_image_shape[-1], padding='same', activation='relu', kernel_initializer='he_normal')
        
        self.decoder_tensors = [self.conv_1, self.conv_2,\
                                self.conv_3, self.conv_4, self.conv_5,\
                                self.conv_6, self.decoded_output]
    
    def get_network(self, encoder_output):
        self.decoder_tensors.insert(0, encoder_output)
        self.decoder_layers = []
        for i in range(len(self.decoder_tensors) - 1):
            self.decoder_layers.append(self.decoder_tensors[i + 1](self.decoder_tensors[i]))
        
        # return decoded output
        return self.decoder_layers[-1]