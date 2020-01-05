from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input

class EncoderNetwork:
    def __init__(self):


        self._init_branch_payload()
        self._init_branch_carrier()
        
    def _init_branch_payload(self):

        self.branch__payload_conv_1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch__payload_conv_2 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch__payload_conv_3 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch__payload_conv_4 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch__payload_conv_5 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch__payload_conv_6 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch__payload_conv_7 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        
        self.payload_tensors = [self.branch__payload_conv_1, self.branch__payload_conv_2,\
                                self.branch__payload_conv_3, self.branch__payload_conv_4, self.branch__payload_conv_5,\
                                self.branch__payload_conv_6, self.branch__payload_conv_7]
    
    def _init_branch_carrier(self):
        
        self.branch_carrier_conv_1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch_carrier_concat_1 = Concatenate([self.branch_carrier_conv_1, self.branch__payload_conv_1])
        
        self.branch_carrier_conv_2 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        
        self.branch_carrier_conv_3 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch_carrier_concat_2 = Concatenate([self.branch_carrier_conv_3, self.branch__payload_conv_3])
        
        self.branch_carrier_conv_4 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        
        self.branch_carrier_conv_5 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch_carrier_concat_3 = Concatenate([self.branch_carrier_conv_5, self.branch__payload_conv_5])
        
        self.branch_carrier_conv_6 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        
        self.branch_carrier_conv_7 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch_carrier_concat_4 = Concatenate([self.branch_carrier_conv_7, self.branch__payload_conv_7])
        
        self.branch_carrier_conv_8 = Conv2D(16, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.branch_carrier_conv_9 = Conv2D(8, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.encoded_output = Conv2D(3, 1, padding='same', kernel_initializer='he_normal')
        
        self.carrier_tensors = [self.branch_carrier_conv_1, self.branch_carrier_concat_1,\
                               self.branch_carrier_conv_2, self.branch_carrier_conv_3, self.branch_carrier_concat_2,\
                               self.branch_carrier_conv_4, self.branch_carrier_conv_5, self.branch_carrier_concat_3,\
                               self.branch_carrier_conv_6, self.branch_carrier_conv_7, self.branch_carrier_concat_4,\
                               self.branch_carrier_conv_8, self.branch_carrier_conv_9, self.encoded_output]
    
    def get_network(self, payload, carrier):
        self.payload_tensors.insert(0, payload)
        self.carrier_tensors.insert(0, carrier)
        
        self.payload_layers = []
        for i in range(len(self.payload_tensors) - 1):
            if i == 0:
                self.payload_layers.append(self.payload_tensors[i + 1](self.payload_tensors[i]))
            else:
                self.payload_layers.append(layer(self.payload_tensors[i + 1]))
                
        self.carrier_layers = []
        for i in range(len(self.carrier_tensors) - 1):
            if i == 0:
                self.carrier_layers.append(self.carrier_tensors[i + 1](self.carrier_tensors[i]))
            else:
                self.carrier_layers.append(self.carrier_layers[-1](self.carrier_tensors[i + 1]))
        
        # return encoded output
        return self.carrier_layers[-1]