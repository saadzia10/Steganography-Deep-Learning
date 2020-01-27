from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input

class EncoderNetwork:
    def __init__(self, carrier_shape=(32, 32, 3), payload_shape=(32, 32, 1)):
        
#         super(EncoderModel, self).__init__()
        self.carrier_shape = carrier_shape
        self.payload_shape = payload_shape
        
    def _init_branch_payload(self, payload):

        self.branch__payload_conv_1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal', input_shape=self.payload_shape)(payload)
        self.branch__payload_conv_2 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch__payload_conv_1)
        self.branch__payload_conv_3 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch__payload_conv_2)
        self.branch__payload_conv_4 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch__payload_conv_3)
        self.branch__payload_conv_5 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch__payload_conv_4)
        self.branch__payload_conv_6 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch__payload_conv_5)
        self.branch__payload_conv_7 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch__payload_conv_6)
        
        self.payload_tensors = [self.branch__payload_conv_1, self.branch__payload_conv_2,\
                                self.branch__payload_conv_3, self.branch__payload_conv_4, self.branch__payload_conv_5,\
                                self.branch__payload_conv_6, self.branch__payload_conv_7]
    
    def _init_branch_carrier(self, carrier):
        
        self.branch_carrier_conv_1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(carrier)
        self.branch_carrier_concat_1 = Concatenate()([self.branch_carrier_conv_1, self.branch__payload_conv_1])
        
        self.branch_carrier_conv_2 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_carrier_concat_1)
        
        self.branch_carrier_conv_3 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_carrier_conv_2)
        self.branch_carrier_concat_2 = Concatenate()([self.branch_carrier_conv_3, self.branch__payload_conv_3])
        
        self.branch_carrier_conv_4 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_carrier_concat_2)
        
        self.branch_carrier_conv_5 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_carrier_conv_4)
        self.branch_carrier_concat_3 = Concatenate()([self.branch_carrier_conv_5, self.branch__payload_conv_5])
        
        self.branch_carrier_conv_6 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_carrier_concat_3)
        
        self.branch_carrier_conv_7 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_carrier_conv_6)
        self.branch_carrier_concat_4 = Concatenate()([self.branch_carrier_conv_7, self.branch__payload_conv_7])
        
        self.branch_carrier_conv_8 = Conv2D(16, 1, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_carrier_concat_4)
        self.branch_carrier_conv_9 = Conv2D(8, 1, padding='same', activation='relu', kernel_initializer='he_normal')(self.branch_carrier_conv_8)
        self.encoded_output = Conv2D(3, 1, padding='same', kernel_initializer='he_normal', name='encoded_output')(self.branch_carrier_conv_9)
        
        self.carrier_tensors = [self.branch_carrier_conv_1, self.branch_carrier_concat_1,\
                               self.branch_carrier_conv_2, self.branch_carrier_conv_3, self.branch_carrier_concat_2,\
                               self.branch_carrier_conv_4, self.branch_carrier_conv_5, self.branch_carrier_concat_3,\
                               self.branch_carrier_conv_6, self.branch_carrier_conv_7, self.branch_carrier_concat_4,\
                               self.branch_carrier_conv_8, self.branch_carrier_conv_9, self.encoded_output]
    
    def get_network(self, carrier, payload):
        
        self._init_branch_payload(payload)
        self._init_branch_carrier(carrier)
        
        return self.encoded_output