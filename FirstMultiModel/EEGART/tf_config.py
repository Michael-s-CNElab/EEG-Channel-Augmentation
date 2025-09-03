from transformers import PretrainedConfig
from typing import List


"""
To-do:
Two class to define, ART model and EEG encoder.
"""
class ARTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ART`]. It is used to instantiate a ART Model.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ART model
    [!!ART paper address] architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        src_channel_size ('int', *optional*, defaults to 30):
            Dimensionality of the input channel of EEG signal.
            It will be projects to d_model later.
        tgt_channel_size ('int', *optional*, defaults to 30):
            Dimensionality of the output channel of EEG signal.
            It will be projects to d_model later.
        N (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        d_model ('int', *optional*, defaults to 128)
            Dimensionality of the encoder layers and the pooler layer.
        d_ff ('int', *optional*, defaults to 2048)
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        h ('int', *optional*, defaults to 8)
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout ('float', *optional*, defaults to 0.1)
            Dropout ratio of the model.

    Example:

    ```python
        
    ```"""


    model_type = "ART"

    def __init__(
            self,
            src_len=30,
            tgt_len=30,
            N=6,
            d_model=128,
            d_ff=2048,
            h=8,
            dropout=0.1,
            **kwargs,
        ):
            super().__init__(**kwargs)

            self.src_len = src_len
            self.tgt_len = tgt_len
            self.N = N
            self.d_model = d_model
            self.d_ff = d_ff
            self.h = h
            self.dropout = dropout

class SLTConfig(ARTConfig):

    model_type = "SLT"

    def __init__(
            self,
            tgt_len=204,
            source_voxel_time = 200,
            sensor_time = 200,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.tgt_len= tgt_len
            self.source_voxel_time = source_voxel_time
            self.sensor_time = sensor_time


class ARTEncoder_CLSConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ART`]. It is used to instantiate a ART Model.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ART model
    [!!ART paper address] architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        src_channel_size ('int', *optional*, defaults to 30):
            Dimensionality of the input channel of EEG signal.
            It will be projects to d_model later.
        tgt_channel_size ('int', *optional*, defaults to 30):
            Dimensionality of the output channel of EEG signal.
            It will be projects to d_model later.
        N (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        d_model ('int', *optional*, defaults to 128)
            Dimensionality of the encoder layers and the pooler layer.
        d_ff ('int', *optional*, defaults to 2048)
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        h ('int', *optional*, defaults to 8)
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout ('float', *optional*, defaults to 0.1)
            Dropout ratio of the model.

    Example:

    ```python
        
    ```"""

    model_type = "ARTEncoder_CLSConfig"

    def __init__(
            self,
            src_channel_size=30,
            tgt_channel_size=30,
            N=6,
            d_model=128,
            d_ff=2048,
            h=8,
            dropout=0.1,
            time_len=1024,
            **kwargs,
        ):
            super().__init__(**kwargs)

            self.src_channel_size = src_channel_size
            self.tgt_channel_size = tgt_channel_size
            self.N = N
            self.d_model = d_model
            self.d_ff = d_ff
            self.h = h
            self.dropout = dropout
            self.time_len = time_len
