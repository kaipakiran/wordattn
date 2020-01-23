import numpy as np 
from tensorflow.keras.layers import Layer,InputSpec
from tensorflow.keras import backend as K 
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops


class Attn(Layer):
    def __init__(self,uit_dimension, **kwargs):
        self.uit_dimension = uit_dimension
        self.out_dimension = 1
        super(Attn,self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config["uit_dimension"] = self.uit_dimension
        return config
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape = (self.uit_dimension*2,1),
            initializer = 'glorot_uniform', trainable = True)
        super(Attn,self).build(input_shape)
    def call(self, hit,mask=None):
        val = K.dot(hit,self.kernel)
        if mask is not None:
            print("Shape of mask:",K.int_shape(mask))
            if isinstance(mask,list):
                mask = mask[0]
                val *=K.cast(mask,K.floatx())
                return val
        return val
    def compute_mask(self, hit, mask=None):
        # Also split the mask into 2 if it presents.
        if isinstance(mask,list):
            mask = mask[0]
        return mask
    def compute_output_shape(self, input_shape):
        print(K.int_shape(input_shape))
        return (K.int_shape(input_shape)[1], 1)

class MyFlatten(Layer):
    """Flattens the input. Does not affect the batch size.
    # Arguments
        data_format: A string,
            one of `'channels_last'` (default) or `'channels_first'`.
            The ordering of the dimensions in the inputs.
            The purpose of this argument is to preserve weight
            ordering when switching a model from one data format
            to another.
            `'channels_last'` corresponds to inputs with shape
            `(batch, ..., channels)` while `'channels_first'` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `'channels_last'`.
    # Example
    ```python
        model = Sequential()
        model.add(Conv2D(64, (3, 3),
                         input_shape=(3, 32, 32), padding='same',))
        # now: model.output_shape == (None, 64, 32, 32)
        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, data_format=None, **kwargs):
        super(MyFlatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.data_format = conv_utils.normalize_data_format(data_format)
    def compute_mask(self, inputs, mask=None):
        # Also split the mask into 2 if it presents.
        return mask
    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '). '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))

    def call(self, inputs, mask=None):
        if self.data_format == 'channels_first':
            # Ensure works for any dim
            permutation = [0]
            permutation.extend([i for i in
                                range(2, K.ndim(inputs))])
            permutation.append(1)
            inputs = K.permute_dimensions(inputs, permutation)
            val = K.batch_flatten(inputs)
            if mask is not None:
                val *= K.cast(mask,K.floatx())
        return val

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(MyFlatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class MyReshape(Layer):
    """Reshapes an output to a certain shape.
    Arguments:
    target_shape: Target shape. Tuple of integers,
        does not include the samples dimension (batch size).
    Input shape:
    Arbitrary, although all dimensions in the input shaped must be fixed.
    Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
    Output shape:
    `(batch_size,) + target_shape`
    Example:
    ```python
    # as first layer in a Sequential model
    model = Sequential()
    model.add(Reshape((3, 4), input_shape=(12,)))
    # now: model.output_shape == (None, 3, 4)
    # note: `None` is the batch dimension
    # as intermediate layer in a Sequential model
    model.add(Reshape((6, 2)))
    # now: model.output_shape == (None, 6, 2)
    # also supports shape inference using `-1` as dimension
    model.add(Reshape((-1, 2, 2)))
    # now: model.output_shape == (None, None, 2, 2)
    ```
    """

    def __init__(self, target_shape, **kwargs):
        super(MyReshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Find and replace a missing dimension in an output shape.
        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`
        Arguments:
            input_shape: Shape of array being reshaped
            output_shape: Desired shape of the array with at most
            a single -1 which indicates a dimension that should be
            derived from the input shape.
        Returns:
            The new output shape with a -1 replaced with its computed value.
        Raises:
            ValueError: If the total array size of the output_shape is
            different than the input_shape, or more than one unknown dimension
            is specified.
        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)
        return output_shape

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if None in input_shape[1:]:
            output_shape = [input_shape[0]]
            # input shape (partially) unknown? replace -1's with None's
            output_shape += tuple(s if s != -1 else None for s in self.target_shape)
        else:
            output_shape = [input_shape[0]]
            output_shape += self._fix_unknown_dimension(input_shape[1:],
                                                        self.target_shape)
        return tensor_shape.TensorShape(output_shape)

    def compute_mask(self, inputs, mask=None):
        # Also split the mask into 2 if it presents.
        if isinstance(mask,list):
            mask = mask[0]
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            if isinstance(mask,list):
                mask = mask[0]
                print("Hurray!!")
            inputs *=K.cast(mask,K.floatx())
        return array_ops.reshape(inputs,
                                    (array_ops.shape(inputs)[0],) + self.target_shape)

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(MyReshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape  