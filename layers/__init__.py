"""
TensorLayer provides rich layer implementations trailed for
various benchmarks and domain-specific problems. In addition, we also
support transparent access to native TensorFlow parameters.
For example, we provide not only layers for local response normalization, but also
layers that allow user to apply ``tf.nn.lrn`` on ``network.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`__.
"""

from .activation import *

from .convolution import *
from .deconvolution import *

from .dense import *

from .inception_blocks import *
from .mobilenet_blocks import *
from .squeezenet_blocks import *
