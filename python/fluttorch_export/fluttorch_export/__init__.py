"""Export PyTorch models for consumption by Flutter.

Three artifacts come out of a single export, and they are meant to travel
together:

* the runtime artifact the device executes,
* a manifest describing shapes, dtypes, preprocessing and the weight hash,
* a bundle of goldens -- reference inputs and the outputs the source model
  produced for them.

The manifest is what makes the Dart side typed. The goldens are what make
quantization drift visible instead of silent.
"""

__version__ = "0.0.1.dev0"
