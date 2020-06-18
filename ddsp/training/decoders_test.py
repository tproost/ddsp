# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for ddsp.training.decoders."""

from absl.testing import parameterized
from ddsp.core import hz_to_midi
from ddsp.spectral_ops import F0_RANGE
from ddsp.spectral_ops import LD_RANGE
import ddsp.training.decoders as decoders
import numpy as np
import tensorflow.compat.v2 as tf


class DilatedConvDecoderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some common default values for decoder."""
    super().setUp()
    # For decoder.
    self.ch = 4
    self.layers_per_stack = 3
    self.stacks = 2
    self.output_splits = (('amps', 1), ('harmonic_distribution', 10),
                          ('noise_magnitudes', 10))

    # For audio features and conditioning.
    self.f0_hz_val = 440
    self.loudness_db_val = -50
    self.frame_rate = 100
    self.length_in_sec = 0.20
    self.time_steps = int(self.frame_rate * self.length_in_sec)

  def _gen_dummy_conditioning(self):
    """Generate dummy scaled f0 and ld conditioning."""
    conditioning = {}
    # Generate dummy `f0_hz` with batch and channel dims.
    f0_hz = np.repeat(self.f0_hz_val,
                      self.length_in_sec * self.frame_rate)[np.newaxis, :,
                                                            np.newaxis]
    conditioning['f0_scaled'] = hz_to_midi(f0_hz) / F0_RANGE
    # Generate dummy `loudness_db` with batch and channel dims.
    loudness_db = np.repeat(self.loudness_db_val,
                            self.length_in_sec * self.frame_rate)[np.newaxis, :,
                                                                  np.newaxis]
    conditioning['ld_scaled'] = (loudness_db / LD_RANGE) + 1.0
    return conditioning

  def test_correct_output_splits_and_shapes_dilated_conv_decoder(self):
    decoder = decoders.DilatedConvDecoder(
        ch=self.ch,
        layers_per_stack=self.layers_per_stack,
        stacks=self.stacks,
        conditioning_keys=None,
        output_splits=self.output_splits)

    conditioning = self._gen_dummy_conditioning()
    output = decoder(conditioning)
    for output_name, output_dim in self.output_splits:
      dummy_output = np.zeros((1, self.time_steps, output_dim))
      self.assertShapeEqual(dummy_output, output[output_name])


if __name__ == '__main__':
  tf.test.main()
