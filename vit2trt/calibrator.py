#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import glob

from torchvision import transforms, datasets
from PIL import Image

def image_preprocess(args, path):
    # torchvision.io.read_image("data/CIFAR-10-images-master/test/airplane/0000.jpg")
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = transform(img)
    #  print (img.shape)
    #  print (img)
    return img

class ViTCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self, args, cache_file, batch_size, num_inputs):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8LegacyCalibrator.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        
        # Read inputs from the calibration data directory
        self.imgs = []
        self.current_index = 0
        
        if args.calib_path:
            import glob
            # Get all image files from the calibration path
            img_files = glob.glob(os.path.join(args.calib_path, "*/*"))[:num_inputs]
            for img_file in img_files:
                img = image_preprocess(args, img_file)
                self.imgs.append(img.numpy())
            
            # Convert to numpy array
            self.imgs = np.array(self.imgs, dtype=np.float32)
            print(f"Loaded {len(self.imgs)} images for calibration")

        # Allocate enough memory for a whole batch.
        if len(self.imgs) > 0:
            self.device_input = cuda.mem_alloc(self.batch_size * self.imgs[0].nbytes)
        else:
            self.device_input = None

    def free(self):
        if self.device_input is not None:
            self.device_input.free()

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index >= len(self.imgs):
            # Calibration is finished
            return None
        
        # Calculate batch size for the current iteration
        remaining_images = len(self.imgs) - self.current_index
        current_batch_size = min(self.batch_size, remaining_images)
        
        # Prepare batch data
        batch_imgs = []
        for i in range(current_batch_size):
            batch_imgs.append(self.imgs[self.current_index + i])
        
        # Stack images into a batch
        batch_data = np.stack(batch_imgs, axis=0)
        
        # Copy batch data to device
        cuda.memcpy_htod(self.device_input, batch_data.ravel())
        
        # Update current index
        self.current_index += current_batch_size
        
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None

