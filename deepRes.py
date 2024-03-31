#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:     Erney Ramirez Aportela (CSIC)
 *
 *  Updated by: 
 *         J.M. de la Rosa Trevin (2024-03-28)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'

 *
 ***************************************************************************/
"""

from keras.models import load_model
from keras.utils import Sequence
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

import mrcfile

# The method accepts as input a 3D crioEM map and the mask
# both with sampling rate of 1 A/pixel for network 1 or 0.5 A/pixel for network 2

VOL_TH = 0.00015
MASK_TH = 0.15


class VolumeManager(Sequence):
    def __init__(self, volArray, maskArray, boxSize=13, maxSize=1000):
        self.boxSize = boxSize
        self.maxSize = maxSize
        self.volArray = volArray
        self.maskArray = maskArray
        print(">>> DEBUG: ", f"volShape: {volArray.shape}")
        print(">>> DEBUG: ", f"maskArray: {maskArray.shape}")
        vx = 0
        for z, y, x in self.iterBox():
            if self.evalPos(z, y, x):
                vx += 1

        self.st = vx // self.maxSize + (vx % self.maxSize)
        h = self.boxSize // 2
        self.pos = np.asarray([h, h, h])
        if self.maskArray[h, h, h] <= MASK_TH:
            self.advance()

        print(">>> DEBUG: ", f"volShape: {self.volArray.shape}")
        print(">>> DEBUG: ", f"maskArray: {self.maskArray.shape}")

    def __len__(self):
        return self.maxSize

    @property
    def numberOfBlocks(self):
        return self.st

    def advancePos(self):
        """ Return new the position when need to move, None otherwise. """
        advance = True
        zdim, ydim, xdim = self.volArray.shape
        h = self.boxSize // 2
        print(f">>> DEBUG: BEFORE \n\tpos = {self.pos}")
        z, y, x = self.pos
        x += 1
        if x == xdim - h:
            x = h
            y += 1
            if y == ydim - h:
                y = h
                z += 1
                if z == zdim - h:
                    advance = False
        self.pos[:] = z, y, x
        print(f">>> DEBUG: AFTER \n\tpos = {self.pos}, advance: {advance}")
        return advance

    def advance(self):
        ok = self.advancePos()
        while ok:
            z, y, x = self.pos
            if self.evalPos(z, y, x):
                break
            ok = self.advancePos()
        return ok

    def evalPos(self, z, y, x, rest=0):
        M = self.maskArray
        V = self.volArray
        try:
            v = M[z, y, x] > MASK_TH and V[z, y, x] > VOL_TH and (z + y + x) % 2 == rest
        except Exception as e:
            print(f">>> DEBUG: EXCEPTION ON INDEX {(z, y, x)}")
            raise e
        return v

    def getBox(self):
        """ Get a box (with selected boxSize) at the given coordinates. """
        z, y, x = self.pos
        h1 = self.boxSize // 2
        h2 = self.boxSize - h1
        box = self.volArray[z - h1:z + h2, y - h1:y + h2, x - h1:x + h2]
        box = box / np.linalg.norm(box)
        return box

    def __getitem__(self, idx):
        count = 0
        batchX = []
        ok = True 
        while count < self.maxSize and ok:
            batchX.append(self.getBox())
            print(">>>> --------  idx", idx)
            ok = self.advance()
            count += 1
        batchX = np.asarray(batchX).astype("float32")
        batchX = batchX.reshape(count, batchX.shape[1], batchX.shape[2], batchX.shape[3], 1)
        return batchX

    def iterBox(self, offset=0):
        """ Iterate z, y, x over box boundaries. """
        h = self.boxSize // 2
        zdim, ydim, xdim = self.volArray.shape
        for z in range(h + offset, zdim - h):
            for y in range(h + offset, ydim - h):
                for x in range(h + offset, xdim - h):
                    yield z, y, x

    def getOutput(self, modelNumber, sampling, Y):
        V = self.volArray
        OV = V * 0

        samplingTh = 2.5 if modelNumber == 1 else 1.5
        minValue = max(sampling * 2, samplingTh)
        maxValue = 12.9 if modelNumber == 1 else 5.9

        idx = 0
        for z, y, x in self.iterBox():
            if self.evalPos(z, y, x):
                OV[z, y, x] = max(min(Y[idx], maxValue), minValue)
                idx += 1

        for z, y, x in self.iterBox(offset=1):
            if self.evalPos(z, y, x, rest=1):
                col = 0
                ct = 0
                if OV[z + 1, y, x] > 0:
                    col += OV[z + 1, y, x]
                    ct += 1
                if OV[z - 1, y, x] > 0:
                    col += OV[z - 1, y, x]
                    ct += 1
                if OV[z, y + 1, x] > 0:
                    col += OV[z, y + 1, x]
                    ct += 1
                if OV[z, y - 1, x] > 0:
                    col += OV[z, y - 1, x]
                    ct += 1
                if OV[z, y, x + 1] > 0:
                    col += OV[z, y, x + 1]
                    ct += 1
                if OV[z, y, x - 1] > 0:
                    col += OV[z, y, x - 1]
                    ct += 1
                if ct == 0:
                    OV[z, y, x] = 0
                    ct = 1
                else:
                    meansum = col / ct
                    OV[z, y, x] = meansum
        return OV


def main(fnModel, fnVolIn, fnMask, sampling, fnVolOut):
    modelName = os.path.basename(fnModel)
    if modelName.endswith("model_w13.h5"):
        modelNumber = 1
    elif modelName.endswith("model_w7.h5"):
        modelNumber = 2
    else:
        raise Exception("Unknown model {modelName}, it should be either"
                        "model_w13.h5 or model_w7.h5")
    model = load_model(fnModel)
    mrcVol = mrcfile.open(fnVolIn)
    mrcMask = mrcfile.open(fnMask)
    manager = VolumeManager(mrcVol.data, mrcMask.data)
    Y = model.predict_generator(manager, manager.numberOfBlocks)
    outputArray = manager.getOutput(modelNumber, sampling, Y)

    # Write output volume
    with mrcfile.open(fnVolOut, mode='w+') as mrc:
        mrc.set_data(outputArray)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine local resolution "
                                                 "of the input map")
    parser.add_argument("-dl", "--dl_model",
                        help="input deep learning model", required=True)
    parser.add_argument("-i", "--map",
                        help="input map", required=True)
    parser.add_argument("-m", "--mask",
                        help="input mask", required=True)
    parser.add_argument("-s", "--sampling",
                        help="sampling rate", required=True, type=float)
    parser.add_argument("-o", "--output",
                        help="output resolution map", required=True)
    args = parser.parse_args()

    main(args.dl_model, args.map, args.mask, args.sampling, args.output)


