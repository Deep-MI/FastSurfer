#!/usr/bin/env python3


# Copyright 2021 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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



# functions to align paired point sets 
# - find_rotation
# - find_rigid
# - find_affine

# IMPORTS
import numpy as np



def rmat2angles(R):
    # Extracts rotation angles (alpha,beta,gamma) in FreeSurfer format (mris_register)
    # from a rotation matrix
    alpha = np.degrees(-np.arctan2(R[1,0],R[0,0]))
    beta  = np.degrees(np.arcsin(R[2,0]))
    gamma = np.degrees(np.arctan2(R[2,1],R[2,2]))
    return (alpha,beta,gamma)


def angles2rmat(alpha, beta, gamma):
    # Converts FreeSurfer angles (alpha,beta,gamma) in degrees to a rotation matrix
    sa = np.sin(np.radians(alpha))
    sb = np.sin(np.radians(beta))
    sg = np.sin(np.radians(gamma))
    ca = np.cos(np.radians(alpha))
    cb = np.cos(np.radians(beta))
    cg = np.cos(np.radians(gamma))
    R = np.array([[ ca*cb, cg*sa-ca*sb*sg, -ca*cg*sb-sa*sg],
                  [-cb*sa, ca*cg+sa*sb*sg,  cg*sa*sb-ca*sg],
                  [    sb,          cb*sg,           cb*cg]])
    return R


def find_rotation(p_mov, p_dst):
    if p_mov.shape != p_dst.shape:
        raise ValueError("Shape of points should be identical, but mov = {}, dst = {} expecting Nx3".format(p_mov.shape,p_dst.shape))    
    # average SSD
    #dd = p_mov-p_dst
    #print("Initial avg SSD: {}".format(np.sum(dd*dd)/p_mov.shape[0]))
    # find rotation
    H = np.dot(p_mov.T, p_dst)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    m = p_mov.shape[1]
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    #print("Rotation Matrix: \n{}".format(R))
    # average SSD after rotation
    #dd = np.transpose(R @ np.transpose(p_mov)) - p_dst
    #print("Final avg SSD: {}".format(np.sum(dd*dd)/p_mov.shape[0]))
    #print("Angles FS format alpha, beta, gamma: {}".format(mat2angle(R)))
    return R



def find_rigid(p_mov, p_dst):
    if p_mov.shape != p_dst.shape:
        raise ValueError("Shape of points should be identical, but mov = {}, dst = {} expecting Nx3".format(p_mov.shape,p_dst.shape))        # average SSD
    # translate points to be centered around origin
    centroid_mov = np.mean(p_mov, axis=0)
    centroid_dst = np.mean(p_dst, axis=0)
    pn_mov = p_mov - centroid_mov
    pn_dst = p_dst - centroid_dst
    # find rotation of point pairs
    R = find_rotation(pn_mov,pn_dst)
    # get translation
    t = centroid_dst.T - np.dot(R,centroid_mov.T)
    # homogeneous transformation
    m = p_mov.shape[1]
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    # compute disteances
    dd = p_mov-p_dst
    print("Initial avg SSD: {}".format(np.sum(dd*dd)/p_mov.shape[0]))    
    dd = (np.transpose(R @ np.transpose(p_mov)) + t)  - p_dst
    print("Final avg SSD: {}".format(np.sum(dd*dd)/p_mov.shape[0]))
    #return T, R, t
    return T

def find_affine(p_mov, p_dst):
    # find affine by least squares solution of overdetermined system
    # (assuming we have more than 4 point pairs)
    from scipy.linalg import pinv
    if p_mov.shape != p_dst.shape:
        raise ValueError("Shape of points should be identical, but mov = {}, dst = {} expecting Nx3".format(p_mov.shape,p_dst.shape))        # average SSD
    n = len(p_mov)
    # Solve overdetermined system for the three rows of 
    # affine matrix in one step (same matrix A for different b=cols_of_p_dst)
    A = np.hstack([p_mov, np.ones((n,1))])
    L, _, _, _ = np.linalg.lstsq(A, p_dst,rcond=None)
    T = np.vstack([np.transpose(L),np.array((0.0,0.0,0.0,1.0))])
    return T


