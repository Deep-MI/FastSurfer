# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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


# IMPORTS
import optparse
import sys
import nibabel.freesurfer.io as fs
import numpy as np
import math
from lapy.DiffGeo import tria_mean_curvature_flow
from lapy.TriaMesh import TriaMesh
from lapy.read_geometry import read_geometry
from lapy.Solver import Solver

HELPTEXT = """
Script to compute ShapeDNA using linear FEM matrices. 

After correcting sign flips, embeds a surface mesh into the spectral domain, 
then projects it onto a unit sphere.  This is scaled and rotated to match the
atlas used for FreeSurfer surface registion.


USAGE:
spherically_project  -i <input_surface> -o <output_surface>


References:

Martin Reuter et al. Discrete Laplace-Beltrami Operators for Shape Analysis and
Segmentation. Computers & Graphics 33(3):381-390, 2009

Martin Reuter et al. Laplace-Beltrami spectra as "Shape-DNA" of surfaces and 
solids Computer-Aided Design 38(4):342-366, 2006

Bruce Fischl at al. High-resolution inter-subject averaging and a coordinate 
system for the cortical surface. Human Brain Mapping 8:272-284, 1999


Dependencies:
    Python 3.5

    Scipy 0.10 or later to solve the generalized eigenvalue problem.
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer surface meshes
    http://nipy.org/nibabel/


Original Author: Martin Reuter
Date: Jan-18-2016


"""

h_input = 'path to input surface'
h_output = 'path to ouput surface, spherically projected'


def options_parse():
    """
    Command line option parser for spherically_project.py
    """
    parser = optparse.OptionParser(version='$Id: spherically_project,v 1.1 2017/01/30 20:42:08 ltirrell Exp $',
                                   usage=HELPTEXT)
    parser.add_option('--input', '-i', dest='input_surf', help=h_input)
    parser.add_option('--output', '-o', dest='output_surf', help=h_output)
    (options, args) = parser.parse_args()

    if options.input_surf is None or options.output_surf is None:
        sys.exit('ERROR: Please specify input and output surfaces')

    return options


def tria_spherical_project(tria, flow_iter=3, debug=False, use_cholmod=True):
    """
    spherical(tria) computes the first three non-constant eigenfunctions
           and then projects the spectral embedding onto a sphere. This works
           when the first functions have a single closed zero level set,
           splitting the mesh into two domains each. Depending on the original
           shape triangles could get inverted. We also flip the functions
           according to the axes that they are aligned with for the special
           case of brain surfaces in FreeSurfer coordinates.

    Inputs:   tria      : TriaMesh
              flow_iter : mean curv flow iterations (3 should be enough)

    Outputs:  tria      : TriaMesh
    """
    if not tria.is_closed():
        raise ValueError('Error: Can only project closed meshes!')

    # sub-function to compute flipped area of trias where normal
    # points towards origin, meaningful for the sphere, centered at zero
    def get_flipped_area(tria):
        v1 = tria.v[tria.t[:, 0], :]
        v2 = tria.v[tria.t[:, 1], :]
        v3 = tria.v[tria.t[:, 2], :]
        v2mv1 = v2 - v1
        v3mv1 = v3 - v1
        cr = np.cross(v2mv1, v3mv1)
        spatvol = np.sum(v1 * cr, axis=1)
        areas = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
        area = np.sum(areas[np.where(spatvol < 0)])
        return area

    fem = Solver(tria, lump=False, use_cholmod=use_cholmod)
    evals, evecs = fem.eigs(k=4)

    if debug:
        data = dict()
        data['Eigenvalues'] = evals
        data['Eigenvectors'] = evecs
        data['Creator'] = 'spherically_project.py'
        data['Refine'] = 0
        data['Degree'] = 1
        data['Dimension'] = 2
        data['Elements'] = tria.t.shape[0]
        data['DoF'] = evecs.shape[0]
        data['NumEW'] = 4
        from lapy.FuncIO import export_ev
        export_ev(data, 'debug.ev')

    # flip efuncs to align to coordinates consistently
    ev1 = evecs[:, 1]
    # ev1maxi = np.argmax(ev1)
    # ev1mini = np.argmin(ev1)
    # cmax = v[ev1maxi,:]
    # cmin = v[ev1mini,:]
    cmax1 = np.mean(tria.v[ev1 > 0.5 * np.max(ev1), :], 0)
    cmin1 = np.mean(tria.v[ev1 < 0.5 * np.min(ev1), :], 0)
    ev2 = evecs[:, 2]
    cmax2 = np.mean(tria.v[ev2 > 0.5 * np.max(ev2), :], 0)
    cmin2 = np.mean(tria.v[ev2 < 0.5 * np.min(ev2), :], 0)
    ev3 = evecs[:, 3]
    cmax3 = np.mean(tria.v[ev3 > 0.5 * np.max(ev3), :], 0)
    cmin3 = np.mean(tria.v[ev3 < 0.5 * np.min(ev3), :], 0)

    # we trust ev 1 goes from front to back
    l11 = abs(cmax1[1] - cmin1[1])
    l21 = abs(cmax2[1] - cmin2[1])
    l31 = abs(cmax3[1] - cmin3[1])
    if l11 < l21 or l11 < l31:
        print("ERROR: direction 1 should be (anterior -posterior) but is not!")
        print("  debug info: {} {} {} ".format(l11, l21, l31))
        # sys.exit(1)
        raise ValueError('Direction 1 should be anterior - posterior')

    # only flip direction if necessary
    print("ev1 min: {}  max {} ".format(cmin1, cmax1))
    # axis 1 = y is aligned with this function (for brains in FS space)
    v1 = cmax1 - cmin1
    if cmax1[1] < cmin1[1]:
        ev1 = -1 * ev1
        print("inverting direction 1 (anterior - posterior)")
    l1 = abs(cmax1[1] - cmin1[1])

    # for ev2 and ev3 there could be also a swap of the two
    l22 = abs(cmax2[2] - cmin2[2])
    l32 = abs(cmax3[2] - cmin3[2])
    # usually ev2 should be superior inferior, if ev3 is better in that direction, swap
    if l22 < l32:
        print("swapping direction 2 and 3")
        ev2, ev3 = ev3, ev2
        cmax2, cmax3 = cmax3, cmax2
        cmin2, cmin3 = cmin3, cmin2
    l23 = abs(cmax2[0] - cmin2[0])
    l33 = abs(cmax3[0] - cmin3[0])
    if l33 < l23:
        print("WARNING: direction 3 wants to swap with 2, but cannot")

    print("ev2 min: {}  max {} ".format(cmin2, cmax2))
    # axis 2 = z is aligned with this function (for brains in FS space)
    v2 = cmax2 - cmin2
    if cmax2[2] < cmin2[2]:
        ev2 = -1 * ev2
        print("inverting direction 2 (superior - inferior)")
    l2 = abs(cmax2[2] - cmin2[2])

    print("ev3 min: {}  max {} ".format(cmin3, cmax3))
    # axis 0 = x is aligned with this function (for brains in FS space)
    v3 = cmax3 - cmin3
    if cmax3[0] < cmin3[0]:
        ev3 = -1 * ev3
        print("inverting direction 3 (right - left)")
    l3 = abs(cmax3[0] - cmin3[0])

    v1 = v1 * (1.0 / np.sqrt(np.sum(v1 * v1)))
    v2 = v2 * (1.0 / np.sqrt(np.sum(v2 * v2)))
    v3 = v3 * (1.0 / np.sqrt(np.sum(v3 * v3)))
    spatvol = abs(np.dot(v1, np.cross(v2, v3)))
    print("spat vol: {}".format(spatvol))

    mvol = tria.volume()
    print("orig mesh vol {}".format(mvol))
    bvol = l1 * l2 * l3
    print("box {}, {}, {} volume: {} ".format(l1, l2, l3, bvol))
    print("box coverage: {}".format(bvol / mvol))

    # we map evN to -1..0..+1 (keep zero level fixed)
    # I have the feeling that this helps a little with the stretching
    # at the poles, but who knows...
    ev1min = np.amin(ev1)
    ev1max = np.amax(ev1)
    ev1[ev1 < 0] /= - ev1min
    ev1[ev1 > 0] /= ev1max

    ev2min = np.amin(ev2)
    ev2max = np.amax(ev2)
    ev2[ev2 < 0] /= - ev2min
    ev2[ev2 > 0] /= ev2max

    ev3min = np.amin(ev3)
    ev3max = np.amax(ev3)
    ev3[ev3 < 0] /= - ev3min
    ev3[ev3 > 0] /= ev3max

    # set evec as new coordinates (spectral embedding)
    vn = np.empty(tria.v.shape)
    vn[:, 0] = ev3
    vn[:, 1] = ev1
    vn[:, 2] = ev2

    # do a few mean curvature flow euler steps to make more convex
    # three should be sufficient
    if flow_iter > 0:
        tflow = tria_mean_curvature_flow(TriaMesh(vn, tria.t), max_iter=flow_iter, use_cholmod=use_cholmod)
        vn = tflow.v

    # project to sphere and scaled to have the same scale/origin as FS:
    dist = np.sqrt(np.sum(vn * vn, axis=1))
    vn = 100 * (vn / dist[:, np.newaxis])

    trianew = TriaMesh(vn, tria.t)
    svol = trianew.area() / (4.0 * math.pi * 10000)
    print("sphere area fraction: {} ".format(svol))

    flippedarea = get_flipped_area(trianew) / (4.0 * math.pi * 10000)
    if flippedarea > 0.95:
        print("ERROR: global normal flip, exiting ..")
        raise ValueError('global normal flip')

    print("flipped area fraction: {} ".format(flippedarea))

    if svol < 0.99:
        print("ERROR: sphere area fraction should be above .99, exiting ..")
        raise ValueError('sphere area fraction should be above .99')

    if flippedarea > 0.0008:
        print("ERROR: flipped area fraction should be below .0008, exiting ..")
        raise ValueError('flipped area fraction should be below .0008')

    # here we finally check also the spat vol (orthogonality of direction vectors)
    # we could stop earlier, but most failure cases will be covered by the svol and
    # flipped area which can be better interpreted than spatvol
    if spatvol < 0.6:
        print("ERROR: spat vol (orthogonality) should be above .6, exiting ..")
        raise ValueError('spat vol (orthogonality) should be above .6')

    return trianew


def spherically_project_surface(insurf, outsurf, use_cholmod=True):
    """ (string) -> None
    takes path to insurf, spherically projects it, outputs it to outsurf
    """
    surf = read_geometry(insurf, read_metadata=True)
    projected = tria_spherical_project(TriaMesh(surf[0], surf[1]), flow_iter=3, use_cholmod=use_cholmod)
    fs.write_geometry(outsurf, projected.v, projected.t, volume_info=surf[2])


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()
    surf_to_project = options.input_surf
    projected_surf = options.output_surf

    print("Reading in surface: {} ...".format(surf_to_project))
    # switching cholmod off will be slower, but does not require scikit sparse cholmod
    spherically_project_surface(surf_to_project, projected_surf, use_cholmod=False)
    print ("Outputing spherically projected surface: {}".format(projected_surf))

    sys.exit(0)
