import optparse
import os
import sys
import nibabel.freesurfer.io as fs
import numpy as np
import math
from scipy import sparse
from scipy.sparse.linalg import eigsh

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


def computeABtria(v, t, lump=False):
    """
    computeABtria(v,t) computes the two sparse symmetric matrices representing
           the Laplace Beltrami Operator for a given triangle mesh using
           the linear finite element method (assuming a closed mesh or 
           the Neumann boundary condition).

    Inputs:   v - vertices : list of lists of 3 floats
              t - triangles: list of lists of 3 int of indices (>=0) into v array

    Outputs:  A - sparse sym. (n x n) positive semi definite numpy matrix 
              B - sparse sym. (n x n) positive definite numpy matrix (inner product)

    Can be used to solve sparse generalized Eigenvalue problem: A x = lambda B x
    or to solve Poisson equation: A x = B f (where f is function on mesh vertices)
    or to solve Laplace equation: A x = 0
    or to model the operator's action on a vector x:   y = B\(Ax) 
    """
    import sys
    v = np.array(v)
    t = np.array(t)
    # Compute vertex coordinates and a difference vector for each triangle:
    t1 = t[:, 0]
    t2 = t[:, 1]
    t3 = t[:, 2]
    v1 = v[t1, :]
    v2 = v[t2, :]
    v3 = v[t3, :]
    v2mv1 = v2 - v1
    v3mv2 = v3 - v2
    v1mv3 = v1 - v3
    # Compute cross product and 4*vol for each triangle:
    cr = np.cross(v3mv2, v1mv3)
    vol = 2 * np.sqrt(np.sum(cr * cr, axis=1))
    # zero vol will cause division by zero below, so set to small value:
    vol_mean = 0.0001 * np.mean(vol)
    vol[vol < sys.float_info.epsilon] = vol_mean
    # compute cotangents for A
    # using that v2mv1 = - (v3mv2 + v1mv3) this can also be seen by 
    # summing the local matrix entries in the old algorithm
    A12 = np.sum(v3mv2 * v1mv3, axis=1) / vol
    A23 = np.sum(v1mv3 * v2mv1, axis=1) / vol
    A31 = np.sum(v2mv1 * v3mv2, axis=1) / vol
    # compute diagonals (from row sum = 0)
    A11 = -A12 - A31
    A22 = -A12 - A23
    A33 = -A31 - A23
    # stack columns to assemble data
    localA = np.column_stack((A12, A12, A23, A23, A31, A31, A11, A22, A33))
    I = np.column_stack((t1, t2, t2, t3, t3, t1, t1, t2, t3))
    J = np.column_stack((t2, t1, t3, t2, t1, t3, t1, t2, t3))
    # Flatten arrays:
    I = I.flatten()
    J = J.flatten()
    localA = localA.flatten()
    # Construct sparse matrix:
    # A = sparse.csr_matrix((localA, (I, J)))
    A = sparse.csc_matrix((localA, (I, J)))
    if not lump:
        # create b matrix data (account for that vol is 4 times area)
        Bii = vol / 24
        Bij = vol / 48
        localB = np.column_stack((Bij, Bij, Bij, Bij, Bij, Bij, Bii, Bii, Bii))
        localB = localB.flatten()
        B = sparse.csc_matrix((localB, (I, J)))
    else:
        # when lumping put all onto diagonal  (area/3 for each vertex)
        Bii = vol / 12
        localB = np.column_stack((Bii, Bii, Bii))
        I = np.column_stack((t1, t2, t3))
        I = I.flatten()
        localB = localB.flatten()
        B = sparse.csc_matrix((localB, (I, I)))

    return A, B


def laplaceTria(v, t, k=10):
    """
    Compute linear finite-element method Laplace-Beltrami spectrum
    """
    from scipy.sparse.linalg import LinearOperator, eigsh, splu
    useCholmod = True
    try:
        from sksparse.cholmod import cholesky
    except ImportError:
        useCholmod = False
    if useCholmod:
        print("Solver: cholesky decomp - performance optimal ...")
    else:
        print("Package scikit-sparse not found (Cholesky decomp)")
        print("Solver: spsolve (LU decomp) - performance not optimal ...")
    # import numpy as np
    # from shapeDNA import computeABtria
    A, M = computeABtria(v, t, lump=True)
    # turns out it is much faster to use cholesky and pass operator
    sigma = -0.01
    if useCholmod:
        chol = cholesky(A - sigma * M)
        OPinv = LinearOperator(matvec=chol, shape=A.shape, dtype=A.dtype)
    else:
        lu = splu(A - sigma * M)
        OPinv = LinearOperator(matvec=lu.solve, shape=A.shape, dtype=A.dtype)
    eigenvalues, eigenvectors = eigsh(A, k, M, sigma=sigma, OPinv=OPinv)
    return eigenvalues, eigenvectors


def exportEV(d,outfile):
    """
    Save EV file
    
    usage: exportEV(data,outfile)

    """

    # open file
    try:
        f = open(outfile,'w')
    except IOError:
        print("[File "+outfile+" not writable]")
        return

    # check data structure
    if not 'Eigenvalues' in d:
        print("ERROR: no Eigenvalues specified")
        exit(1)
    
    # ...

    #
    if 'Creator' in d: f.write(' Creator: '+d['Creator']+'\n')
    if 'File' in d: f.write(' File: '+d['File']+'\n')
    if 'User' in d: f.write(' User: '+d['User']+'\n')
    if 'Refine' in d: f.write(' Refine: '+str(d['Refine'])+'\n')
    if 'Degree' in d: f.write(' Degree: '+str(d['Degree'])+'\n')
    if 'Dimension' in d: f.write(' Dimension: '+str(d['Dimension'])+'\n')
    if 'Elements' in d: f.write(' Elements: '+str(d['Elements'])+'\n')
    if 'DoF' in d: f.write(' DoF: '+str(d['DoF'])+'\n')
    if 'NumEW' in d: f.write(' NumEW: '+str(d['NumEW'])+'\n')
    f.write('\n')
    if 'Area' in d: f.write(' Area: '+str(d['Area'])+'\n')    
    if 'Volume' in d: f.write(' Volume: '+str(d['Volume'])+'\n')
    if 'BLength' in d: f.write(' BLength: '+str(d['BLength'])+'\n')
    if 'EulerChar' in d: f.write(' EulerChar: '+str(d['EulerChar'])+'\n')
    f.write('\n')
    if 'TimePre' in d: f.write(' Time(Pre) : '+str(d['TimePre'])+'\n')
    if 'TimeCalcAB' in d: f.write(' Time(calcAB) : '+str(d['TimeCalcAB'])+'\n')
    if 'TimeCalcEW' in d: f.write(' Time(calcEW) : '+str(d['TimeCalcEW'])+'\n')
    if 'TimePre' in d and 'TimeCalcAB' in d and 'TimeCalcEW' in d:
        f.write(' Time(total ) : '+str(d['TimePre']+d['TimeCalcAB']+d['TimeCalcEW'])+'\n')

    f.write('\n')
    f.write('Eigenvalues:\n')
    f.write('{ '+' ; '.join(map(str,d['Eigenvalues']))+' }\n') # consider precision
    f.write('\n')
    
    if 'Eigenvectors' in d:
        f.write('Eigenvectors:\n')
        #f.write('sizes: '+' '.join(map(str,d['EigenvectorsSize']))+'\n')
	# better compute real sizes from eigenvector array?
        f.write('sizes: '+' '.join(map(str,d['Eigenvectors'].shape))+'\n')
        f.write('\n')
        f.write('{ ')
        for i in range(np.shape(d['Eigenvectors'])[1]-1):
            f.write('(')
            f.write(','.join(map(str,d['Eigenvectors'][:,i])))
            f.write(') ;\n')
        f.write('(')
        f.write(','.join(map(str,d['Eigenvectors'][:,np.shape(d['Eigenvectors'])[1]-1])))
        f.write(') }\n')

    # close file
    f.close()


def get_tria_areas(v,t):
    """
    Computes the surface area of triangles using cross product of edges.
    
    Inputs:   v - vertices   List of lists of 3 float coordinates
              t - trias      List of lists of 3 int of indices (>=0) into v array
                             
    Output:   areas          Areas of each triangle
    """
    v1 = v[t[:, 0], :]
    v2 = v[t[:, 1], :]
    v3 = v[t[:, 2], :]
    v2mv1 = v2 - v1
    v3mv1 = v3 - v1
    cr = np.cross(v2mv1, v3mv1)
    areas = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
    return areas


def get_area(v, t):
    """
    Computes the surface area of triangle mesh using cross product of edges.
    
    Inputs:   v - vertices   List of lists of 3 float coordinates
              t - trias      List of lists of 3 int of indices (>=0) into v array
                             
    Output:   area           Total surface area
    """
    areas = get_tria_areas(v, t)
    area = np.sum(areas)
    return area


def get_volume(v, t):
    """
    Computes the volume of closed triangle mesh, summing tetrahedra at origin
    
    Inputs:   v - vertices   List of lists of 3 float coordinates
              t - trias      List of lists of 3 int of indices (>=0) into v array
                             
    Output:   volume         Total enclosed volume
    """
    # if not is_closed(t):
    #    return 0.0
    v1 = v[t[:, 0], :]
    v2 = v[t[:, 1], :]
    v3 = v[t[:, 2], :]
    v2mv1 = v2 - v1
    v3mv1 = v3 - v1
    cr = np.cross(v2mv1, v3mv1)
    spatvol = np.sum(v1 * cr, axis=1)
    # spatv = sum(cr .* v1,2);
    vol = np.sum(spatvol) / 6.0
    return vol


def get_flipped_area(v, t):
    v1 = v[t[:, 0], :]
    v2 = v[t[:, 1], :]
    v3 = v[t[:, 2], :]
    v2mv1 = v2 - v1
    v3mv1 = v3 - v1
    cr = np.cross(v2mv1, v3mv1)
    spatvol = np.sum(v1 * cr, axis=1)
    areas = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
    area = np.sum(areas[np.where(spatvol < 0)])
    return area


def sphericalProject(v, t):
    """
    spherical(v,t) computes the first three non-constant eigenfunctions
           and then projects the spectral embedding onto a sphere. This works
           when the first functions have a single closed zero level set,
           splitting the mesh into two domains each. Depending on the original 
           shape triangles could get inverted. We also flip the functions 
           according to the axes that they are aligned with for the special 
           case of brain surfaces in FreeSurfer coordiates.

    Inputs:   v - vertices : list of lists of 3 floats
              t - triangles: list of lists of 3 int of indices (>=0) into v array

    Outputs:  v - vertices : of projected mesh
              t - triangles: same as before

    """
    
    #print("Tria min vidx: {}  max vidx: {}".format(np.min(t), np.max(t)))
    #print("Tria range: {}".format(np.max(t)- np.min(t)+1))
    #print("Vnumber: {}".format(v.shape[0]))
    
    evals, evecs = laplaceTria(v, t, k=4)
    
    #debug=True
    debug=False
    
    if debug:
        data = dict()
        data['Eigenvalues'] = evals
        data['Eigenvectors'] = evecs
        data['Creator'] = 'spherically_project.py'
        data['Refine'] = 0
        data['Degree'] = 1
        data['Dimension'] = 2
        data['Elements'] = t.shape[0]
        data['DoF'] = evecs.shape[0]
        data['NumEW'] = 4
        exportEV(data,'debug.ev')


    # flip efuncs to align to coordinates consistently
    ev1 = evecs[:,1]
    #ev1maxi = np.argmax(ev1)
    #ev1mini = np.argmin(ev1)
    #cmax = v[ev1maxi,:]
    #cmin = v[ev1mini,:]
    cmax1 = np.mean(v[ev1>0.5*np.max(ev1),:],0)
    cmin1 = np.mean(v[ev1<0.5*np.min(ev1),:],0)
    ev2 = evecs[:,2]
    cmax2 = np.mean(v[ev2>0.5*np.max(ev2),:],0)
    cmin2 = np.mean(v[ev2<0.5*np.min(ev2),:],0)
    ev3 = evecs[:,3]
    cmax3 = np.mean(v[ev3>0.5*np.max(ev3),:],0)
    cmin3 = np.mean(v[ev3<0.5*np.min(ev3),:],0)
    
    # we trust ev 1 goes from front to back
    l11 = abs(cmax1[1]-cmin1[1])
    l21 = abs(cmax2[1]-cmin2[1])
    l31 = abs(cmax3[1]-cmin3[1])
    if (l11 < l21 or l11 < l31):
        print("ERROR: direction 1 should be (anterior -posterior) but is not!")
        print("  debug info: {} {} {} ".format(l11, l21, l31))
        sys.exit(1)
	
    # only flip direction if necessary
    print("ev1 min: {}  max {} ".format(cmin1,cmax1))
    # axis 1 = y is aligned with this function (for brains in FS space)
    v1 = cmax1 - cmin1
    if (cmax1[1] < cmin1[1]):
        ev1 = -1 * ev1;
        print("inverting direction 1 (anterior - posterior)")
    l1 = abs(cmax1[1]-cmin1[1])
    
    # for ev2 and ev3 there could be also a swap of the two
    l22 = abs(cmax2[2]-cmin2[2])
    l32 = abs(cmax3[2]-cmin3[2])
    #usually ev2 should be superior inferior, if ev3 is better in that direction, swap
    if ( l22 < l32 ):
        print("swapping direction 2 and 3")
        ev2, ev3 = ev3, ev2
        cmax2, cmax3 = cmax3, cmax2
        cmin2, cmin3 = cmin3, cmin2
    l23 = abs(cmax2[0]-cmin2[0])
    l33 = abs(cmax3[0]-cmin3[0])
    if ( l33 < l23 ): 
        print("WARNING: direction 3 wants to swap with 2, but cannot")
    
    print("ev2 min: {}  max {} ".format(cmin2,cmax2))
    # axis 2 = z is aligned with this function (for brains in FS space)
    v2 = cmax2 - cmin2
    if (cmax2[2] < cmin2[2]):
        ev2 = -1 * ev2;
        print("inverting direction 2 (superior - inferior)")
    l2 = abs(cmax2[2]-cmin2[2])
        
    print("ev3 min: {}  max {} ".format(cmin3,cmax3))
    # axis 0 = x is aligned with this function (for brains in FS space)
    v3 = cmax3 - cmin3
    if (cmax3[0] < cmin3[0]):
        ev3 = -1 * ev3;
        print("inverting direction 3 (right - left)")
    l3 = abs(cmax3[0] - cmin3[0])
    
    v1 = v1 * (1.0 / np.sqrt(np.sum(v1 * v1)))
    v2 = v2 * (1.0 / np.sqrt(np.sum(v2 * v2)))
    v3 = v3 * (1.0 / np.sqrt(np.sum(v3 * v3)))
    spatvol = abs(np.dot(v1, np.cross(v2, v3)))
    print("spat vol: {}".format(spatvol))

    mvol = get_volume(v, t)
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


    # project to sphere and scaled to have the same scale/origin as FS:
    dist = np.sqrt(np.square(ev1) + np.square(ev2) + np.square(ev3))
    v[:, 0] = 100 * (ev3 / dist)
    v[:, 1] = 100 * (ev1 / dist)
    v[:, 2] = 100 * (ev2 / dist)

    svol = get_area(v, t) / (4.0 * math.pi * 10000)
    print("sphere area fraction: {} ".format(svol))

    flippedarea = get_flipped_area(v, t) / (4.0 * math.pi * 10000)
    if (flippedarea > 0.95):
        print("ERROR: global normal flip, exiting ..")
        sys.exit(1)
        #print("WARNING: flipping normals globally!")
        #tt = np.copy(t[:,1])
        #t[:,1] = np.copy(t[:,2])
        #t[:,2] = tt
        #flippedarea=get_flipped_area(v,t)/(4.0*math.pi*10000)
    
    print("flipped area fraction: {} ".format(flippedarea))

    if svol < 0.99:
        print("ERROR: sphere area fraction should be above .99, exiting ..")
        sys.exit(1)

    if flippedarea > 0.004:
        print("ERROR: flipped area fraction should be below .0004, exiting ..")
        sys.exit(1)

    # here we finally check also the spat vol (orthogonality of direction vectors)
    # we could stop earlier, but most failure cases will be covered by the svol and
    # flipped area which can be better interpreted than spatvol
    if spatvol < 0.6:
        print("ERROR: spat vol (orthogonality) should be above .6, exiting ..")
        sys.exit(1)

    return v, t


def spherically_project_surface(insurf, outsurf):
    """ (string) -> None
    takes path to insurf, spherically projects it, outputs it to outsurf
    """
    surf = fs.read_geometry(insurf, read_metadata=True)
    projected = sphericalProject(surf[0], surf[1])
    fs.write_geometry(outsurf, projected[0], projected[1], volume_info=surf[2])


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()
    surf_to_project = options.input_surf
    projected_surf = options.output_surf

    print("Reading in surface: {} ...".format(surf_to_project))
    spherically_project_surface(surf_to_project, projected_surf)
    print ("Outputing spherically projected surface: {}".format(projected_surf))

    sys.exit(0)
