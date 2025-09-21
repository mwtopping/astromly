import time
from scipy.optimize import curve_fit
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from tqdm import tqdm
import itertools
import numpy as np
import cv2 as cv
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip


from astropy.visualization import ZScaleInterval

def plot_one(image, invert=True):

    scaler = ZScaleInterval()

    # do some preprocessing
    image = np.nan_to_num(image, posinf=0, neginf=0)
    image -= np.min(image)

    limits = scaler.get_limits(image)

    _, ax = plt.subplots()
    cmap="Greys_r"
    if invert:
        cmap="Greys"
    
    ax.imshow(image, 
              aspect='auto', 
              origin='lower', 
              vmin=limits[0], vmax=limits[1]+0.1,
              cmap=cmap)

    return ax

def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function
    
    Parameters:
    -----------
    coords : tuple
        (x, y) coordinates where x and y are meshgrid arrays
    amplitude : float
        Height of the gaussian
    x0, y0 : float
        Center position of the gaussian
    sigma_x, sigma_y : float
        Width of the gaussian in x and y directions
    theta : float
        Rotation angle in radians
    offset : float
        Baseline offset
    
    Returns:
    --------
    z : ndarray
        2D Gaussian evaluated at x, y points
    """
    x, y = coords
    
    # Rotation
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    # Gaussian function
    z = offset + amplitude * np.exp(
        -(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))
    )
    
    return z.ravel()

def fit_gaussian_2d(image):
    """
    Fit a 2D Gaussian to the input image
    
    Parameters:
    -----------
    image : ndarray
        Input image
    
    Returns:
    --------
    popt : ndarray
        Optimal parameters (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
    pcov : ndarray
        Covariance matrix for the parameters
    """
    y, x = np.indices(image.shape)
    
    height = np.max(image) - np.min(image)
    offset = np.min(image)
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    sigma_x = sigma_y = np.sqrt(np.sum((image - offset) * ((x - x_max)**2 + (y - y_max)**2)) / np.sum(image - offset))
    p0 = [height, x_max, y_max, sigma_x, sigma_y, 0, offset]
    
    lower_bounds = [0, 0, 0, 0, 0, -np.pi/2, -np.inf]
    upper_bounds = [np.inf, image.shape[1], image.shape[0], image.shape[1], image.shape[0], np.pi/2, np.inf]
    bounds = (lower_bounds, upper_bounds)
    
    popt, pcov = curve_fit(
        gaussian_2d, 
        (x, y), 
        image.ravel(), 
        p0=p0,
        bounds=bounds
    )
    
    return popt, pcov


def pedastal(img):

    pos_img = img[img!=0]

    return img - np.nanmedian(pos_img), img!=0

def get_noise_level(img, mask=None):
    if mask is None:
        return np.nanstd(img)

    return np.nanstd(img[mask])


def get_star_locs(img, sigma=3, return_image=False, padding=1):
    starttime = time.time()


    sigma_clip = SigmaClip(sigma=5.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(img, (250, 250), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    img = img - bkg.background

    noise = get_noise_level(img)

    masked_img = img.copy()
    masked_img[masked_img < sigma*noise] = 0
    masked_img[masked_img > 0] = 1

    # cast to uint8?
    masked_img = masked_img.astype(np.uint8)

    labels_img = cv.threshold(masked_img, 0, 1, cv.THRESH_BINARY)[1]

    num_labels, labels_img = cv.connectedComponents(masked_img.astype(np.int8), connectivity=4)

    stars = {}
    coo = sparse.coo_matrix(labels_img)

    # go through each of the new labels
    for ii in range(1, num_labels):
        this_coo = (coo==ii).tocoo()
        xinds, yinds = this_coo.row, this_coo.col

        # this will remove hot pixels, etc.
        if len(xinds) <= 8:
            continue

        left = np.min(xinds)
        bottom = np.min(yinds)
        width = np.max(xinds)-left+1
        height = np.max(yinds)-bottom+1

        cutout = img[left:left+width, bottom:bottom+height]

        if np.any(cutout >= 2**16-1):
            continue

        if left < padding or bottom < padding:
            continue

        stars[ii] = (left-padding,
                     bottom-padding,
                     width+2*padding,
                     height+2*padding)

    finalxs = []
    finalys = []

    for inum, ii in enumerate(stars.keys()):
        left   = stars[ii][0]
        bottom = stars[ii][1]
        width  = stars[ii][2]
        height = stars[ii][3]

        cutout = img[left:left+width, bottom:bottom+height]

        try:
            popt, pcov = fit_gaussian_2d(cutout)
        except (RuntimeError, ValueError) as e:
            print("ERROR", e)
            continue

        x = popt[1]
        y = popt[2]

        finalxs.append(float(bottom+x))
        finalys.append(float(left+y))


    return np.array(finalxs), np.array(finalys)



#def get_total_shift_inv(shift):
#
#    tot_shift_h = [0, 0]
#    tot_shift_v = [0, 0]
#    if shift[0]>shift[1]:
#        tot_shift_v = [0, shift[0]-shift[1]]
#    else:
#        tot_shift_v = [shift[1]-shift[0], 0]
#
#    if shift[2]>shift[3]:
#        tot_shift_h = [0, shift[2]-shift[3]]
#    else:
#        tot_shift_h = [shift[3]-shift[2], 0]
#
#
#    return [tot_shift_v[0], tot_shift_v[1],
#            tot_shift_h[0], tot_shift_h[1]]
#
## currently unused
#def transform_points(rot, shift, points, recenter=False):
#
#    if recenter:
#        mean = np.mean(points, axis=0)
#        points -= mean
#    else:
#        mean = 0
#
#
#    ones = np.ones(np.shape(points)[0])
#    points = np.column_stack((points, ones))
#
#    rotmat = np.matrix([[np.cos(rot),-1*np.sin(rot), shift[0]],
#                        [np.sin(rot),np.cos(rot), shift[1]],
#                        [0,0,1]])
#    outpoints = (rotmat*points.T).T[:,:2]
#    if recenter:
#        outpoints += mean
#    return outpoints 
#
#

def get_angles(xs, ys):

    a = np.sqrt((xs[0]-xs[1])**2 + (ys[0]-ys[1])**2)
    b = np.sqrt((xs[1]-xs[2])**2 + (ys[1]-ys[2])**2)
    c = np.sqrt((xs[0]-xs[2])**2 + (ys[0]-ys[2])**2)
    
    a1 = np.arccos((a*a + c*c - b*b) / (2*a*c))#opposite point 0
    a2 = np.arccos((a*a + b*b - c*c) / (2*a*b))#opposite point 1
    a3 = np.arccos((b*b + c*c - a*a) / (2*b*c))#opposite point 2

    angles = np.array([a1, a2, a3])
    sorted_inds = np.argsort(angles)
    sorted_angles = angles[sorted_inds]
    sorted_pos = [xs[sorted_inds], ys[sorted_inds]]

    return sorted_angles, sorted_pos


def get_all_tris(ids):
    perms = list(itertools.combinations(ids, 3))
    return perms



def match_angles(angles1, angles2):
    # get first two angles of each list
    #  angles are sorted, so this is smallest 2 angles
    angles1 = np.array(angles1)[:,:2]
    angles2 = np.array(angles2)[:,:2]

#    tree_angles1 = KDTree(angles1)
    tree_angles2 = KDTree(angles2)

    dists, inds = tree_angles2.query(angles1, k=1)

    return dists, inds



def get_all_angles(xs, ys, N=4):
    allperms = set()
    for ii in range(len(xs)):
        dists = (xs-xs[ii])**2 + (ys-ys[ii])**2
        closest = np.argsort(dists).tolist()[:N+1]

        perms = [tuple(sorted(p)) for p in get_all_tris(closest)]
        for p in perms:
            allperms.add(p)

    allperms = list(allperms)
    angles = []
    pos = []
    for p in allperms:
        sorted_angles, sorted_pos = get_angles(xs[list(p)], ys[list(p)])
        angles.append(sorted_angles)
        pos.append(sorted_pos)

    return angles, pos



def get_frame_transformation_matrix(reference_image, other_image):
    ref_xs, ref_ys = get_star_locs(reference_image)
    other_xs, other_ys = get_star_locs(other_image)

    starttime = time.time()
    ref_angles, ref_pos = get_all_angles(ref_xs, ref_ys)
    other_angles, other_pos = get_all_angles(other_xs, other_ys)

    dists, inds = match_angles(ref_angles, other_angles)
    weights = np.pow(np.clip(1-dists*10, 0, 1), 4)

    all_refs = []
    all_targs = []


    for ii, ind in enumerate(inds):
        w = weights[ii][0]
        ind = ind[0]
        p = ref_pos[ii]
        ps = other_pos[ind]

        ref_point  = [p[0][0] , p[1][0]]
        targ_point = [ps[0][0], ps[1][0]]

        if ref_point not in all_refs:
            all_refs.append(ref_point)
            all_targs.append(targ_point)

    H, inpts = cv.estimateAffinePartial2D(np.array(all_targs),
                                        np.array(all_refs),
                                       ransacReprojThreshold=5.0)


    return H



