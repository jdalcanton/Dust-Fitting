## to deal with display when running as background jobs
from matplotlib import use
use('Agg')
import pyfits
from numpy import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pylab
import PIL
import time
import ezfig  # morgan's plotting code
import read_brick_data as rbd
import isolatelowAV as iAV
import analysis as analysis
from scipy import ndimage
from scipy.stats import norm
from scipy.ndimage import filters as filt
import scipy.special as special
import string
import os.path as op
import random as rnd
import numpy as np
from astropy.io import fits
from astropy import wcs

# Principle: "rotate" to frame where all "colors" are along reddening
# vector. Rotate the -data- into this frame, once. Rotate the original
# unreddened CMD -once-.  Then all convolutions are fast 1-d.


############# code for generating foreground CMD ###############

def split_ra_dec(ra, dec, d_arcsec=10.0, ra_bins='', dec_bins=''): 
    """
    indices, ra_bins, dec_bins = split_ra_dec(ra, dec, d_arcsec=10.0, ra_bins=0, dec_bins=0)

    Takes lists of ra and dec, divides the list into pixels of width
    d_arcsec, and returns an array, where each entry is a list of the
    indices from the original ra, dec list whose ra and dec fell into
    that pixel. Also returns the boundaries of the bins used to
    generate the index array. (Functions like reverse indices in IDL's
    histogram function)

    OR

    if ra_bins and dec_bins are provided as arguments, uses those, overriding d_arcsec

    To setup: 
    filename = '../../Data/12056_M31-B15-F09-IR_F110W_F160W.st.fits'
    fitstab = pyfits.open(filename)
    ra  = fitstab[1].data.field('RA') 
    dec = fitstab[1].data.field('DEC')
    fitstab.close()
    """

    if (len(dec_bins) == 0):
        
        print 'Generating new bins for declination '

        # figure out ranges
        dec_min = nanmin(dec)
        dec_max = nanmax(dec)

        # find center
        dec_cen = (dec_max + dec_min) / 2.0
        print 'dec_cen: ', dec_cen

        # figure out number of bins, rounding up to nearest integer
        d_dec =  d_arcsec / 3600.0
        print 'd_dec: ',d_dec
        n_dec = int((dec_max - dec_min) / d_dec + 1.0)

        # calculate histogram bin edges
        dec_bins = linspace(dec_min,dec_min + n_dec*d_dec, num=n_dec+1) 
        print 'Dec Range: ', dec_bins.min(), dec_bins.max()

    else:

        print 'Using user-provided declination bins: ',[dec_bins.min(), dec_bins.max()]

    # recalculate maximums and center, since range is not necessarily 
    # an integer multiple of d_arcsec
    n_dec = len(dec_bins) - 1
    dec_min = dec_bins.min()
    dec_max = dec_bins.max()
    dec_cen = (dec_max + dec_min) / 2.0

    if (len(ra_bins) == 0):

        print 'Generating new bins for RA '

        # figure out ranges
        ra_min =  nanmin(ra)
        ra_max =  nanmax(ra)

        # figure out number of bins, rounding up to nearest integer
        d_ra  = (d_arcsec / cos(math.pi * dec_cen / 180.0)) / 3600.0
        print 'd_ra:  ',d_ra
        n_ra  = int((ra_max - ra_min)   / d_ra  + 1.0)

        # calculate histogram bin edges
        ra_bins  = linspace(ra_min, ra_min  + n_ra*d_ra,   num=n_ra+1)
        print 'RA Range:  ',ra_bins.min(), ra_bins.max()
    
    else:

        print 'Using user-provided RA bins: ',[ra_bins.min(), ra_bins.max()]

    # recalculate maximums and center, since range is not necessarily 
    # an integer multiple of d_arcsec
    n_ra = len(ra_bins) - 1
    ra_min  = ra_bins.min()
    ra_max  = ra_bins.max()
    ra_cen = (ra_max + ra_min) / 2.0

    # get indices falling into each pixel in ra, dec bins

    raindex  = digitize(ra,  ra_bins) - 1
    decindex = digitize(dec, dec_bins) - 1

    # generate n_ra by n_dec array of lists, giving indices in each bin

    indices = empty( (n_ra,n_dec), dtype=object )
    for (i,j), value in ndenumerate(indices):
        indices[i,j]=[]

    # populate the array with the indices in each bin
    for k in range(len(ra)):
        indices[raindex[k],decindex[k]].append(k)
    
    print 'Generated indices in array of dimensions', indices.shape
    print 'Median number of stars per bin: ', median([len(indices[x,y]) for 
                 (x,y),v in ndenumerate(indices) if (len(indices[x,y]) > 1)])

    return indices, ra_bins, dec_bins


def median_rgb_color_map(indices,c,m,mrange=[18.8,22.5],crange=[0.3,2.5],
                         rgbparam=[22.0, 0.74, -0.13, -0.012]):
    """
    Return array of median NIR color of RGB in range, and standard
    deviation, relative to fiducial curve, and list of the good pixel
    values.
    
    indices = array of indices in RA, Dec bins (from split_ra_dec()) 
    c, m = list of colors,magnitudes referred to by indices in i 
    mrange = restrict to range of magnitudes
    rgbparam = 2nd order polynomial of RGB location [magref, a0, a1, a2]

    cmap, cmapgoodvals, cstdmap, cstdmapgoodvals, cmean, cmeanlist = 
                  median_rgb_color_map(indices,c,m,mrange=[19,21.5])

    to display:
    plt.imshow(cmap,interpolation='nearest')
    plt.hist(cmapgoodvals, cumulative=True,histtype='step',normed=1)
    """

    # fiducial approximation of isochrone to flatten RGB to constant color

    magref = rgbparam[0]
    a = rgbparam[1:4]
    cref = a[0] + a[1]*(m-magref) + a[2]*(m-magref)**2
    dc = c - cref

    # initialize color map

    emptyval = -1
    cmap    = empty( indices.shape, dtype=float )
    cmeanmap    = empty( indices.shape, dtype=float )
    cstdmap = empty( indices.shape, dtype=float )
    nstarmap = empty( indices.shape, dtype=float )

    # calculate median number of stars per bin, to set threshold for
    # ignoring partially filled ra-dec pixels.

    n_per_bin = median([len(indices[x,y]) for (x,y),v in ndenumerate(indices) 
                        if (len(indices[x,y]) > 1)])
    n_thresh = n_per_bin - 7*sqrt(n_per_bin)
    
    print 'Dimension of Color Map: ',cmap.shape

    for (i,j), value in ndenumerate(indices):

        if (len(indices[i,j]) > n_thresh):

            dctmp = dc[indices[i,j]]
            ctmp  = c[indices[i,j]]
            mtmp  = m[indices[i,j]]
            cmap[i,j] = median(dctmp.compress(((mtmp > mrange[0]) & 
                                               (mtmp<mrange[1]) & 
                                               (ctmp > crange[0]) & 
                                               (ctmp<crange[1])).flat))
            cmeanmap[i,j] = mean(dctmp.compress(((mtmp > mrange[0]) & 
                                               (mtmp<mrange[1]) & 
                                               (ctmp > crange[0]) & 
                                               (ctmp<crange[1])).flat))
            cstdmap[i,j] = std(dctmp.compress(((mtmp > mrange[0]) & 
                                               (mtmp<mrange[1]) & 
                                               (ctmp > crange[0]) & 
                                               (ctmp<crange[1])).flat))
            nstarmap[i,j] = len(dctmp.compress(((mtmp > mrange[0]) & 
                                               (mtmp<mrange[1]) & 
                                               (ctmp > crange[0]) & 
                                               (ctmp<crange[1])).flat))
        else:

            cmap[i,j] = emptyval
            cmeanmap[i,j] = emptyval
            cstdmap[i,j] = emptyval
            nstarmap[i,j] = emptyval

    # calculate list of good values

    igood = where(cmap > -1)
    print 'Median Color offset:', median(cmap[igood])

    return cmap, cmap[igood], cstdmap, cstdmap[igood], cmeanmap, cmeanmap[igood], nstarmap, nstarmap[igood]


def isolate_low_AV_color_mag(filename = '../../Data/12056_M31-B15-F09-IR_F110W_F160W.st.fits',
                             frac=0.2, mrange=[19,21.5],
                             rgbparam=[22.0, 0.74, -0.13, -0.012],
                             d_arcsec=10):
    """
    Return a list of color and magnitude for stars in the frac of low
    extinction pixels defined by either blueness or narrowness of RGB.

    cblue,mblue,iblue,cnarrow,mnarrow,inarrow,cmap,cstdmap = 
              isolate_low_AV_color_mag(filename, fraction, mrange, rgbparam)

    filename = FITS file of stellar parameters
    fraction = return stars that fall in the fraction of pixels with bluest RGB
    mrange = F160W magnitude range for evaluating color of RGB
    rgbparam = polynomial approximation of color of RGB = [magref,a0,a1,a2]

    cblue,mblue,iblue = color, magnitude, and indices of stars in the
                        bluest bins
    cnarrow,mnarrow,inarrow = same, but for narrowest RGB bins
    cmap = 2d array of RGB color
    cstd = 2d array of RGB width

    """

    m1, m2, ra, dec = rbd.read_mag_position_gst(filename)
    c = array(m1 - m2)
    m = array(m2)

    # cut out main sequence stars

    crange = [0.3, 2.0]

    indices, rabins, decbins = split_ra_dec(ra, dec, d_arcsec)

    cm,cmlist,cstd,cstdlist,cmean,cmeanlist = median_rgb_color_map(indices, c, m,
                                                   mrange, crange, rgbparam)
    print 'Number of valid areas in color map: ', len(cmlist)

    # find bluest elements

    cmlist.sort()
    n_cm_thresh = int(frac * (len(cmlist) + 1))
    cmthresh = cmlist[n_cm_thresh]
    cmnodataval = -1
    
    # find elements with the narrowest color sequence

    cstdlist.sort()
    n_cstd_thresh = int(frac * (len(cstdlist) + 1))
    cstdthresh = cstdlist[n_cstd_thresh]
    cstdnodataval = -1
    
    ikeep_blue = []

    for (x,y), value in ndenumerate(cm): 
        if ((cm[x,y] < cmthresh) & (cm[x,y] > cmnodataval)) :
            ikeep_blue.extend(indices[x,y])

    ikeep_narrow = []

    for (x,y), value in ndenumerate(cm): 
        if ((cstd[x,y] < cstdthresh) & (cm[x,y] > cmnodataval)) :
            ikeep_narrow.extend(indices[x,y])

    print 'Blue: Returning ', len(ikeep_blue),' out of ', len(c),' stars from ',n_cm_thresh,' bins.'

    print 'Narrow: Returning ', len(ikeep_narrow),' out of ',len(c),' stars from ',n_cstd_thresh,' bins.'

    return c[ikeep_blue], m[ikeep_blue], ikeep_blue, c[ikeep_narrow], m[ikeep_narrow], ikeep_narrow, cm, cstd


def color_sigma_vs_mag(c, m, magrange=[18.5,24.0], nperbin=50):
    """
    Return color dispersion vs magnitude, calculated in bins with
    constant numbers of stars, limited to magrange
    
    cdisp, mbins_disp = color_sigma_vs_mag(colorlist,maglist,magrange,nperbin)
    """

    # sort into ascending magnitude

    isort = argsort(m)
    
    # take groups of nperbin, and calculate the interquartile range in
    # color, and the median magnitude of the group.

    nout = int(len(c)/nperbin)
    mmed = empty(nout,dtype=float)
    cvar = empty(nout,dtype=float)

    for i in range(nout):
        k = i * nperbin + arange(nperbin)
        mmed[i] = median(m[isort[k]])
        ctmp = array(c[isort[k]])  # make numpy array
        cmed = median(ctmp)
        c25 = median(ctmp.compress(ctmp <= cmed))
        c75 = median(ctmp.compress(ctmp > cmed))
        cvar[i] = (c75 - c25) / (2.0 * 0.6745)
    
    return cvar, mmed


def clean_fg_cmd(fg_cmd, nsigvec, niter=4, showplot=0):
    """
    Line by line, fit for mean color and width of RGB, then mask 
    outside of nsig[0] on blue and nsig[1] on red.
    Fit iteratively, using rejection.
    
    Return mask (same dimensions as fg_cmd), and vector of mean 
    color and width of RGB
    """

    mask = 1.0 + zeros(fg_cmd.shape)

    # Record size of array, in preparation for looping through magnitude bins

    nmag = fg_cmd.shape[0]
    ncol = fg_cmd.shape[1]

    nsig_clip = 2.5
    nanfix = 0.0000000000001
    
    for j in range(niter):

        # Get mean color and width of each line

        meancol = array([sum(arange(ncol) * fg_cmd[i,:] * mask[i,:]) / 
                  sum(fg_cmd[i,:] * (mask[i,:] + nanfix)) for i in range(nmag)])
        sigcol = array([sqrt(sum(fg_cmd[i,:]*mask[i,:] *
                                 (range(ncol) - meancol[i])**2) / 
                             sum(fg_cmd[i,:]*mask[i,:] + nanfix)) 
                        for i in range(nmag)])

        if showplot != 0:
            plt.plot(meancol,range(nmag),color='blue')
            plt.plot(meancol-nsigvec[0]*sigcol,range(nmag),color='red')
            plt.plot(meancol+nsigvec[1]*sigcol,range(nmag),color='red')

        # Mask the region near the RGB

        mask = array([where(abs(range(ncol) - meancol[i]) < 
                            nsig_clip*sigcol[i], 1.0, 0.0) 
                for i in range(nmag)])

    # Regenerate mask using requested sigma clip

    mask = array([where((range(ncol) - meancol[i] > -nsigvec[0] * sigcol[i]) &
                        (range(ncol) - meancol[i] <  nsigvec[1] * sigcol[i]), 
                        1.0, 0.0) 
                  for i in range(nmag)])
    
    return mask, meancol, sigcol


def display_CM_diagram(c, m, crange=[-1,3], mrange=[26,16], nbins=[50,50],
                       alpha=1.0):
    """
    Plot binned CMD, and return the binned histogram, extent vector, 
    cbins, mbins
    """
        
    h, xedges, yedges = histogram2d(m, c, range=[sort(mrange), crange], 
                                    bins=nbins)
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]

    plt.imshow(log(h),  extent=extent, origin='upper', aspect='auto', 
               interpolation='nearest',alpha=alpha)
    plt.xlabel('F110W - F160W')
    plt.ylabel('F160W')

    return h, extent, yedges, xedges


########## CODE TO DEAL WITH INPUT DATA #################

def get_star_indices(c_star, m_star, cedges, medges):
    """
    Input: list of star colors and magnitudes, and vectors of the 
    edges returned by 2d histogram of the comparison CMD

    Return: i_color and i_magnitude -- indices giving location within
    the comparison CMD
    """

    # calculate indices in image of star locations

    dcol = cedges[1] - cedges[0]
    dmag = medges[1] - medges[0]
    i_c = floor((array(c_star) - cedges[0])/dcol)
    i_m = floor((array(m_star) - medges[0])/dmag)
    
    return i_c.astype(int), i_m.astype(int)


def make_data_mask(fg_cmd, cedges, medges, m1range, m2range, clim, useq=0):
    """
    fg_cmd = image to make mask for
    cedges, medges = values of color and magnitude along array sides
    m1lims = 2 element vector for blue filter [bright,faint]  
    m2lims = 2 element vector for red filter [bright,faint]   
    useq = if set, assume fg_cmd is rotated to reddening free mags, 
           and translate mag limits accordingly. Assume value = [c0=reference_color]
    clim = color to cut on blue side

    Returns mask of 1's where data falls within magnitude limits
    """
    mask = 1.0 + zeros(fg_cmd.shape)

    # Define reddening parameters

    #print 'Defining reddening parameters'
    Amag_AV = 0.20443
    Acol_AV = 0.33669 - 0.20443
    #t = np.arctan(-Amag_AV / Acol_AV)

    if useq != 0:
        c0 = useq[0]
        #t = useq[1]
        mlim2_faint  = m2range[1] + (cedges[:-1] - c0)*(-Amag_AV / Acol_AV)
        mlim2_bright = m2range[0] + (cedges[:-1] - c0)*(-Amag_AV / Acol_AV)
        mlim1_faint  = ((m1range[1] - cedges[:-1]) + 
                        (cedges[:-1] - c0)*(-Amag_AV / Acol_AV))
        mlim1_bright = ((m1range[0] - cedges[:-1]) + 
                        (cedges[:-1] - c0)*(-Amag_AV / Acol_AV))
    else:
        mlim2_faint  = m2range[1] + 0.0*cedges[:-1]
        mlim2_bright = m2range[0] + 0.0*cedges[:-1]
        mlim1_faint  = (m1range[1] - cedges[:-1])
        mlim1_bright = (m1range[0] - cedges[:-1])

    nmag = fg_cmd.shape[0]
    ncol = fg_cmd.shape[1]

    dm = medges[1] - medges[0]
    mask = array([where((cedges[:-1] > clim[i]) & 
                        (medges[i] > mlim2_bright) & (medges[i] < mlim2_faint) & 
                        (medges[i] > mlim1_bright) & (medges[i] < mlim1_faint), 
                        #(medges[i] > mlim2_bright) & (mlim2_faint- medges[i] > -dm) & 
                        #(medges[i] > mlim1_bright) & (mlim1_faint - medges[i] > -dm), 
                        1.0, 0.0) for i in range(nmag)])

    return mask

############# CODE TO GENERATE MODEL CMD #######################

def makefakecmd(fg_cmd, cvec, mvec, AVparam, floorfrac=0.0, 
                mask = 1.0, SHOWPLOT=True, 
                noise_model=0.0, noise_frac=0.0, frac_red_mean=0.5, print_fred=False):
    """
    fg_cmd, cvec, mvec = 2-d binned CMD of lowreddening stars, binned
            in color c & DEREDDENED magnitude m' =
            m+(c-c0)*[A_1/(A_1-A_2)], and the bin coordinates. c0 is
            arbitrary color coordinate around which the rotation such
            that reddening is purely horizontal occured.  Inputs
            calculated as fg_cmd, extent, cvec, mvec =
            display_CM_diagram(c, m') with c, m derived from cnar,
            mnar outputs of isolate_low_AV_color_mag, and cleaned of
            points with unreasonable colors, and m' =
            m+(c-c0)*[A_1/(A_1-A_2)].  

            (write second function to generate a mask of where to compare data)
    """
    
    # vectors defining mapping from array location to color and magnitude

    crange = [cvec[0],cvec[-1]]
    mrange = [mvec[0],mvec[-1]]
    
    dcol = cvec[1] - cvec[0]
    dmag = mvec[1] - mvec[0]
    
    # set up reddening parameters
    
    #fracred = AVparam[0]
    alpha = log(0.5) / log(frac_red_mean)
    fracred = (exp(AVparam[0]) / (1. + exp(AVparam[0])))**(1./alpha) # x = ln(f/(1-f))
    if (print_fred):
        print 'In makefakecmd: f_red = ', fracred
    medianAV = AVparam[1]
    #stddev = AVparam[2] * medianAV
    #sigma_squared = log((1. + sqrt(1. + 4. * (stddev/medianAV)**2)) / 2.)
    #sigma = sqrt(sigma_squared)
    sigma = AVparam[2]
    stddev_squared = (exp(sigma**2)-1.0)*exp(sigma**2)*(medianAV**2)
    stddev = sqrt(stddev_squared)
    Amag_AV = AVparam[3]
    Acol_AV = AVparam[4]
    
    # translate color shifts into equivalent AV
    
    dAV = dcol / Acol_AV
    
    # make log-normal convolution kernel in color, based on lg-normal
    # in A_V (Note, currently does not integrate lg-normal over bin
    # width. Could make the kernel with color spacing of dcol/4, and
    # then sum every 4 bins back down to normal dcol width to
    # approximate integration.)
    
    nstddev = 5.0                # set range based on width of log normal
    nresamp = 4.0

    #kernmaxcolor = Acol_AV * exp(nstddev * sigma + log(medianAV))
    kernmaxcolor = Acol_AV * (medianAV + nstddev * stddev)
    nkern = int(kernmaxcolor / dcol)
    # make sure kernel size is <= size of array and > 0
    nkern = minimum(len(cvec), nkern) 
    nkern = maximum(1, nkern)
    nkernresamp = nkern * nresamp

    kernmaxcolor = nkern * dcol  # recalculate maximum 
    #colorkern = arange(int(nkern), dtype=float) * dcol
    #AVkern = colorkern / Acol_AV
    resampcolorkern = arange(int(nkernresamp), dtype=float) * (dcol/nresamp)
    AVkern = resampcolorkern / Acol_AV
    AVkern[0] = 1.0e-17
    pAVkernresamp = ((1.0 / (AVkern * sigma * sqrt(2.0 * 3.1415926))) * 
               exp(-0.5 * (log(AVkern / medianAV) / sigma)**2))

    # undo resampling by adding up bins
    pAVkern = array([(pAVkernresamp[i*nresamp:(i+1)*nresamp-1]).sum() / nresamp 
                     for i in range(nkern)])

    # normalize the kernels

    pAVkernnorm = pAVkern.sum()
    pAVkern = pAVkern / pAVkernnorm
    
    # zero out where not finite
    
    i = where (isfinite(pAVkern) == False)

    if len(i) > 0:
        pAVkern[i]  = 0.0
    
    # generate combined foreground + reddened CMD
    
    cmdorig = fg_cmd.copy()
    
    # sequentially shift cmd and co-add, weighted by pEBV
    
    dmagshift = dAV * Amag_AV / dmag
    dcolshift = dAV * Acol_AV / dcol

    cmdred = pAVkern[0] * fg_cmd.copy()

    for i in range(1, len(pAVkern)):

        cmdred[:,i:] += cmdorig[:,0:-i] * pAVkern[i] 

    # combination of unreddened and unreddend cmd

    cmdcombo = (1.0-fracred)*fg_cmd + fracred*cmdred

    # add in noise model 

    cmdcombo = (1.0 - noise_frac)*cmdcombo + noise_frac*noise_model

    # display commands

    if SHOWPLOT:

        plt.imshow(cmdcombo,extent=[crange[0],crange[1],mrange[1],mrange[0]], 
                   origin='upper',aspect='auto', interpolation='nearest')
        plt.xlabel('F110W - F160W')
        plt.ylabel('Extinction Corrected F160W')

    # mask regions (defaults to *1 if mask not set in function call)
    
    cmdcombo = cmdcombo * mask
                              
    # renormalize the entire PDF
    
    norm = cmdcombo.sum()
    cmdcombo = cmdcombo / norm
    
    # permit a constant offset in unmasked regions, to give non-zero probability
    # of a random star.

    if floorfrac > 0:

       if len(array(mask).shape) == 0:
           masknorm = len(cmdcombo)

       else:
           masknorm = mask.sum()

       cmdcombo = mask*((1.0 - floorfrac) * cmdcombo + 
                        floorfrac * (mask / masknorm))
    
    # return cmd

    return cmdcombo


def makefakecmd_AVparamsonly(param):
    """
    Same as makefakecmd, but uses global parameters for most input
    """

    return makefakecmd(foreground_cmd, color_boundary, qmag_boundary,
                       [param[0], param[1], param[2], 0.20443, 
                        (0.33669 - 0.20443)], 
                       floorfrac=floorfrac_value, 
                       mask=color_qmag_datamask, SHOWPLOT=False,
                       noise_model = noise_model, 
                       frac_red_mean = frac_red_mean)

def cmdlikelihoodfunc(param, i_star_color, i_star_magnitude):
    """
    Return likelihood compared to dust model using param.
    Passes parameters for dust model, and the color and magnitude
    indices of stars (i.e., figure out which pixel of dusty CMD all 
    stars fall in before the call -- do the masking in advance as well)
    """

    if (ln_priors(param) == False):
        return -inf

    # calculate probability distribution

    img = makefakecmd_AVparamsonly(param)
    
    # calculate log likelihood

    pval = img[i_star_magnitude, i_star_color]

    lnp =  (log(pval[where(pval > 0)])).sum()

    return lnp


################################################
# global parameters setting up foreground CMD

datadir = '/astro/store/angst4/dstn/v8/'  # bagel
datadir = '/mnt/angst4/dstn/v8/'          # chex
datadir = '../../Data/'                   # poptart

resultsdir = '../Results/'

# Set up input data file and appropriate magnitude cuts 

fnroot = 'ir-sf-b21-v8-st'
m110range = [16.0,25.0]
m160range = [18.4,23.25] # just above RC -- worse constraints on AV
m160range = [18.4,24.5]  # just below RC
m160range = [18.4,24.0]  # middle of RC

fnroot = 'ir-sf-b17-v8-st'
m110range = [16.0,24.0]
m160range = [18.4,24.0]

fnroot = 'ir-sf-b14-v8-st'
m110range = [16.0,23.5]
m160range = [18.4,24.0]

fnroot = 'ir-sf-b16-v8-st'
m110range = [16.0,23.5]
m160range = [18.4,24.0]

fnroot = 'ir-sf-b18-v8-st'
m110range = [16.0,24.0]
m160range = [18.4,24.0]

fnroot = 'ir-sf-b19-v8-st'
m110range = [16.0,24.0]
m160range = [18.4,24.0]

fnroot = 'ir-sf-b22-v8-st'
m110range = [16.0,24.5]
m160range = [18.4,24.0]

fnroot = 'ir-sf-b12-v8-st'
m110range = [16.0,23.5]
m160range = [18.4,23.25]

fn = datadir + fnroot + '.fits'

mfitrange = [18.7,21.3]   # range for doing selection for "narrow" RGB

# Set up color range and binning for color and magnitude

#crange    = [0.3,3.0]
crange    = [0.3,2.5]
deltapix_approx =  [0.025,0.3]
nbins = [int((m160range[1] - m160range[0]) / deltapix_approx[1]),
         int((crange[1] - crange[0]) / deltapix_approx[0])]

# Define reddening parameters

Amag_AV = 0.20443
Acol_AV = 0.33669 - 0.20443
t = arctan(-Amag_AV / Acol_AV)
reference_color = 1.0

# define fraction of uniform "noise" to include in data model

floorfrac_value = 0.05


# generate foreground CMD in reddening free magnitudes

#clo, mlo, ilo, cnar, mnar, inr, cm, cstd = isolate_low_AV_color_mag(
#                    filename=fn, frac=0.025, mrange=mfitrange, d_arcsec=10.)

#qnar = mnar + (cnar-reference_color)*(-Amag_AV / Acol_AV)

#foreground_cmd_orig, qmag_boundary, color_boundary = histogram2d(qnar, cnar, 
#                                 range=[sort(m160range), crange], bins=nbins)

#foregroundmask, meancol, sigcol = clean_fg_cmd(foreground_cmd_orig, [3.0,3.5], 
#                                               niter=4, showplot=0)

#foreground_cmd = foreground_cmd_orig * foregroundmask

# make the noise model by masking out foreground and smoothing

#noisemask, meancol, sigcol = clean_fg_cmd(foreground_cmd_orig, [4.5,4.5], 
#                                          niter=5, showplot=0)
#noisemask = abs(noisemask - 1)
#noise_smooth = [3, 10]  # mag, color, in pixels
#noise_model_orig = foreground_cmd_orig * noisemask
#noise_model = ndimage.filters.uniform_filter(noise_model_orig,
#                                             size=noise_smooth)

# generate mask of data regions to ignore

#nsig_blue_color_cut = 2.0
#bluecolorlim = color_boundary[maximum(rint(meancol - nsig_blue_color_cut * 
#                                           sigcol).astype(int),0)]
#color_qmag_datamask = make_data_mask(foreground_cmd, color_boundary, 
#                                     qmag_boundary, m110range, m160range, 
#                                     bluecolorlim, useq=[reference_color])

# relative normalization of foreground model and noise model

#nfg = (foreground_cmd * color_qmag_datamask).sum()
#nnoise = (noise_model * color_qmag_datamask).sum()
#frac_noise = nnoise / (nnoise + nfg)
#print 'Noise fraction: ', frac_noise

# read in main data file

#m1, m2, ra, dec = rbd.read_mag_position_gst(fn)
#m = array(m2)
#c = array(m1 - m2)
#q = m + (c-reference_color)*(-Amag_AV / Acol_AV)
#ra = array(ra)
#dec = array(dec)

# exclude data outside the color-magnitude range

#igood = where((c > color_boundary[0]) & (c < color_boundary[-1]) &
#              (q > qmag_boundary[0]) & (q < qmag_boundary[-1]))
#m = m[igood]
#c = c[igood]
#q = q[igood]
#ra = ra[igood]
#dec = dec[igood]

# exclude data outside the data mask

#i_c, i_q = get_star_indices(c, q, color_boundary, qmag_boundary)
#igood = where(color_qmag_datamask[i_q, i_c] != 0)
#m = m[igood]
#c = c[igood]
#q = q[igood]
#ra = ra[igood]
#dec = dec[igood]
#i_c = i_c[igood]
#i_q = i_q[igood]

# split into RA-Dec

#binarcsec = 10.0
#i_ra_dec_vals, ra_bins, dec_bins = split_ra_dec(ra, dec, d_arcsec = binarcsec)

# grab data for a test bin, if needed for testing.

#i_test_ra = 60
#i_test_dec = 20
#ctest = c[i_ra_dec_vals[i_test_ra,i_test_dec]]
#mtest = m[i_ra_dec_vals[i_test_ra,i_test_dec]]
#qtest = q[i_ra_dec_vals[i_test_ra,i_test_dec]]
#i_ctest, i_qtest = get_star_indices(ctest, qtest, color_boundary, qmag_boundary)


########### CODE FOR LOOPING THROUGH RA-DEC BINS, RUNNING EMCEE ########

def generate_global_ra_dec_grid(d_arcsec=7.5):

    filename = 'radius_range_of_bricks.npz'  # generated by get_radius_range_of_all_bricks in isolatelowAV.py

    dat = load(filename)
    r_range_brick_array = dat['r_range_brick_array']
    ra_min  = min(r_range_brick_array[:,3])
    ra_max  = max(r_range_brick_array[:,4])
    dec_min = min(r_range_brick_array[:,5])
    dec_max = max(r_range_brick_array[:,6])

    # default M31 parameters (see also compleness.py)
    m31ra  = 10.6847929
    m31dec = 41.2690650    # use as tangent point

    # figure out number of bins, rounding up to nearest integer
    d_ra  = (d_arcsec / cos(math.pi * m31dec / 180.0)) / 3600.0
    d_dec =  d_arcsec / 3600.0
    print 'd_ra:  ',d_ra
    print 'd_dec: ',d_dec
    n_ra  = int((ra_max - ra_min)   / d_ra  + 1.0)
    n_dec = int((dec_max - dec_min) / d_dec + 1.0)

    # calculate histogram bin edges
    ra_bins  = linspace(ra_min,  ra_min  + n_ra*d_ra,   num=n_ra+1)
    dec_bins = linspace(dec_min, dec_min + n_dec*d_dec, num=n_dec+1) 
    print 'RA Range:  ', ra_bins.min(),  ra_bins.max()
    print 'Dec Range: ', dec_bins.min(), dec_bins.max()
    
    # recalculate maximums and center, since range is not necessarily 
    # an integer multiple of d_arcsec
    ra_max  = ra_bins.max()
    dec_max = dec_bins.max()
    ra_cen = (ra_max + ra_min) / 2.0
    dec_cen = (dec_max + dec_min) / 2.0

    return ra_bins, dec_bins

def extract_local_subgrid(ra_range, dec_range, 
                          ra_bins_global, dec_bins_global):
    """
    For a given range in ra and dec (rarange=[ramin, ramax], decrange
    = [decmin, decmax]), extract only the relevant subgrid of the
    global ra, dec grid defined by ra_bins_global and dec_bins_global
    (evenly spaced boundaries of global grid).

    Returns ra_bins_local, dec_bins_local and [iramin, iramax],
    [idecmin, idecmax] to show where resulting local array should be
    placed back in the global array:
        array_global[iramin:iramax, idecmin:idecmax] = array_local
    """
    dra = ra_bins_global[1] - ra_bins_global[0]
    ddec = dec_bins_global[1] - dec_bins_global[0]

    i_ra =  (array(ra_range)  - ra_bins_global[0])  / dra
    i_dec = (array(dec_range) - dec_bins_global[0]) / ddec

    i_ra[0]  = max((floor(i_ra[0])).astype(int), 0)  # prevent negatives
    i_dec[0] = max((floor(i_dec[0])).astype(int), 0)

    i_ra[1]  = (ceil(i_ra[1] + 1)).astype(int)   # add 1 for slicing to work
    i_dec[1] = (ceil(i_dec[1] + 1)).astype(int)  # add 1 for slicing to work

    print 'Extracting RA Global points between ',i_ra[0], ' and ', i_ra[1]
    print 'Extracting Dec Global points between ',i_dec[0], ' and ', i_dec[1]

    return ra_bins_global[i_ra[0]:i_ra[1]], dec_bins_global[i_dec[0]:i_dec[1]], i_ra, i_dec

    
def get_ra_dec_bin(i_ra=60, i_dec=20):
    
    i_ctest, i_qtest = get_star_indices(c[i_ra_dec_vals[i_ra,i_dec]], 
                                        q[i_ra_dec_vals[i_ra,i_dec]], 
                                        color_boundary, qmag_boundary)
    return i_ctest, i_qtest



class lnpriorobj(object):

    def __init__(self, f_mean):
        self.f_mean = f_mean

    def __call__(self, param, **kwargs):

        return ln_priors_function(param, f_mean = self.f_mean, **kwargs)

    def map_call(self, args):

        return self(*args)


class likelihoodobj(object):

    def __init__(self, foreground_cmd, noise_model, datamask, color_boundary, mag_boundary, 
                 noise_frac, floorfrac_value, f_mean):
        self.foreground_cmd = foreground_cmd
        self.noise_model = noise_model
        self.datamask = datamask
        self.color_boundary = color_boundary
        self.mag_boundary = mag_boundary
        self.noise_frac = noise_frac
        self.floorfrac_value = floorfrac_value
        self.f_mean = f_mean

    def __call__(self, param, i_star_color, i_star_magnitude):

        ln_priors = lnpriorobj(self.f_mean)
        lnp = ln_priors(param)

        if (lnp == -Inf):     # parameters out of range
            return -Inf

        # calculate probability distribution

        img = makefakecmd(self.foreground_cmd, self.color_boundary, self.mag_boundary,
                          [param[0], param[1], param[2], 0.20443, 
                           (0.33669 - 0.20443)], 
                          floorfrac=self.floorfrac_value, 
                          mask=self.datamask, SHOWPLOT=False,
                          noise_model = self.noise_model,
                          noise_frac = self.noise_frac,
                          frac_red_mean = self.f_mean)
    
        # calculate log likelihood

        pval = img[i_star_magnitude, i_star_color]

        lnlikelihood = (log(pval[where(pval > 0)])).sum()

        return lnp + lnlikelihood

    def map_call(self, args):

        return self(*args)

def test_run_one_brick(datadir='../../Data/', results_extension = '', AV_fitsfile=''):
    """
    Test code using single pixel
    """

    #datadir = '../../Data/'
    #results_extension = ''
    nwalkers = 50
    nsamp = 15
    nburn = 150
    #nburn = 300

    fileroot = 'ir-sf-b15-v8-st'
    d_arcsec = 6.64515
    grab_ira_idec = [7, 59]    # high extinction, high f_red
    grab_ira_idec = [45, 63]    # high extinction, moderate f_red
    grab_ira_idec = [45, 60]    # high extinction, moderate f_red
    grab_ira_idec = [49, 56]    # high extinction, moderate f_red
    grab_ira_idec = [16, 18]    # high extinction, loweish f_red
    grab_ira_idec = [59, 35]    # lowish extinction, loweish f_red
    grab_ira_idec = [61, 42]    # lowish extinction, loweish f_red
    grab_ira_idec = [61, 59]    # very high extinction, loweish f_red
    grab_ira_idec = [95, 12]    # low extinction, loweish f_red
    grab_ira_idec = [95, 13]    # low extinction, loweish f_red

    fileroot = 'ir-sf-b16-v8-st'
    d_arcsec = 6.64515
    grab_ira_idec = [90, 28]    # low extinction, low f_red
    grab_ira_idec = [90, 29]    # low extinction, low f_red

    fileroot = 'ir-sf-b17-v8-st'
    d_arcsec = 6.64515
    grab_ira_idec = [10, 76]    # moderate extinction, high f_red, low N
    grab_ira_idec = [12, 62]    # moderate extinction, high f_red
    grab_ira_idec = [11, 76]    # moderate extinction, high f_red
    grab_ira_idec = [13, 74]    # moderate extinction, high f_red
    grab_ira_idec = [29, 67]    # moderate extinction, high f_red
    grab_ira_idec = [45, 71]    # moderate extinction, high f_red
    grab_ira_idec = [45, 70]    # moderate extinction, high f_red

    fileroot = 'ir-sf-b16-v8-st'
    d_arcsec = 6.64515
    grab_ira_idec = [103, 52]    # moderate extinction, high f_red, low N
    grab_ira_idec = [104, 1]    # moderate extinction, high f_red, low N
    grab_ira_idec = [104, 24]    # moderate extinction, high f_red, low N
    grab_ira_idec = [110, 19]    # big outlier
    grab_ira_idec = [110, 42]    # modest outlier
    grab_ira_idec = [114, 44]    # modest outlier

    fileroot = 'ir-sf-b02-v8-st'
    d_arcsec = 25.0
    grab_ira_idec = [25, 0]    # high sigma in low res
    grab_ira_idec = [28, 4]    # high sigma in low res

    fileroot = 'ir-sf-b12-v8-st'
    d_arcsec = 25.0
    grab_ira_idec = [30, 15]    # high sigma in low res

    fileroot = 'ir-sf-b06-v8-st'
    d_arcsec = 25.0
    grab_ira_idec = [5, 12]    # high sigma in low res
    grab_ira_idec = [17, 18]    # high sigma in low res
    grab_ira_idec = [9, 17]    # high sigma in low res

    d_derived, percentile_derived, samp, d, bestfit, sigmavec, i_c, i_q, fg_cmd, fake_cmd \
        = run_one_brick(fileroot, datadir=datadir, results_extension=results_extension, 
                        deltapixorig = [0.025,0.2], d_arcsec = d_arcsec, d_pix_offset=[0,0],
                        showplot='', nwalkers=nwalkers, nsamp=nsamp, nburn=nburn, 
                        grab_ira_idec=grab_ira_idec, AV_fitsfile=AV_fitsfile)

    return

def run_one_brick(fileroot, datadir='../../Data/', results_extension='', 
                  deltapixorig = [0.025,0.2], d_arcsec = 6.64515, d_pix_offset=[0,0],
                  showplot='', nwalkers=50, nsamp=15, nburn=150, grab_ira_idec=[],
                  AV_fitsfile=''):
    """
    For a given fits file, do all the necessary prep for running fits
    - read file
    - extract appropriate ra-dec subgrid
    - calculate major axis length radius for each pixel
    - Load unreddened foreground, noise, & mask
      (using same color, mag range as data)
    - trim data to appropriate color, mag range (global CMD range + datamask)
    - grid data in ra-dec
    - if grab_ira_idec = integer pair, save useful info for that pixel number (i_ra, i_dec)
    """

    # set up grid sizes, magnitude limits, etc
    crange    = [0.3,3.5]        # range of colors to use in CMD fitting
    #maglimoff = [0.0, 0.25]      # shift 50% magnitude limits this much brighter
    maglimoff = [0.0, 0.5]      # shift 50% magnitude limits this much brighter
    mfitrange = [18.7,21.3]      # range for doing selection for "narrow" RGB
    floorfrac_value = 0.05       # define fraction of uniform "noise" to include in data model
    dr = 0.025                   # radius interval within which to do analyses
    r_interval_range = [0.15, 1.8] # range over which to calculate foreground CMDs (clips bulge)
    #nrgbstars = 3000             # target number of stars on upper RGB
    nrgbstars = 2750             # target number of stars on upper RGB
    n_substeps = 6               # number of substeps before radial foreground CMDs are independent
    masksig = [2.5, 3.0]         # limits of data mask for clipping noise from foreground CMD
    noisemasksig = [4.5,3.5]     # limits of noise mask for clipping foreground CMD from noise
    n_fit_min = 15
    frac_red_mean = 0.2          # matters at low AV, where filling factor is small
    f_red_model_pa = 37.0
    f_red_model_incl = 78.0
    f_red_model_hz_over_hr = 0.15             # needed to calculate model of f_red
    xinitval = 0.05
    AVinitval = 0.35
    sigmainitval = 0.4
    param_init = [xinitval, AVinitval, sigmainitval]  # starting at x=0 locks in, because random*0 = 0
        
    ln_priors = lnpriorobj(frac_red_mean)
    prior_parameters = ln_priors(param_init, return_prior_parameters=True)

    # set up file names

    datafile = datadir + fileroot + '.fits'
    savefile = '../Results/' + fileroot + results_extension + '.npz'
    pngfileroot = '../Results/' + fileroot + results_extension
    completenessdir = '../../Completeness/'
    m110completenessfile = 'completeness_ra_dec.st.F110W.nstar.npz'
    m160completenessfile = 'completeness_ra_dec.st.F160W.nstar.npz'

    # Define reddening parameters

    Amag_AV = 0.20443
    Acol_AV = 0.33669 - 0.20443
    t = arctan(-Amag_AV / Acol_AV)
    reference_color = 1.0

    # Store processing parameters in dictionary

    allparamvals = locals()
    param_names = ['crange', 'maglimoff', 'deltapixorig', 'mfitrange', 'd_arcsec',  'd_pix_offset',
                   'floorfrac_value', 'dr', 'r_interval_range', 'nrgbstars', 'n_substeps', 
                   'masksig', 'noisemasksig', 'n_fit_min', 'frac_red_mean', 
                   'f_red_model_incl', 'f_red_model_pa', 'f_red_model_hz_over_hr',
                   'Amag_AV', 'Acol_AV', 'reference_color', 
                   'nwalkers', 'nsamp', 'nburn', 
                   'fileroot', 'datadir', 'datafile', 'savefile', 
                   'param_init', 'prior_parameters',
                   'completenessdir', 'm110completenessfile', 'm160completenessfile']
    processing_params = {k: allparamvals[k] for k in param_names}

    # read in data file

    m1, m2, ra, dec = rbd.read_mag_position_gst(datafile)
    m = array(m2)
    c = array(m1 - m2)
    ra = array(ra)
    dec = array(dec)
    ra_range = [min(ra), max(ra)]
    dec_range = [min(dec), max(dec)]

    # Set up global grid, and shift by fractional pixel if requested

    ra_global, dec_global = generate_global_ra_dec_grid(d_arcsec = d_arcsec)
    if (d_pix_offset != [0,0]):
        print 'Offsetting RA, Dec grid by fractional pixel: ',d_pix_offset
        dra  = ra_global[1]  - ra_global[0]
        ddec = dec_global[1] - dec_global[0]
        ra_global = ra_global + dra * d_pix_offset[0]
        dec_global = dec_global + ddec * d_pix_offset[1]

    # extract local subset of global grid

    ra_local,  dec_local, ira, idec = extract_local_subgrid(ra_range, dec_range, 
                                                            ra_global, dec_global)

    # Calculate radius and interpolated log10 of surface density at each ra-dec grid point

    ira, idec = indices([len(ra_local),len(dec_local)])
    ra_vec  = ra_local[ira]
    dec_vec = dec_local[idec]
    ra_vec_cen = array([(ra_vec[i] + ra_vec[i+1])/2. for i in range(len(ra_vec)-1)])
    dec_vec_cen = array([(dec_vec[i] + dec_vec[i+1])/2. for i in range(len(dec_vec)-1)])
    
    ira, idec = indices([len(ra_local)-1, len(dec_local)-1])
    ra_cen_array  = (ra_local[ira] + ra_local[ira+1]) / 2.
    dec_cen_array = (dec_local[idec] + dec_local[idec+1]) / 2.

    r_array = iAV.get_major_axis(ra_cen_array, dec_cen_array)
    r_range = [nanmin(r_array), nanmax(r_array)]
    print 'Radial Fitting Range: ', r_range

    lgnstar_array = np.log10(iAV.get_nstar_at_ra_dec(ra_cen_array, dec_cen_array))
    lgnstar_range = [nanmin(lgnstar_array), nanmax(lgnstar_array)]
    print 'Log10 Nstar density Fitting Range: ', lgnstar_range

    # Calculate expected values of f_red to set position-dependent prior properly.

    frac_red_array = analysis.get_model_frac_red(ra_cen_array, dec_cen_array,
                                              pa = f_red_model_pa,
                                              inclination = f_red_model_incl,
                                              hz_over_hr = f_red_model_hz_over_hr,
                                              make_plot=False)

    # Interpolate to get position-dependent initial guess for A_V, if file given

    if (AV_fitsfile != ''):
        print 'Initializing guesses for A_V from ', AV_fitsfile
        hdulist = fits.open(AV_fitsfile)
        AVfits = np.array(hdulist[0].data)
        hdr = hdulist[0].header
        hdulist.close()
        # set up WCS and get x, y of AV_fits image at locations of new grid
        w = wcs.WCS(hdr)
        i_fits_interp = np.array(w.wcs_world2pix(ra_cen_array, dec_cen_array, 0)).astype(int)
        i_y_fits_interp = i_fits_interp[0,:,:]
        i_x_fits_interp = i_fits_interp[1,:,:]
        # grab closest element of new grid, after clipping at image size
        AVinitmin = 0.05   # to handle if interpolates into zero or -666 region
        AV_init_array = np.maximum(AVfits[np.minimum(i_x_fits_interp, AVfits.shape[0] - 1),
                                          np.minimum(i_y_fits_interp, AVfits.shape[1] - 1)],
                                   AVinitmin)
        print 'Range of AV_init_array: ', np.min(AV_init_array), np.max(AV_init_array)

    else:

        print 'Using fixed initial guesses for A_V = ', param_init[1]
        AV_init_array = AVinitval + 0.0 * ra_cen_array
        

    # limit the number of radial foreground CMDs to return to save space
    dlgn_padding = 0.1
    lgnstar_range_limit = [lgnstar_range[0]-dlgn_padding, lgnstar_range[1]+dlgn_padding]
    print 'Expanded Log10 Nstar density Fitting Range: ', lgnstar_range_limit
    
    # Get range of magnitude limits to set CMD limits appropriately.
    # (i.e., tighten in to speed computation)

    m110dat = load(completenessdir + m110completenessfile)
    m110polyparam = m110dat['param']
    m160dat = load(completenessdir + m160completenessfile)
    m160polyparam = m160dat['param']
        
    p110 = poly1d(m110polyparam)
    p160 = poly1d(m160polyparam)

    maglim110_array = p110(lgnstar_array) - maglimoff[0]
    maglim160_array = p160(lgnstar_array) - maglimoff[1]

    m160brightlim = 18.5
    m110brightlim = m160brightlim + crange[0]
    # set up plotting range to be a tad generous.
    m110range = [m110brightlim, (nanmax(maglim110_array) + 
                                 (deltapixorig[0] + deltapixorig[1]))]
    m160range = [m160brightlim, nanmax(maglim160_array) + deltapixorig[1]]
    print 'F110W Range: ', m110range
    print 'F160W Range: ', m160range

    # exclude data outside the color-magnitude range

    igood = where((c >= crange[0]) & (c <= crange[1]) &
                  (m >= m160range[0]) & (m <= m160range[1]))
    m = m[igood]
    c = c[igood]
    ra = ra[igood]
    dec = dec[igood]

    # Calculate reddening-free magnitude

    q = m - (c-reference_color)*(Amag_AV / Acol_AV)

    # Initialize unreddened CMDs and noise models

    foreground_cmd_array, datamask_array, noise_array, noisefrac_vec, \
        meanlgnstar_vec, lgnstarrange_array, meancolor_array, sigmacolor_array, \
        n_per_cmd_vec, maglim_array, qmag_boundary, color_boundary = \
        iAV.make_nstar_selected_low_AV_cmds(nrgbstars = nrgbstars, nsubstep=n_substeps, 
                                    mrange = m160range,
                                    crange = crange,
                                    deltapixorig = deltapixorig,
                                    mnormalizerange = [19,21.5], 
                                    maglimoff = maglimoff,
                                    nsig_blue_color_cut = 2.0, 
                                    blue_color_cut_mask_only=False,
                                    usemask=True, masksig=masksig,
                                    makenoise=True, noisemasksig=noisemasksig,
                                    useq=True, reference_color=reference_color,
                                    restricted_n_range=lgnstar_range_limit)

    # set up boundaries of lgNstar density interval
    # i.e., for lgnstar_interval[i] < lgnstar < lgnstar_interval[i+1], use foreground_cmd_array[i].

    lgnstar_intervals = array([(meanlgnstar_vec[i] + meanlgnstar_vec[i+1])/2.0 
                               for i in range(len(meanlgnstar_vec)-1)])
    # append real outer limit (note: was maximum for radius, but is minimum for lgn!)
    max_lgnstarrange = nanmax(lgnstarrange_array)
    max_lgnstarrange = 0.75
    max_bricklgn = nanmax(lgnstar_range_limit)
    max_lgnstar = maximum(max_lgnstarrange, max_bricklgn)
    min_lgnstarrange = nanmin(lgnstarrange_array)
    #min_lgnstarrange = -10.
    min_bricklgn = nanmin(lgnstar_range_limit)
    min_lgnstar = minimum(min_lgnstarrange, min_bricklgn)
    print min_lgnstar, min_lgnstarrange, max_bricklgn, max_lgnstar
    lgnstar_intervals = append([min_lgnstar], lgnstar_intervals)
    lgnstar_intervals = append(lgnstar_intervals, [max_lgnstar])
    print 'Using global lgnstar_intervals: ', lgnstar_intervals

    # bin data into ra-dec

    i_ra_dec_vals, ra_bins_out, dec_bins_out = split_ra_dec(ra, dec, d_arcsec = d_arcsec,
                                                            ra_bins = ra_local, 
                                                            dec_bins=dec_local)

    # initialize sizes of output arrays.  -666 is default for no data
    nx, ny = i_ra_dec_vals.shape
    nz_bestfit = 3
    nz_sigma = nz_bestfit * 3
    nz_derived = 8
    nz_derived_sigma = nz_derived * 3
    nz_quality = 2
    nz_fred_prior = 3
    bestfit_values = zeros([nx, ny, nz_bestfit]) - 666.
    percentile_values = zeros([nx, ny, nz_sigma]) - 666.
    quality_values = zeros([nx, ny, nz_quality]) - 666.
    derived_values = zeros([nx, ny, nz_derived]) - 666.
    derived_percentile_values = zeros([nx, ny, nz_derived_sigma]) - 666.
    fred_prior_values = zeros([nx, ny, nz_fred_prior]) - 666.
    output_map = zeros([nx,ny]) - 666.


    # Loop through all possible lgnstar values
    # Try implementing with python map()

    for i_lgn in range(len(lgnstar_intervals) - 1):

        # only analyze a radial range if some of the lgnstar-interval is
        # covered by the brick
        if (((lgnstar_intervals[i_lgn]   >= lgnstar_range[0]) & 
             (lgnstar_intervals[i_lgn]   <  lgnstar_range[1])) | 
            ((lgnstar_intervals[i_lgn+1] >= lgnstar_range[0]) & 
             (lgnstar_intervals[i_lgn+1] <  lgnstar_range[1]))) :

            # get ra-dec pixels in the appropriate radial range
        
            i_ra, i_dec = where((lgnstar_array > lgnstar_intervals[i_lgn]) & 
                                (lgnstar_array <= lgnstar_intervals[i_lgn + 1]))
            
            print 'Fitting ',len(i_ra),' pixels in i_lgn=',i_lgn,' in lgn range ', \
                round(lgnstar_intervals[i_lgn],2), round(lgnstar_intervals[i_lgn+1],2), \
                ' Pix Fraction: ', round(float(len(i_ra)) / float(len(lgnstar_array)), 3)

            output_map[i_ra,i_dec] = meanlgnstar_vec[i_lgn]
            
            # Set up datamask, foreground CMD, and noise model for r-interval
    
            fg_cmd = foreground_cmd_array[:, :, i_lgn]
            datamask = datamask_array[:, :, i_lgn]
            noise_model = noise_array[:, :, i_lgn]
            noise_frac = noisefrac_vec[i_lgn]

            # normalize models

            fg_cmd = fg_cmd * datamask
            fg_cmd = fg_cmd / fg_cmd.sum()

            noise_model = noise_model * datamask
            noise_model = noise_model / noise_model.sum()

            #
            #plt.figure(2)
            #plt.clf()
            #max_fg = nanmax(fg_cmd)
            #plt.imshow(fg_cmd, interpolation='nearest', aspect='auto', vmin=0, vmax=max_fg)
            #plt.draw()

            #plt.figure(3)
            #plt.clf()
            #plt.imshow(datamask, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
            #plt.draw()

            #plt.figure(5)
            #plt.clf()
            #img = makefakecmd(fg_cmd, color_boundary, qmag_boundary,
            #              [-0.01, 1.5, 0.5, 0.20443, 
            #               (0.33669 - 0.20443)], 
            #              floorfrac=floorfrac_value, 
            #              mask=datamask, SHOWPLOT=False,
            #              noise_model = noise_model,
            #              noise_frac = noise_frac)
            #plt.imshow(img, interpolation='nearest', aspect='auto')
            #plt.draw()

            # get magnitude limits of interval
        
            maglim110 = maglim_array[0, i_lgn]
            maglim160 = maglim_array[1, i_lgn]

            # loop through pixels in the lgnstar density bin, fitting for parameters

            for i_pix in range(len(i_ra)):

                i_stars = i_ra_dec_vals[i_ra[i_pix], i_dec[i_pix]]
                nstar_1 = len(i_stars)
                nstar   = len(i_stars)
                frac_red_mean = frac_red_array[i_ra[i_pix], i_dec[i_pix]]

                trypix = True
                if nstar_1 > n_fit_min: 

                    i_c, i_q = get_star_indices(c[i_stars], q[i_stars], 
                                                color_boundary, qmag_boundary)
                    #plt.plot(i_c,i_q,',',color='white')

                    # cull points fainter than the magnitude limits, 
                    #     if there are sufficient stars

                    i_keep = where((i_q >= 0) & (i_q <= datamask.shape[0]-1) &
                                   (i_c >= 0) & (i_c <= datamask.shape[1]-1))
                    i_c = i_c[i_keep]
                    i_q = i_q[i_keep]

                    # cross check against datamask

                    i_keep = where(datamask[i_q, i_c] != 0)
                    i_c = i_c[i_keep]
                    i_q = i_q[i_keep]

                    nstar = len(i_c)
                    #print 'Pixel ', i_pix, ' has ', nstar_1,' stars before, ', nstar,' after'

                # If there are still enough stars, do fit.

                save_single_pix = False
                if (len(grab_ira_idec) == 2): 
                    save_single_pix = True
                    trypix = False
                    if ((i_ra[i_pix] == grab_ira_idec[0]) & (i_dec[i_pix] == grab_ira_idec[1])):
                        print 'Saving pixel number ', i_ra[i_pix], ' ', i_dec[i_pix]
                        trypix = True

                #if nstar > n_fit_min: 
                if ((nstar > n_fit_min) & trypix):

                    nsamp = 75

                    # set up function for likelihood fitting

                    lnp_func = likelihoodobj(fg_cmd, noise_model, datamask, 
                                             color_boundary, qmag_boundary, 
                                             noise_frac, floorfrac_value,
                                             frac_red_mean)

                    # set up function for priors
                    ln_priors = lnpriorobj(frac_red_mean)

                    ## run fit...
                    samp, d, bestfit, sigma, acor = run_emcee(i_c, i_q,
                                                              #param_init,
                                                              [param_init[0], 
                                                               AV_init_array[i_ra[i_pix], i_dec[i_pix]],
                                                               param_init[2]],
                                                              likelihoodfunction = lnp_func,
                                                              priorfunction = ln_priors, 
                                                              nwalkers=nwalkers, 
                                                              nsteps=nsamp, 
                                                              nburn=nburn)

                    # record results in output arrays
                    bestfit_values[i_ra[i_pix], i_dec[i_pix], :] = bestfit
                    sigmavec = sigma.flatten()
                    percentile_values[i_ra[i_pix], i_dec[i_pix], :] = sigmavec
                    idx = d['lnp'].argmax()
                    quality_values[i_ra[i_pix], i_dec[i_pix], :] = [(d['lnp'][idx])/nstar, nstar]
                    
                    # update prior parameters for best fit, and update into processing_params
                    prior_parameters = ln_priors(bestfit, return_prior_parameters=True)
                    fred_prior_values[i_ra[i_pix], i_dec[i_pix], :] = [prior_parameters['f_mean'],
                                                                       prior_parameters['f_fill'],
                                                                       prior_parameters['f_mean_corr']]

                    # change x=ln(f/(1-f)) to f in bestfit
                    alpha = log(0.5) / log(frac_red_mean)
                    x = bestfit_values[i_ra[i_pix], i_dec[i_pix], 0]
                    f = (exp(x) / (1.0 + exp(x)))**(1./alpha)
                    bestfit_values[i_ra[i_pix], i_dec[i_pix], 0] = f
                    # change x=ln(f/(1-f)) to f in percentile
                    x = percentile_values[i_ra[i_pix], i_dec[i_pix], 0:3]
                    f = (exp(x) / (1.0 + exp(x)))**(1./alpha)
                    percentile_values[i_ra[i_pix], i_dec[i_pix], 0:3] = f

                    # calculate mean and fractions above various thresholds
                    x = bestfit[0]
                    A = bestfit[1]
                    s = bestfit[2]
                    meanAV = A * np.exp(s**2 / 2.0)
                    AVthreshvals = [0.1, 0.5, 0.8, 0.9, 1.0, 5.0, 7.0]
                    fdense_01 = 0.5*special.erfc(np.log(AVthreshvals[0]/A) / 
                                                      (np.sqrt(2)*s))
                    fdense_05 = 0.5*special.erfc(np.log(AVthreshvals[1]/A) / 
                                                      (np.sqrt(2)*s))
                    fdense_08 = 0.5*special.erfc(np.log(AVthreshvals[2]/A) / 
                                                      (np.sqrt(2)*s))
                    fdense_09 = 0.5*special.erfc(np.log(AVthreshvals[3]/A) / 
                                                      (np.sqrt(2)*s))
                    fdense_1 = 0.5*special.erfc(np.log(AVthreshvals[3]/A) / 
                                                      (np.sqrt(2)*s))
                    fdense_5 = 0.5*special.erfc(np.log(AVthreshvals[4]/A) / 
                                                      (np.sqrt(2)*s))
                    fdense_7 = 0.5*special.erfc(np.log(AVthreshvals[5]/A) / 
                                                      (np.sqrt(2)*s))
                    derived_value_vec = [meanAV, fdense_01,  fdense_05,  
                                         fdense_08,  fdense_09,  fdense_1,  
                                         fdense_5,  fdense_7]
                    derived_values[i_ra[i_pix], i_dec[i_pix], :] = derived_value_vec

                    # calculate distributions of mean and fractions above various thresholds
                    xvec = d['x']
                    Amedvec = d['A_V']
                    sigmavec = d['sigma']
                    fredvec = (exp(xvec) / (1.0 + exp(xvec)))**(1./alpha)
                    sigma_squaredvec = sigmavec**2
                    #sigma_squaredvec = log((1. + sqrt(1. + 4. * (wvec)**2)) / 2.)
                    #sigmavec = sqrt(sigma_squaredvec)
                    meanAVvec = Amedvec * np.exp(sigma_squaredvec / 2.0)
                    percval = [16, 60, 84]
                    fdensevec_01 = 0.5*special.erfc(np.log(AVthreshvals[0]/Amedvec) / 
                                                      (np.sqrt(2)*sigmavec))
                    fdensevec_05 = 0.5*special.erfc(np.log(AVthreshvals[1]/Amedvec) / 
                                                      (np.sqrt(2)*sigmavec))
                    fdensevec_08 = 0.5*special.erfc(np.log(AVthreshvals[2]/Amedvec) / 
                                                      (np.sqrt(2)*sigmavec))
                    fdensevec_09 = 0.5*special.erfc(np.log(AVthreshvals[3]/Amedvec) / 
                                                      (np.sqrt(2)*sigmavec))
                    fdensevec_1 = 0.5*special.erfc(np.log(AVthreshvals[3]/Amedvec) / 
                                                      (np.sqrt(2)*sigmavec))
                    fdensevec_5 = 0.5*special.erfc(np.log(AVthreshvals[4]/Amedvec) / 
                                                      (np.sqrt(2)*sigmavec))
                    fdensevec_7 = 0.5*special.erfc(np.log(AVthreshvals[5]/Amedvec) / 
                                                      (np.sqrt(2)*sigmavec))
                    derived_names = ['meanAVvec',
                             'fdensevec_01', 'fdensevec_05', 'fdensevec_08', 'fdensevec_09',
                             'fdensevec_1', 'fdensevec_5', 'fdensevec_7']
                    d_derived = {'meanAVvec': meanAVvec, 
                                 'fdensevec_01': fdensevec_01, 
                                 'fdensevec_05': fdensevec_05, 'fdensevec_08': fdensevec_08, 
                                 'fdensevec_09': fdensevec_09, 'fdensevec_1': fdensevec_1, 
                                 'fdensevec_5': fdensevec_5, 'fdensevec_7': fdensevec_7,
                                 'AVthreshvals': AVthreshvals}
                    percentile_derived = array([percentile(d_derived[derived_names[k]], percval) 
                                                for k in range(len(derived_names))])
                    derived_percentile_values[i_ra[i_pix], i_dec[i_pix], :] = percentile_derived.flatten()

                    if (save_single_pix): 
                        print 'Bestfit: ', bestfit
                        print 'alpha: ', alpha
                        print 'frac_red_mean', frac_red_mean
                        print ' best f: ', (exp(bestfit[0]) / (1.0 + exp(bestfit[0])))**(1.0/alpha)
                        fake_cmd = makefakecmd(fg_cmd, color_boundary, qmag_boundary,
                                               [bestfit[0], bestfit[1], bestfit[2], 0.20443, 
                                                (0.33669 - 0.20443)], 
                                               floorfrac=floorfrac_value, 
                                               mask=datamask, SHOWPLOT=False,
                                               noise_model = noise_model,
                                               noise_frac = noise_frac,
                                               frac_red_mean = frac_red_mean,
                                               print_fred = True)

                        # update prior parameters for best fit, and update into processing_params
                        prior_parameters = ln_priors(bestfit, return_prior_parameters=True)
                        allparamvals = locals()
                        processing_params = {k: allparamvals[k] for k in param_names}
                        f_fill = prior_parameters['f_fill']
                        f_mean_corr = prior_parameters['f_mean_corr']
                        p0sig = prior_parameters['p0sig']
                        print 'f_red_geom: %4.2f   f_fill: %4.2f   f_mean_corr: %4.2f ' % (frac_red_mean, f_fill, f_mean_corr)
                        print 'p0sig: ', p0sig
                        print 'Used initial guess of AV = ', AV_init_array[i_ra[i_pix], i_dec[i_pix]]

                        demosavefile = 'demo_modelfit_data_' + str(i_ra[i_pix]) + '_' +    \
                            str(i_dec[i_pix]) + '.npz'
                        print 'Saving demo data to ', demosavefile
                        savez(demosavefile, 
                              processing_params = processing_params,
                              derived_value_vec = derived_value_vec,
                              d_derived = d_derived,
                              percentile_derived = percentile_derived, 
                              #samp = samp,    # python doesn't know how to pickle this class
                              d = d, 
                              bestfit = bestfit, 
                              sigmavec = sigmavec,
                              i_c = i_c, 
                              i_q = i_q, 
                              fg_cmd = fg_cmd, 
                              fake_cmd = fake_cmd, 
                              color_boundary = color_boundary, 
                              qmag_boundary = qmag_boundary,
                              AVthreshvals = AVthreshvals,
                              datafile = datafile,
                              d_arcsec = d_arcsec,
                              deltapixorig = deltapixorig,
                              i_pix = i_pix,
                              i_ra = i_ra[i_pix], 
                              i_dec = i_dec[i_pix],
                              alpha = alpha,
                              frac_red_mean = frac_red_mean,
                              f_fill = f_fill,
                              f_mean_corr = f_mean_corr,
                              p0sig = p0sig)
                        
                        return d_derived, percentile_derived, samp, d, bestfit, sigmavec, \
                            i_c, i_q, fg_cmd, fake_cmd
                
                    ## if requested, plot results

                    if showplot == 'all':  # plot every entry
                        try: 
                            plot_mc_results(d, bestfit, datamag=i_q, datacol=i_c)
                        except:
                            pass

                    if showplot == 'bad':  # plot only questionable fits

                        if ((bestfit[0] < 0.15) | (bestfit[1] > 2) |
                            (bestfit[0] < sigmavec[0]) | (sigmavec[2] < bestfit[0]) |
                            (bestfit[1] < sigmavec[3]) | (sigmavec[5] < bestfit[1]) |
                            (bestfit[2] < sigmavec[6]) | (sigmavec[8] < bestfit[2])):
                            try: 
                                plot_mc_results(d, bestfit, datamag=i_q, datacol=i_c)
                            except:
                                pass
                
                else:        # if not enough stars to fit, insert dummy value

                    dummy = array([-666, -666, -666])
                    bestfit_values[i_ra[i_pix], i_dec[i_pix], :] = [-666, -666, -666]
                    percentile_values[i_ra[i_pix], i_dec[i_pix], :] = [-666, -666, -666,
                                                                        -666, -666, -666,
                                                                        -666, -666, -666]
                    derived_values[i_ra[i_pix], i_dec[i_pix], :] = np.zeros(nz_derived) - 666
                    derived_percentile_values[i_ra[i_pix], i_dec[i_pix], :] = np.zeros(nz_derived_sigma) - 666
                    quality_values[i_ra[i_pix], i_dec[i_pix], :] = [-666, nstar]
                    fred_prior_values[i_ra[i_pix], i_dec[i_pix], :] = [-666, -666, -666]
                    acor = -666
                
                print i_ra[i_pix], i_dec[i_pix], bestfit_values[i_ra[i_pix], i_dec[i_pix]], \
                    fred_prior_values[i_ra[i_pix], i_dec[i_pix]], quality_values[i_ra[i_pix], i_dec[i_pix]]

            plt.figure(4)
            plt.clf()
            plt.imshow(bestfit_values[:,::-1,1],vmin=0, vmax=4, 
                       extent=[ra_range[0], ra_range[1], dec_range[0], dec_range[1]],
                       interpolation='nearest', 
                       origin='upper')
            plt.draw()

        else:

            print 'Skipping lgnstar_annulus: ', i_lgn, ' because ', \
                lgnstar_intervals[i_lgn],',',lgnstar_intervals[i_lgn+1],' out of range ', \
                lgnstar_range[0],',',lgnstar_range[1]

    # Record results to file

    filename = savefile
    if op.isfile(filename):  # check if file exists, to avoid overwrites
        # if it does, append some random characters
        print 'Output file ', filename, ' exists. Changing filename...'
        filenameorig = filename
        filename = op.splitext(filenameorig)[0] + '.' + id_generator(4) + '.npz'
        print 'New name: ', filename

    try: 
        print 'Saving results to ',filename
        savez(filename, 
              bestfit_values=bestfit_values, 
              percentile_values=percentile_values,
              quality_values=quality_values,
              derived_values=derived_values,
              derived_percentile_values=derived_percentile_values,
              fred_prior_values=fred_prior_values,
              ra_bins = ra_local,
              dec_bins = dec_local,
              ra_global = ra_global,
              dec_global = dec_global,
              processing_params = processing_params,
              output_map=output_map)
    except:
        print 'Failed to save file', filename

    try:
        plot_bestfit_results(results_file = filename, brickname=fileroot, pngroot=pngfileroot)
    except:
        print 'Failed to plot results'

    return bestfit_values, percentile_values


def fit_ra_dec_regions(ra, dec, d_arcsec = 10.0, nmin = 15.0,
                       ra_bin_num='', dec_bin_num='',
                       nwalkers=50, nsamp=15, nburn=150,
                       filename=resultsdir + fnroot+'.npz',
                       showplot='bad'):

    i_ra_dec_vals, ra_bins, dec_bins = split_ra_dec(ra, dec, 
                                                    d_arcsec = d_arcsec)

    nx, ny = i_ra_dec_vals.shape
    nz_bestfit = 3
    nz_sigma = nz_bestfit * 3
    nz_quality = 2
    bestfit_values = zeros([nx, ny, nz_bestfit])
    percentile_values = zeros([nx, ny, nz_sigma])
    quality_values = zeros([nx, ny, nz_quality])

    param_init = [0.005, 0.5, 0.4]  # starting at x=0 locks in, because random*0 = 0
    #param_init = [0.5, 0.5, 0.2]   # use for f fitting

    if ra_bin_num == '':
        ra_bin_num = range(len(ra_bins)-1)

    if dec_bin_num == '':
        dec_bin_num = range(len(dec_bins)-1)

    for i_ra in ra_bin_num:
    #for i_ra in [36, 37, 38, 39]:
    #for i_ra in [10, 11, 12, 13, 36, 37, 38, 39]:
    #for i_ra in [18]:

        for i_dec in dec_bin_num:
        #for i_dec in [15, 16, 17, 18, 19, 20]:
        #for i_dec in [35, 36, 37, 38, 39, 15, 16, 17, 18, 19, 20]:
        #for i_dec in [42,43,45]:

            i_c, i_q = get_star_indices(c[i_ra_dec_vals[i_ra, i_dec]], 
                                        q[i_ra_dec_vals[i_ra, i_dec]], 
                                        color_boundary, qmag_boundary)

            nstar = len(i_c)

            if nstar > nmin: 
                samp, d, bestfit, sigma, acor = run_emcee(i_c, i_q,
                                                          param_init,
                                                          nwalkers=nwalkers, 
                                                          nsteps=nsamp, 
                                                          nburn=nburn)
                bestfit_values[i_ra, i_dec, :] = bestfit
                sigmavec = sigma.flatten()
                percentile_values[i_ra, i_dec, :] = sigmavec
                idx = d['lnp'].argmax()
                quality_values[i_ra, i_dec, :] = [d['lnp'][idx], nstar]

                # change x=ln(f/(1-f)) to f in bestfit and percentile
                frac_red_mean = 0.5
                alpha = log(0.5) / log(frac_red_mean)
                x = bestfit_values[i_ra, i_dec, 0]
                #### NOTE: THIS WAS DIFFERENT CONVERSION THAN ELSEWHERE!!!! exponent Fixed, but not tested
                f = (exp(x) / (1.0 + exp(x)))**(1.0/alpha)
                bestfit_values[i_ra, i_dec, 0] = f
                x = percentile_values[i_ra, i_dec, 0:3]
                f = (exp(x) / (1.0 + exp(x)))**(1.0/alpha)
                percentile_values[i_ra, i_dec, 0:3] = f
                
                # if requested, plot results

                if showplot == 'all':  # plot every entry
                    try: 
                        plot_mc_results(d, bestfit, datamag=i_q, datacol=i_c)
                    except:
                        pass

                if showplot == 'bad':  # plot only questionable fits

                    if ((bestfit[0] < 0.15) | (bestfit[1] > 2) |
                        (bestfit[0] < sigmavec[0]) | (sigmavec[2] < bestfit[0]) |
                        (bestfit[1] < sigmavec[3]) | (sigmavec[5] < bestfit[1]) |
                        (bestfit[2] < sigmavec[6]) | (sigmavec[8] < bestfit[2])):
                        try: 
                            plot_mc_results(d, bestfit, datamag=i_q, datacol=i_c)
                        except:
                            pass

                
            else:
                dummy = array([-666, -666, -666])
                bestfit_values[i_ra, i_dec, :] = [-666, -666, -666]
                percentile_values[i_ra, i_dec, :] = [-666, -666, -666,
                                                      -666, -666, -666,
                                                      -666, -666, -666]
                quality_values[i_ra, i_dec, :] = [-666, nstar]
                acor = -666
                
            print i_ra, i_dec, bestfit_values[i_ra, i_dec], acor, quality_values[i_ra, i_dec]

    # check if file exists, to avoid overwrites
    if op.isfile(filename):
        # if it does, append some random characters
        print 'Output file ', filename, ' exists. Changing filename...'
        filenameorig = filename
        filename = op.splitext(filenameorig)[0] + '.' + id_generator(4) + '.npz'
        print 'New name: ', filename

    try: 
        print 'Saving results to ',filename
        savez(filename, 
              bestfit_values=bestfit_values, 
              percentile_values=percentile_values,
              quality_values=quality_values,
              ra_bins = ra_bins,
              dec_bins = dec_bins)
    except:
        print 'Failed to save file', filename

    try:
        plot_bestfit_results(results_file = filename, brickname=filename)
    except:
        print 'Failed to plot results'


    return bestfit_values, percentile_values

def test_resolution(ra, dec, resval=[15, 10, 7.5, 5], brickfrac=0.25):

    rarange = array([min(ra), max(ra)])
    decrange = array([min(dec), max(dec)])
    deccen = average(decrange)
    print 'RA Range:  ', rarange
    print 'Dec Range: ', decrange
    print 'Dec Cen:   ', deccen

    for res in resval: 

        outfile = resultsdir + fnroot + '.' + str(res) + '.npz'

        # get range of ra indices to use
        
        d_ra  = (float(res) / cos(math.pi * deccen / 180.0)) / 3600.0
        d_dec =  float(res) / 3600.0
        print 'dRA:   ', d_ra
        print 'dDec:  ', d_dec
        ntotra  = (1.0 + (rarange[1] - rarange[0]) / d_ra)
        ntotdec = (1.0 + (decrange[1] - decrange[0]) / d_dec)
        nra = int(round(brickfrac * ntotra))
        ira = arange(nra) 
        print 'Analyzing resolution = ', res, ' in ', nra, ' rows from ', ntotra,', ', ntotdec

        #i_ra_dec_vals, ra_bins, dec_bins = split_ra_dec(ra, dec, 
        #                                            d_arcsec = res)
        print 'Saving to ', outfile
        b, s = fit_ra_dec_regions(ra, dec, d_arcsec=res, ra_bin_num = ira, filename=outfile)

    return 

def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    """
    return string of random characters: 
    http://stackoverflow.com/questions/2257441/python-random-string-generation-with-upper-case-letters-and-digits
    """

    return ''.join(rnd.choice(chars) for x in range(size))    

def skew_normal(x, mean=0.0, stddev=1.0, alpha=0.0, align_mode=True, printstuff=False):
    """
    Return a skew_normal distribution with the desired mean and stddev. alpha controls the 
    degree of skewness.  Based on: http://en.wikipedia.org/wiki/Skew_normal_distribution
    """
    
    xsig = stddev / sqrt(1. - (2./pi) * (alpha**2 / (1. + alpha**2)))
    xmean = mean - xsig * (alpha / sqrt(1. + alpha**2)) * sqrt(2./pi)

    if align_mode:   # treat mean as desired mode
        mode_shift_poly = [3.04035639e-29,   5.58273969e-21,  -1.67582716e-26,  -3.62284547e-18,
                           4.30636532e-24,   1.04139549e-15,  -6.77910067e-22,  -1.74658626e-13,
                           7.08698868e-20,   1.89492508e-11,  -4.98639476e-18,  -1.39368656e-09,
                           2.35161849e-16,   7.08204586e-08,  -7.76046641e-15,  -2.48823548e-06,
                           2.20786871e-13,   5.95653863e-05,  -6.65071653e-12,  -9.41834368e-04,
                           1.60896830e-10,   9.31976144e-03,  -2.11035367e-09,  -5.12998001e-02,
                           1.06153311e-08,  -2.71757803e-02,  -8.61473382e-09]  # from find_peak_skew_normal
        mode_shift = poly1d(mode_shift_poly)
        xmean = xmean - stddev * mode_shift(alpha)

    if printstuff:
        print 'Mean: ', mean, ' Stddev:  ',stddev, ' Alpha: ',alpha
        print 'XMean:', xmean, ' XStddev: ',xsig
    
    return 2.0 * norm.pdf(x,loc=xmean,scale=xsig) * norm.cdf(alpha*((x - xmean)/xsig))

def find_peak_skew_normal(mean=0., stddev=1.):
    
    avec = linspace(-10.,10.,1000)
    x = linspace(-5.,5.,1000)
    maxvec = array([nanmax(skew_normal(x,mean=mean,stddev=stddev,alpha=a,align_mode=False)) for a in avec])
    pkvec = array([x[where(skew_normal(x,mean=mean,stddev=stddev,alpha=a,align_mode=False) == 
                     maxvec[i])][0] for i, a in enumerate(avec)])

    plt.plot(avec,pkvec/stddev)
    plt.xlabel('alpha')
    plt.ylabel('Mode / Stddev')

    #approx = 1 - 2.0*norm.cdf(avec/5.6)
    #plt.plot(avec, approx)
    #plt.plot(avec, pkvec - approx)

    #npoly=20
    #param = polyfit(avec, pkvec-approx, npoly)
    #print 'Polynomial fit to residuals: ',param
    #p = poly1d(param)
    #plt.plot(avec, p(avec))

    npoly=26
    param2 = polyfit(avec, pkvec, npoly)
    print 'Polynomial fit to residuals: ',param2
    p2 = poly1d(param2)
    plt.plot(avec, p2(avec))
    plt.plot(avec, pkvec - p2(avec))

############# CODE FOR RUNNING EMCEE AND PLOTTING RESULTS ###################

def ln_priors_function(p, return_prior_parameters=False, f_mean=0.2):

    # set up easy ranges (AV, sig)

    p1 = [0.0001, 10.0]         # median A_V
    p2 = [0.01, 1.5]            # sigma
                              
    if ((p1[0] > p[1]) | (p[1] > p1[1]) | 
        (p2[0] > p[2]) | (p[2] > p2[1])):

        return -np.inf

    # set up ranges for x (i.e., regularlized f_red)

    alpha = np.log(0.5) / np.log(f_mean)
    frange = np.array([0.05, 0.95])
    p0 = np.log(frange**alpha / (1.0 - frange**alpha))
                              
    if ((p0[0] > p[0]) | (p[0] > p0[1])):

        return -np.inf

    # correct geometrical f_mean for A_V-dependent filling factors
    gamma = 3.0
    AV0 = 0.25
    f_mean_min = 0.1
    #f_fill_min = f_mean_min / f_mean
    f_fill_min = 0.66667

    f_fill = f_fill_min + (1.0 - f_fill_min) * ((p[1]/AV0)**gamma / 
                                                (1.0 + (p[1]/AV0)**gamma))
    f_mean_corr = np.maximum(f_mean * f_fill, f_mean_min)
    
    # shift mean of x from 0 at f=f_mean, to x_corr equivalent to f_mean_corr
    x_corr = np.log(f_mean_corr**alpha / (1.0 - f_mean_corr**alpha))
    
    # set up split-normal for x

    #bsig0 = 0.2
    bsig0 = 0.10
    asig0 = 0.15
    sig0max = 1.5
    p0sigma = min(f_mean**0.35 * 10.0**(bsig0 + asig0*p[1]), sig0max)
    p0scale = [1.5, 1.0]
    p0scale1 = 2.0
    p0scale2 = 1.5
    # set split gaussian widths (for < x_corr and >x_corr)
    p0sig1 = p0scale1 * p0sigma * (f_mean_corr / 0.5)**p0scale[0]
    p0sig2 = p0scale2 * p0sigma * ((1.0 - f_mean_corr) / 0.5)**p0scale[1]
    p0sigvec = [p0sig1, p0sig2]
    # automatically select proper sigma, based on p[0]<0 or p[0]>0
    p0sig = p0sigvec[((np.sign(p[0] - x_corr) + 1) / 2).astype(int)]


    # set up log normal for sigma, keeping same mode
    p2mode = 0.35              # w = broad gaussian 
    p2sigma = 0.5
    p2mu = np.log(p2mode) + p2sigma**2   # mu = ln(median)

    if return_prior_parameters:

        return {'p0': p0, 'p1': p1, 'p2': p2, 'frange': frange,
                'gamma': gamma, 'AV0': AV0, 'f_mean_min': f_mean_min, 'AVparam': p[1],
                'f_mean': f_mean, 'f_fill': f_fill, 'f_mean_corr': f_mean_corr, 'x_corr': x_corr,
                'alpha': alpha,
                'p0sigma': p0sigma, 'p0scale': p0scale, 'p0scale1': p0scale1, 'p0scale2': p0scale2, 
                'p0sigvec': p0sigvec, 'p0sig': p0sig,
                'bsig0': bsig0, 'asig0': asig0, 'sig0max': sig0max,
                'p2mode': p2mode, 'p2sigma': p2sigma,   'p2mu': p2mu}

    else: 

        # if all parameters are in range, return the ln of the Gaussian 
        # (for a Gaussian prior on x) and the ln of the log normal prior
        # on sigma (p[2])
        
        # baseline so no zero problems
        lnp = 0.0000001

        # Split gaussian for p[0] = x
        y = (p[0] - x_corr) / p0sig
        lnp += - 0.5 * y**2

        # Log normal on p[2] (= sigma, width of log normal A_V)
        lnp += - np.log(p[2]) - 0.5 * (np.log(p[2]) - p2mu) ** 2 / p2sigma**2

        return lnp


def ln_priors_old(p, return_prior_parameters=False):
    
    # set up ranges

    p0 = [-0.75, 5.0]           # x = ln(f/(1-f)) where fracred -- red fraction
    p1 = [0.0001, 10.0]         # median A_V
    p2 = [0.01, 1.5]           # sigma
                              
    # set up gaussian for x
    p0mean = 0.0              # symmetric in x means mean of f=0.5
    p0stddev = 1.5

    # set up lognormal for x, using AV-dependent prior on the width.
    p0mode = -p0[0]             # symmetric in x means mean of f=0.5
    p0offset = p0[0]
    bsig0 = -0.5
    asig0 = 1.0
    sig0max = 100.
    p0sigma = min(10.0**(bsig0 + asig0*p[1]), sig0max)
    p0mu = np.log(p0mode) + p0sigma**2   # mu = ln(median)

    # set up log normal for sigma, keeping same mode
    p2mode = 0.4              # w = broad gaussian 
    p2sigma = 0.5
    p2mu = np.log(p2mode) + p2sigma**2   # mu = ln(median)

    if return_prior_parameters:

        return {'p0': p0, 'p1': p1, 'p2': p2, 
                'p0mode': p0mode, 'p0sigma': p0sigma,   'p0mu': p0mu, 'p0offset': p0offset,
                'bsig0': bsig0, 'asig0': asig0, 'sig0max': sig0max,
                'p2mode': p2mode, 'p2sigma': p2sigma,   'p2mu': p2mu}

    else: 

        if ((p0[0] > p[0]) | (p[0] > p0[1]) | 
            (p1[0] > p[1]) | (p[1] > p1[1]) | 
            (p2[0] > p[2]) | (p[2] > p2[1])):
            return -Inf

        # if all parameters are in range, return the ln of the Gaussian 
        # (for a Gaussian prior on x) and the ln of the log normal prior
        # on sigma (p[2])
        lnp = 0.0000001
        #lnp += -0.5 * (p[0] - p0mean) ** 2 / p0stddev**2
        lnp += - np.log(p[0]-p0offset) - 0.5 * (np.log(p[0]-p0offset) - p0mu) ** 2 / p0sigma**2
        lnp += -(- np.log(-p0offset) - 0.5 * (np.log(-p0offset) - p0mu) ** 2 / p0sigma**2)  # keep peak constant for changing p0sigma
        lnp += - np.log(p[2]) - 0.5 * (np.log(p[2]) - p2mu) ** 2 / p2sigma**2

        return lnp
    
def ln_prob(param, i_star_color, i_star_magnitude):

    lnp = ln_priors(param)

    if lnp == -Inf:
        return -Inf

    return lnp + cmdlikelihoodfunc(param, i_star_color, i_star_magnitude)
    

def run_emcee(i_color, i_qmag, param=[0.5,1.5,0.4], likelihoodfunction='', priorfunction='',
              nwalkers=50, nsteps=10, nburn=150, nthreads=0, pool=None):
    # NOTE: nthreads not actually enabled!  Keep nthreads=0!

    import emcee

    # setup emcee

    #assert(ln_priors(param)), "First Guess outside the priors"
    assert(priorfunction(param)), "First Guess outside the priors"

    ndim = len(param)

    p0 = [param*(1. + random.normal(0, 0.01, ndim)) for i in xrange(nwalkers)]
    
    if (likelihoodfunction == ''):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, 
                                    args=[i_color,i_qmag], threads=nthreads, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihoodfunction, 
                                    args=[i_color,i_qmag], threads=nthreads, pool=pool)
    
    
    # burn in....
    pos, prob, state = sampler.run_mcmc(p0, nburn)

    # Correlation function values -- keep at least this many nsteps x 2
    try:
        acor = sampler.acor
    except:
        acor = -666.

    # run final...
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)

    names = ['x', 'A_V', 'sigma']
    d = { k: sampler.flatchain[:,e] for e, k in enumerate(names) }
    d['lnp'] = sampler.lnprobability.flatten()
    idx = d['lnp'].argmax()

    bestfit = array([d[names[k]][idx] for k in range(3)])
    percval = [16, 50, 84]
    sigma = array([percentile(d[names[k]], percval) for k in range(3)])

    return sampler, d, bestfit, sigma, acor


def plot_mc_results(d, bestfit, datamag=0.0, datacol=0.0, 
                    keylist=['x', 'A_V', 'sigma', 'lnp'],
                    model = ''):

    ##  morgan's plotting code, takes standard keywords
    #ezfig.plotCorr(d, d.keys())
    # ezfig.plotMAP(d, d.keys())

    # Plot density of points in parameter banana diagrams

    plt.figure(1, figsize=(6,6))
    plt.clf()
    ezfig.plotCorr(d, keylist, plotfunc=ezfig.plotDensity, bins=50)
    plt.draw()


    # Plot histograms of posterior distributions 

    plt.figure(2, figsize=(6,6))
    plt.clf()
    #for e, (k, v) in enumerate(d.items()):
    for e, k in enumerate(keylist):
        v = d[k]
        ax = plt.subplot(2,2,e+1)
        ezfig.plotMAP(v, ax=ax, frac=[0.65, 0.95, 0.975], hpd=False)
        ax.set_xlabel(k)
    plt.draw()

    if (model == ''):
        model = array(makefakecmd_AVparamsonly(bestfit))
    extent = [color_boundary[0], color_boundary[-1],
              qmag_boundary[-1], qmag_boundary[0]]

    # Plot CMDs of model, data, unreddened CMD, and data-model

    if len(datamag) > 1:

        fig = plt.figure(3, figsize=(9,9))
        plt.clf()
        
        #fig, (axs) = plt.subplots(ncols=2, nrows=2, squeeze=False,
        #                            figsize=(9,9))
        #ax = axs[0][0]

        ######## plot model
        plt.subplot(2,2,1)

        plt.imshow(model,  extent=extent, origin='upper', aspect='auto', 
                       interpolation='nearest')
        plt.xlabel('F110W - F160W')
        plt.ylabel('Extinction Corrected F160W')
        plt.title('Model')

        imgsize = model.shape
        rangevec = [[0, imgsize[0]], [0, imgsize[1]]]
        dataimg, junk1, junk2 = histogram2d(datamag, datacol, 
                                            range=rangevec,
                                            bins=nbins)
        dataimg = dataimg / dataimg.sum()

        ######## plot data

        #ax = axs[0][1]
        plt.subplot(2,2,2)

        im = plt.imshow(dataimg, 
                           extent=extent, origin='upper', 
                           aspect='auto', interpolation='nearest')
        plt.xlabel('F110W - F160W')
        plt.ylabel('Extinction Corrected F160W')
        plt.title('Data')
        #plt.colorbar(im)

        ######## plot residuals

        #ax = axs[1][1]
        plt.subplot(2,2,4)

        resid = dataimg - model

        minresid = nanmin(resid)
        maxresid = nanmax(resid)
        max = nanmax([abs(minresid), maxresid])
        im = plt.imshow(resid, cmap='seismic', vmin=-max, vmax=max,
                           extent=extent, origin='upper', 
                           aspect='auto', interpolation='nearest')
        plt.xlabel('F110W - F160W')
        plt.ylabel('Extinction Corrected F160W')
        plt.title('Residuals')
        plt.colorbar(im)

        ######## plot unreddened model

        #ax = axs[1][0]
        plt.subplot(2,2,3)

        unreddenedmodel = array(makefakecmd_AVparamsonly([0.01,0.01,0.1]))

        im = plt.imshow(unreddenedmodel, 
                           extent=extent, origin='upper', 
                           aspect='auto', interpolation='nearest')
        plt.xlabel('F110W - F160W')
        plt.ylabel('Extinction Corrected F160W')
        plt.title('Foreground')
        plt.draw()

    else:

        fig = plt.figure(3, figsize=(9,9))
        plt.clf()

        plt.imshow(model,  extent=extent, origin='upper', aspect='auto', 
                     interpolation='nearest')
        plt.xlabel('F110W - F160W')
        plt.ylabel('Extinction Corrected F160W')
        #plt.title('Model')

    plt.draw()


def plot_bestfit_results(results_file = resultsdir + fnroot+'.npz', 
                         brickname=' ', pngroot=''):

    dat = load(results_file)
    bf = dat['bestfit_values']
    p  = dat['percentile_values']
    rabin = dat['ra_bins']
    decbin = dat['dec_bins']
    print 'rabin:  ', len(rabin), '[', min(rabin), ',', max(rabin), ']'
    print 'decbin: ', len(decbin), '[', min(decbin), ',', max(decbin), ']'
    #rangevec = [min(decbin), max(decbin), min(rabin), max(rabin)]
    rangevec = [max(decbin), min(decbin), max(rabin), min(rabin)]
    dra = (max(rabin) - min(rabin)) * sin((min(decbin) + max(decbin))/2.0)
    ddec = min(decbin) + max(decbin)
    rat = dra / ddec
    
    # Best fit results

    plt.figure(1)
    plt.close()
    fig1 = plt.figure(1, figsize=(14,7))
    plt.clf()
    plt.suptitle(brickname)
    plt.subplots_adjust(left=0.02, right=0.98)

    plt.subplot(1,3,1)
    A = bf[:,:,1]
    #im = plt.imshow(bf[:,:,1],vmin=0, vmax=4, interpolation='nearest', 
    im = plt.imshow(bf[:,::-1,1],vmin=0, vmax=4, interpolation='nearest', 
                    extent=rangevec, origin='upper', cmap='hot')
    plt.colorbar(im)
    plt.title('$A_V$')
    
    plt.subplot(1,3,2)
    im = plt.imshow(bf[:,::-1,0],vmin=0, vmax=1, interpolation='nearest', 
                    extent=rangevec, origin='upper', 
                    cmap='seismic')
    plt.colorbar(im)
    plt.title('$f_{reddened}$')
    
    plt.subplot(1,3,3)
    im = plt.imshow(bf[:,::-1,2],vmin=0.2, vmax=0.7, interpolation='nearest', 
                    extent=rangevec, origin='upper', cmap='hot')
    plt.colorbar(im)
    plt.title('$\sigma$')

    if (pngroot != ''):
        plt.savefig(pngroot + '.1.png', bbox_inches=0)
        
    # Uncertainty results

    sigf = (p[:,::-1,2] - p[:,::-1,0]) / 2.0
    sigA = (p[:,::-1,5] - p[:,::-1,3]) / 2.0
    sigw = (p[:,::-1,8] - p[:,::-1,6]) / 2.0

    plt.figure(2)
    plt.close()

    fig2 = plt.figure(2, figsize=(14,7))
    plt.clf()
    plt.suptitle(brickname)
    plt.subplots_adjust(left=0.02, right=0.98)

    plt.subplot(1,3,1)
    im = plt.imshow(sigA/bf[:,::-1,1],vmin=0, vmax=1.0, interpolation='nearest', 
                    extent=rangevec, origin='upper', 
                    cmap='gist_ncar')
    plt.colorbar(im)
    plt.title('$\Delta A_V / A_V$')
    
    plt.subplot(1,3,2)
    im = plt.imshow(sigf,vmin=0, vmax=0.5, interpolation='nearest', 
                    extent=rangevec, origin='upper', 
                    cmap='gist_ncar')
    plt.colorbar(im)
    plt.title('$\Delta f_{reddened}$ ')
    
    plt.subplot(1,3,3)
    im = plt.imshow(sigw / (bf[:,::-1,2]),vmin=0.1, vmax=1.0, interpolation='nearest', 
                    extent=rangevec, origin='upper', 
                    cmap='gist_ncar')
    plt.colorbar(im)
    plt.title('$\Delta \sigma / \sigma$')

    if (pngroot != ''):
        plt.savefig(pngroot + '.2.png', bbox_inches=0)
        
    # Best fit value scatter plots

    plt.figure(3)
    plt.close()
    fig3 = plt.figure(3, figsize=(10,7))
    plt.clf()
    plt.suptitle(brickname)

    plt.subplot(2,2,1)
    im = plt.scatter(bf[:,:,1],bf[:,:,0],c=bf[:,:,2],s=7,linewidth=0,
                     alpha=0.4, vmin=0, vmax=1.5)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$f_{red}$')
    plt.axis([0, 4, 0, 1])
    
    plt.subplot(2,2,2)
    im = plt.scatter(bf[:,:,1],bf[:,:,2],c=bf[:,:,0],s=7,linewidth=0,alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\sigma$')
    plt.axis([0, 4, 0, 1.5])
    
    plt.subplot(2,2,3)
    im = plt.scatter(bf[:,:,1],bf[:,:,2],c=bf[:,:,0],s=7,linewidth=0,alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\sigma$')
    plt.axis([0, 4, 0, 1.5])
    
    plt.subplot(2,2,4)
    im = plt.scatter(bf[:,:,0],bf[:,:,2],c=bf[:,:,1],s=7,linewidth=0,alpha=0.4,
                     vmin=0, vmax=4)
    plt.colorbar(im)
    plt.xlabel('$f_{red}$')
    plt.ylabel('$\sigma$')
    plt.axis([0, 1, 0, 1.5])
    
    if (pngroot != ''):
        plt.savefig(pngroot + '.3.png', bbox_inches=0)
        
    # Uncertainty scatter plots, vs self on x-axis

    # Uncertainty results

    sigf = (p[:,:,2] - p[:,:,0]) / 2.0
    sigA = (p[:,:,5] - p[:,:,3]) / 2.0
    sigw = (p[:,:,8] - p[:,:,6]) / 2.0

    plt.figure(4)
    plt.close()
    fig4 = plt.figure(4, figsize=(10,7))
    plt.clf()
    plt.suptitle(brickname)

    plt.subplot(2,2,1)
    #im = plt.scatter(bf[:,::-1,1], sigA / bf[:,::-1,1], c=bf[:,::-1,0],
    im = plt.scatter(bf[:,:,1], sigA / bf[:,:,1], c=bf[:,:,0],
                         s=7, linewidth=0,
                     alpha=0.4, vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta A_V / A_V$')
    plt.axis([0, 4, 0, 1.5])
    
    plt.subplot(2,2,2)
    im = plt.scatter(bf[:,:,0], sigf, c=bf[:,:,1],
                     s=7, linewidth=0, alpha=0.4,
                     vmin=0, vmax=4)
    plt.colorbar(im)
    plt.xlabel('$f_{red}$')
    plt.ylabel('$\Delta f_{red}$')
    plt.axis([0, 1, 0, 0.5])
    
    plt.subplot(2,2,3)
    im = plt.scatter(bf[:,:,2], sigw / (bf[:,:,2]), c=bf[:,:,1],
                     s=7, linewidth=0, alpha=0.4,
                     vmin=0, vmax=4)
    plt.colorbar(im)
    plt.xlabel('$\sigma$')
    plt.ylabel('$\Delta \sigma / \sigma$')
    plt.axis([0, 1.5, 0, 1])
    
    plt.subplot(2,2,4)
    im = plt.scatter(bf[:,:,2],sigw / (bf[:,:,2]), c=bf[:,:,0],
                     s=7, linewidth=0, alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$\sigma$')
    plt.ylabel('$\Delta \sigma / \sigma$')
    plt.axis([0, 1.5, 0, 1])
    
    if (pngroot != ''):
        plt.savefig(pngroot + '.4.png', bbox_inches=0)
        
    # Uncertainty scatter plots, vs A_V on x-axis

    plt.figure(5)
    plt.close()
    fig5 = plt.figure(5, figsize=(10,7))
    plt.clf()
    plt.suptitle(brickname)

    plt.subplot(2,2,1)
    im = plt.scatter(bf[:,:,1], sigA/bf[:,:,1], c=bf[:,:,0],
                     s=7, linewidth=0, alpha=0.4, 
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta A_V / A_V$')
    plt.axis([0, 4, 0, 1.5])
    
    plt.subplot(2,2,2)
    im = plt.scatter(bf[:,:,1], sigf, c=bf[:,:,0],
                     s=7, linewidth=0, alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta f_{red}$')
    plt.axis([0, 4, 0, 0.5])
    
    plt.subplot(2,2,3)
    im = plt.scatter(bf[:,:,1], sigw / (bf[:,:,2]), c=bf[:,:,0],
                     s=7, linewidth=0, alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta \sigma / \sigma)$')
    plt.axis([0, 4, 0, 1])

    if (pngroot != ''):
        plt.savefig(pngroot + '.5.png', bbox_inches=0)
        
    # Force display

    plt.draw()

    # Write files to PDF (tried writing multiple to PDF, but too big!)
    #pp = PdfPages('../Results/test.pdf')
    #pp.savefig(fig1)
    #pp.savefig(fig2)
    #pp.savefig(fig3)
    #pp.savefig(fig4)
    #pp.savefig(fig5)
    #pp.close()

def show_model_examples():

    Avals = [0.25, 1, 2.5]
    wvals = [0.1, 0.5, 1]
    wfacs = [0.1, 0.25, 0.5]

    fig = plt.figure(1, figsize=(9,9))

    for i in range(len(Avals)):

        for j in range(len(wfacs)):

            AV = Avals[i]
            w = wfacs[j] * AV
            params = [0.0, AV, w]
            params = [0.5, AV, w]

            model = array(makefakecmd_AVparamsonly(params))
            extent = [color_boundary[0], color_boundary[-1],
                      qmag_boundary[-1], qmag_boundary[0]]

            #fig, (axs) = plt.subplots(ncols=len(Avals), nrows=len(wvals), 
            #                          squeeze=False, figsize=(9,9))
            #ax = axs[i][j]
            print i, j, i*len(wvals) + j, AV, w, wfacs[j]
            ax = fig.add_subplot(len(Avals), len(wfacs), i*len(wfacs) + j + 1)

            im = ax.imshow(model,  extent=extent, 
                           origin='upper', aspect='auto', 
                           interpolation='nearest')
            #ax.set_xlabel('F110W - F160W')
            #ax.set_ylabel('Extinction Corrected F160W')
            ax.set_title('AV: %s  w/AV: %s' % (AV,wfacs[j]))

    plt.draw()

#  set up to allow calls from the shell with arguments
import sys
#
if __name__ == '__main__':
  datafile = sys.argv[1]
  deltapixorig = [float(sys.argv[2]), float(sys.argv[3])]
  d_arcsec = float(sys.argv[4])
  results_extension = sys.argv[5]
  d_pix_offset = [0,0]
  if (len(sys.argv) > 6):
      d_pix_offset = [float(sys.argv[6]), float(sys.argv[7])]
  print 'Datafile: ', datafile
  print 'deltapixorig: ', deltapixorig
  print 'd_arcsec: ', d_arcsec
  print 'results_extension', results_extension
  print 'd_pix_offset', d_pix_offset
  bfarray, percarray = run_one_brick(datafile, deltapixorig=deltapixorig, d_arcsec=d_arcsec, results_extension=results_extension, datadir='/mnt/angst4/dstn/v8/', showplot='', d_pix_offset = d_pix_offset)
