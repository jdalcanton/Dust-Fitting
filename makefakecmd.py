import pyfits
from numpy import *
import matplotlib.pyplot as plt
import pylab
import PIL
import time
import ezfig  # morgan's plotting code
import read_brick_data as rbd
from scipy import ndimage


# Principle: "rotate" to frame where all "colors" are along reddening
# vector. Rotate the -data- into this frame, once. Rotate the original
# unreddened CMD -once-.  Then all convolutions are fast 1-d.


############# code for generating foreground CMD ###############

def split_ra_dec(ra, dec, d_arcsec=10.0): 
    """
    indices, ra_bins, dec_bins = split_ra_dec(ra, dec, d_arcsec=10.0)

    Takes lists of ra and dec, divides the list into pixels of width
    d_arcsec, and returns an array, where each entry is a list of the
    indices from the original ra, dec list whose ra and dec fell into
    that pixel. Also returns the boundaries of the bins used to
    generate the index array. (Functions like reverse indices in IDL's
    histogram function)

    To setup: 
    filename = '../../Data/12056_M31-B15-F09-IR_F110W_F160W.st.fits'
    fitstab = pyfits.open(filename)
    ra  = fitstab[1].data.field('RA') 
    dec = fitstab[1].data.field('DEC')
    fitstab.close()
    """
    # figure out ranges
    ra_min =  nanmin(ra)
    ra_max =  nanmax(ra)
    dec_min = nanmin(dec)
    dec_max = nanmax(dec)

    # find center
    ra_cen = (ra_max - ra_min) / 2.0
    dec_cen = (dec_max - dec_min) / 2.0

    # figure out number of bins, rounding up to nearest integer
    d_ra  = (d_arcsec / cos(math.pi * dec_cen / 180.0)) / 3600.0
    d_dec =  d_arcsec / 3600.0
    n_ra  = int((ra_max - ra_min)   / d_ra  + 1.0)
    n_dec = int((dec_max - dec_min) / d_dec + 1.0)

    # calculate histogram bin edges
    # ra_bins  = ra_min  + d_ra  * arange(n_ra  + 1, dtype=float)
    # dec_bins = dec_min + d_dec * arange(n_dec + 1, dtype=float)
    ra_bins  = linspace(ra_min, ra_min  + n_ra*d_ra,   num=n_ra+1)
    dec_bins = linspace(dec_min,dec_min + n_dec*d_dec, num=n_dec+1) 
    # print 'RA Range:  ', ra.min(), ra.max()
    # print 'Dec Range: ', dec.min(), dec.max()
    print 'RA Range:  ',ra_bins.min(), ra_bins.max()
    print 'Dec Range: ', dec_bins.min(), dec_bins.max()
    
    
    # recalculate maximums and center, since range is not necessarily 
    # an integer multiple of d_arcsec
    ra_max  = ra_bins.max()
    dec_max = dec_bins.max()
    ra_cen = (ra_max - ra_min) / 2.0
    dec_cen = (dec_max - dec_min) / 2.0

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

    cmap, cmapgoodvals, cstdmap, cstdmapgoodvals = 
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
    cstdmap = empty( indices.shape, dtype=float )

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
            cstdmap[i,j] = std(dctmp.compress(((mtmp > mrange[0]) & 
                                               (mtmp<mrange[1]) & 
                                               (ctmp > crange[0]) & 
                                               (ctmp<crange[1])).flat))
        else:

            cmap[i,j] = emptyval
            cstdmap[i,j] = emptyval

    # calculate list of good values

    igood = where(cmap > -1)
    print 'Median Color offset:', median(cmap[igood])

    return cmap, cmap[igood], cstdmap, cstdmap[igood]


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

    cm,cmlist,cstd,cstdlist = median_rgb_color_map(indices, c, m,
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
           and translate mag limits accordingly. Assume value = [c0,t]
    clim = color to cut on blue side

    Returns mask of 1's where data falls within magnitude limits
    """
    mask = 1.0 + zeros(fg_cmd.shape)

    if useq != 0:
        c0 = useq[0]
        t = useq[1]
        mlim2_faint  = m2range[1] + (cedges[:-1] - c0)*sin(t)/cos(t)
        mlim2_bright = m2range[0] + (cedges[:-1] - c0)*sin(t)/cos(t)
        mlim1_faint  = ((m1range[1] - cedges[:-1]) + 
                        (cedges[:-1] - c0)*sin(t)/cos(t))
        mlim1_bright = ((m1range[0] - cedges[:-1]) + 
                        (cedges[:-1] - c0)*sin(t)/cos(t))
    else:
        mlim2_faint  = m2range[1] + 0.0*cedges[:-1]
        mlim2_bright = m2range[0] + 0.0*cedges[:-1]
        mlim1_faint  = (m1range[1] - cedges[:-1])
        mlim1_bright = (m1range[0] - cedges[:-1])

    nmag = fg_cmd.shape[0]
    ncol = fg_cmd.shape[1]

    mask = array([where((cedges[:-1] > clim[i]) & (medges[i] > mlim2_bright) & 
                        (medges[i] < mlim2_faint) & (medges[i] > mlim1_bright)&
                        (medges[i] < mlim1_faint), 
                        1.0, 0.0) for i in range(nmag)])

    return mask

############# CODE TO GENERATE MODEL CMD #######################

def makefakecmd(fg_cmd, cvec, mvec, AVparam, c0, floorfrac=0.0, 
                mask = 1.0, SHOWPLOT=True, 
                noise_model=0.0, noise_frac=0.0):
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
    
    fracred = AVparam[0]
    #fracred = exp(AVparam[0]) / (1. + exp(AVparam[0])) # x = ln(f/(1-f))
    medianAV = AVparam[1]
    stddev = AVparam[2] * medianAV
    #stddev = AVparam[2]
    sigma = log((1. + sqrt(1. + 4. * (stddev/medianAV)**2)) / 2.)
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

    cmdcombo = cmdcombo + noise_model

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
                       reference_color, floorfrac=floorfrac_value, 
                       mask=color_qmag_datamask, SHOWPLOT=False,
                       noise_model = noise_model)

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

fnroot = 'ir-sf-b12-v8-st'
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

clo, mlo, ilo, cnar, mnar, inr, cm, cstd = isolate_low_AV_color_mag(
                    filename=fn, frac=0.025, mrange=mfitrange, d_arcsec=10.)

qnar = mnar + (cnar-reference_color)*sin(t)/cos(t)

foreground_cmd_orig, qmag_boundary, color_boundary = histogram2d(qnar, cnar, 
                                 range=[sort(m160range), crange], bins=nbins)

foregroundmask, meancol, sigcol = clean_fg_cmd(foreground_cmd_orig, [3.0,3.5], 
                                               niter=4, showplot=0)

foreground_cmd = foreground_cmd_orig * foregroundmask

# make the noise model by masking out foreground and smoothing

noisemask, meancol, sigcol = clean_fg_cmd(foreground_cmd_orig, [4.5,4.5], 
                                          niter=5, showplot=0)
noisemask = abs(noisemask - 1)
noise_smooth = [3, 10]  # mag, color, in pixels
noise_model_orig = foreground_cmd_orig * noisemask
noise_model = ndimage.filters.uniform_filter(noise_model_orig,
                                             size=noise_smooth)

# generate mask of data regions to ignore

nsig_blue_color_cut = 2.0
bluecolorlim = color_boundary[maximum(rint(meancol - nsig_blue_color_cut * 
                                           sigcol).astype(int),0)]
color_qmag_datamask = make_data_mask(foreground_cmd, color_boundary, 
                                     qmag_boundary, m110range, m160range, 
                                     bluecolorlim, useq=[reference_color, t])

# relative normalization of foreground model and noise model

nfg = (foreground_cmd * color_qmag_datamask).sum()
nnoise = (noise_model * color_qmag_datamask).sum()
frac_noise = nnoise / (nnoise + nfg)
print 'Noise fraction: ', frac_noise

# read in main data file

m1, m2, ra, dec = rbd.read_mag_position_gst(fn)
m = array(m2)
c = array(m1 - m2)
q = m + (c-reference_color)*sin(t)/cos(t)
ra = array(ra)
dec = array(dec)

# exclude data outside the color-magnitude range

igood = where((c > color_boundary[0]) & (c < color_boundary[-1]) &
              (q > qmag_boundary[0]) & (q < qmag_boundary[-1]))
m = m[igood]
c = c[igood]
q = q[igood]
ra = ra[igood]
dec = dec[igood]

# exclude data outside the data mask

i_c, i_q = get_star_indices(c, q, color_boundary, qmag_boundary)
igood = where(color_qmag_datamask[i_q, i_c] != 0)
m = m[igood]
c = c[igood]
q = q[igood]
ra = ra[igood]
dec = dec[igood]
i_c = i_c[igood]
i_q = i_q[igood]

# split into RA-Dec

binarcsec = 10.0
i_ra_dec_vals, ra_bins, dec_bins = split_ra_dec(ra, dec, d_arcsec = binarcsec)

# grab data for a test bin, if needed for testing.

i_test_ra = 60
i_test_dec = 20
ctest = c[i_ra_dec_vals[i_test_ra,i_test_dec]]
mtest = m[i_ra_dec_vals[i_test_ra,i_test_dec]]
qtest = q[i_ra_dec_vals[i_test_ra,i_test_dec]]
i_ctest, i_qtest = get_star_indices(ctest, qtest, color_boundary, qmag_boundary)


########### CODE FOR LOOPING THROUGH RA-DEC BINS, RUNNING EMCEE ########

def get_ra_dec_bin(i_ra=60, i_dec=20):
    
    i_ctest, i_qtest = get_star_indices(c[i_ra_dec_vals[i_ra,i_dec]], 
                                        q[i_ra_dec_vals[i_ra,i_dec]], 
                                        color_boundary, qmag_boundary)
    return i_ctest, i_qtest


def fit_ra_dec_regions(ra, dec, d_arcsec = 10.0, nmin = 30.0,
                       nwalkers=100, nsamp=15, nburn=100,
                       filename=resultsdir + fnroot+'.npz'):

    i_ra_dec_vals, ra_bins, dec_bins = split_ra_dec(ra, dec, 
                                                    d_arcsec = d_arcsec)

    nx, ny = i_ra_dec_vals.shape
    nz_bestfit = 3
    nz_sigma = nz_bestfit * 3
    bestfit_values = zeros([nx, ny, nz_bestfit])
    percentile_values = zeros([nx, ny, nz_sigma])

    #param_init = [0.0, 0.5, 0.2]
    param_init = [0.5, 0.5, 0.2]

    for i_ra in range(len(ra_bins)-1):
    #for i_ra in [36, 37, 38, 39]:
    #for i_ra in [10, 11, 12, 13, 36, 37, 38, 39]:
    #for i_ra in [18]:

        for i_dec in range(len(dec_bins)-1):
        #for i_dec in [15, 16, 17, 18, 19, 20]:
        #for i_dec in [35, 36, 37, 38, 39, 15, 16, 17, 18, 19, 20]:
        #for i_dec in [42,43,45]:

            i_c, i_q = get_star_indices(c[i_ra_dec_vals[i_ra, i_dec]], 
                                        q[i_ra_dec_vals[i_ra, i_dec]], 
                                        color_boundary, qmag_boundary)

            if len(i_c) > nmin: 
                samp, d, bestfit, sigma, acor = run_emcee(i_c, i_q,
                                                          param_init,
                                                          nwalkers=nwalkers, 
                                                          nsteps=nsamp, 
                                                          nburn=nburn)
                bestfit_values[i_ra, i_dec, :] = bestfit
                percentile_values[i_ra, i_dec, :] = sigma.flatten()
                
                try: 
                    plot_mc_results(d, bestfit, datamag=i_q, datacol=i_c)
                except:
                    pass
            else:
                dummy = array([-1, -1, -1])
                bestfit_values[i_ra, i_dec, :] = [-1, -1, -1]
                percentile_values[i_ra, i_dec, :] = [-1, -1, -1,
                                                      -1, -1, -1,
                                                      -1, -1, -1]
                acor = -1
                
            print i_ra, i_dec, bestfit_values[i_ra, i_dec], acor

    try: 
        print 'Saving results to ',filename
        savez(filename, 
              bestfit_values=bestfit_values, 
              percentile_values=percentile_values,
              ra_bins = ra_bins,
              dec_bins = dec_bins)
        plot_bestfit_results(results_file = filename, brickname=filename)
    except:
        pass

    return bestfit_values, percentile_values

    
############# CODE FOR RUNNING EMCEE AND PLOTTING RESULTS ###################

def ln_priors(p):
    
    # set up ranges

    #p0 = [-4.0, 4.0]          # x = ln(f/(1-f)) where fracred -- red fraction
    p0 = [0.05, 0.95]         # fracred -- red fraction
    p1 = [0.0001, 6.0]        # median A_V
    p2 = [0.01, 3.0]          # sigma_A/A_V (lognormal stddev / median)
                              
    # set up gaussians
    #p0mean = 0.0              # symmetric in x means mean of f=0.5
    #p0stddev = 1.0
    p0mean = 0.5              # f=0.5 when not much other information
    p0stddev = 0.25

    AV_pix = deltapix_approx[0] / Acol_AV
    #p1mean = 0.5 * AV_pix     # drive to low A_V if not much information
    #p1stddev = 8.0            #  ...but, keep it wide so little influence

    p2mean = 1.0              # w = broad gaussian 
    p2stddev = p2[1] - p2mean
    #p2mean = AV_pix              # sigma_A
    #p2stddev = p2[1] - p2mean

    # return -Inf if the parameters are out of range
    if ((p0[0] > p[0]) | (p[0] > p0[1]) | 
        (p1[0] > p[1]) | (p[1] > p1[1]) | 
        (p2[0] > p[2]) | (p[2] > p2[1])) :
        return -Inf

    # if all parameters are in range, return the ln of the Gaussian 
    # (for a Gaussian prior)
    lnp = 0.0000001
    lnp += -0.5 * (p[0] - p0mean) ** 2 / p0stddev**2
    #lnp += -0.5 * (p[1] - p1mean) ** 2 / p1stddev**2
    lnp += -0.5 * (p[2] - p2mean) ** 2 / p2stddev**2
    return lnp

    
def ln_prob(param, i_star_color, i_star_magnitude):

    lnp = ln_priors(param)

    if lnp == -Inf:
        return -Inf

    return lnp + cmdlikelihoodfunc(param, i_star_color, i_star_magnitude)
    

def run_emcee(i_color, i_qmag, param=[0.5,1.5,0.2], 
              nwalkers=100, nsteps=10, nburn=100):

    import emcee

    # setup emcee

    assert(ln_priors(param)), "First Guess outside the priors"

    ndim = len(param)

    p0 = [param*(1. + random.normal(0, 0.01, ndim)) for i in xrange(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, 
                                    args=[i_color,i_qmag])
    
    # burn in....
    pos, prob, state = sampler.run_mcmc(p0, nburn)

    # Correlation function values -- keep at least this many nsteps x 2
    acor = sampler.acor

    # run final...
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)

    names = ['x', 'A_V', 'w']
    d = { k: sampler.flatchain[:,e] for e, k in enumerate(names) }
    d['lnp'] = sampler.lnprobability.flatten()
    idx = d['lnp'].argmax()

    bestfit = array([d[names[k]][idx] for k in range(3)])
    percval = [16, 50, 84]
    sigma = array([percentile(d[names[k]], percval) for k in range(3)])

    return sampler, d, bestfit, sigma, acor


def plot_mc_results(d, bestfit, datamag=0.0, datacol=0.0, 
                    keylist=['x', 'A_V', 'w', 'lnp']):

    ##  morgan's plotting code, takes standard keywords
    #ezfig.plotCorr(d, d.keys())
    # ezfig.plotMAP(d, d.keys())

    # Plot density of points in parameter banana diagrams

    plt.figure(1)
    plt.clf()
    ezfig.plotCorr(d, keylist, plotfunc=ezfig.plotDensity, bins=50)
    plt.draw()


    # Plot histograms of posterior distributions 

    plt.figure(2)
    plt.clf()
    #for e, (k, v) in enumerate(d.items()):
    for e, k in enumerate(keylist):
        v = d[k]
        ax = plt.subplot(2,2,e+1)
        ezfig.plotMAP(v, ax=ax, frac=[0.65, 0.95, 0.975], hpd=False)
        ax.set_xlabel(k)
    plt.draw()

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

    else:

        fig = plt.figure(3, figsize=(9,9))
        plt.clf()

        plt.imshow(model,  extent=extent, origin='upper', aspect='auto', 
                     interpolation='nearest')
        plt.xlabel('F110W - F160W')
        plt.ylabel('Extinction Corrected F160W')
        #plt.title('Model')

    plt.draw()


def plot_bestfit_results(results_file = resultsdir + fnroot+'.npz', brickname=' '):

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
    fig = plt.figure(1, figsize=(17,7))
    plt.clf()
    plt.suptitle(brickname)

    plt.subplot(1,3,1)
    A = bf[:,:,1]
    #im = plt.imshow(bf[:,:,1],vmin=0, vmax=4, interpolation='nearest', 
    im = plt.imshow(bf[:,::-1,1],vmin=0, vmax=4, interpolation='nearest', 
                    extent=rangevec, origin='upper')
    plt.colorbar(im)
    plt.title('$A_V$')
    
    plt.subplot(1,3,2)
    im = plt.imshow(bf[:,::-1,0],vmin=0, vmax=1, interpolation='nearest', 
                    extent=rangevec, origin='upper', 
                    cmap='seismic')
    plt.colorbar(im)
    plt.title('$f_{reddened}$')
    
    plt.subplot(1,3,3)
    im = plt.imshow(bf[:,::-1,1]*bf[:,::-1,2],vmin=0, vmax=2, interpolation='nearest', 
                    extent=rangevec, origin='upper')
    plt.colorbar(im)
    plt.title('$\sigma_{A_V}$')
        
    # Uncertainty results

    sigf = (p[:,::-1,2] - p[:,::-1,0]) / 2.0
    sigA = (p[:,::-1,5] - p[:,::-1,3]) / 2.0
    sigw = (p[:,::-1,8] - p[:,::-1,6]) / 2.0

    plt.figure(2)
    plt.close()

    fig = plt.figure(2, figsize=(17,7))
    plt.clf()
    plt.suptitle(brickname)

    plt.subplot(1,3,1)
    im = plt.imshow(sigA/bf[:,::-1,1],vmin=0, vmax=1.5, interpolation='nearest', 
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
    im = plt.imshow(sigw / (bf[:,::-1,1]*bf[:,::-1,2]),vmin=0, vmax=4, interpolation='nearest', 
                    extent=rangevec, origin='upper', 
                    cmap='gist_ncar')
    plt.colorbar(im)
    plt.title('$\Delta(\sigma_{A_V}/A_V) / (\sigma_{A_V}/A_V)$')

    # Best fit value scatter plots

    plt.figure(3)
    plt.close()
    fig = plt.figure(3, figsize=(10,7))
    plt.clf()
    plt.suptitle(brickname)

    plt.subplot(2,2,1)
    im = plt.scatter(bf[:,:,1],bf[:,:,0],c=(bf[:,:,2]*bf[:,:,1]),s=7,linewidth=0,
                     alpha=0.4, vmin=0, vmax=2)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$f_{red}$')
    plt.axis([0, 4, 0, 1])
    
    plt.subplot(2,2,2)
    im = plt.scatter(bf[:,:,1],bf[:,:,2]*bf[:,:,1],c=bf[:,:,0],s=7,linewidth=0,alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\sigma_{A_V}$')
    plt.axis([0, 4, 0, 3])
    
    plt.subplot(2,2,3)
    im = plt.scatter(bf[:,:,1],bf[:,:,2],c=bf[:,:,0],s=7,linewidth=0,alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\sigma_{A_V} / A_V$')
    plt.axis([0, 4, 0, 3])
    
    plt.subplot(2,2,4)
    im = plt.scatter(bf[:,:,0],bf[:,:,2],c=bf[:,:,1],s=7,linewidth=0,alpha=0.4,
                     vmin=0, vmax=4)
    plt.colorbar(im)
    plt.xlabel('$f_{red}$')
    plt.ylabel('$\sigma_{A_V} / A_V$')
    plt.axis([0, 1, 0, 3])
    
    # Uncertainty scatter plots, vs self on x-axis

    plt.figure(4)
    plt.close()
    fig = plt.figure(4, figsize=(10,7))
    plt.clf()
    plt.suptitle(brickname)

    plt.subplot(2,2,1)
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
    im = plt.scatter(bf[:,:,2], sigw / (bf[:,:,2]*bf[:,:,1]), c=bf[:,:,1],
                     s=7, linewidth=0, alpha=0.4,
                     vmin=0, vmax=4)
    plt.colorbar(im)
    plt.xlabel('$\sigma_{A_V} / A_V$')
    plt.ylabel('$\Delta(\sigma_{A_V}/A_V) / (\sigma_{A_V}/A_V)$')
    plt.axis([0, 3, 0, 4])
    
    plt.subplot(2,2,4)
    im = plt.scatter(bf[:,:,2],sigw / (bf[:,:,2]*bf[:,:,1]), c=bf[:,:,0],
                     s=7, linewidth=0, alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$\sigma_{A_V} / A_V$')
    plt.ylabel('$\Delta(\sigma_{A_V}/A_V) / (\sigma_{A_V}/A_V)$')
    plt.axis([0, 3, 0, 4])
    
    # Uncertainty scatter plots, vs A_V on x-axis

    plt.figure(5)
    plt.close()
    fig = plt.figure(5, figsize=(10,7))
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
    im = plt.scatter(bf[:,:,1], sigw / (bf[:,:,2]*bf[:,:,1]), c=bf[:,:,0],
                     s=7, linewidth=0, alpha=0.4,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta(\sigma_{A_V}/A_V) / (\sigma_{A_V}/A_V)$')
    plt.axis([0, 4, 0, 4])
    
    # Force display

    plt.draw()

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
