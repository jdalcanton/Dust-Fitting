import math
import numpy as np
import read_brick_data as rbd
import makefakecmd as mfcmd
import os.path as op
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage

def isolate_low_AV(filename = 'ir-sf-b17-v8-st.fits', datadir = '../../Data/',
                   frac=0.05, mrange=[19,21.5],
                   rgbparam=[22.0, 0.72, -0.13, -0.012], nrbins = 5., 
                   d_arcsec=10, 
                   savefile = True, savefiledir = '../Unreddened/', savefilename=''):
    """
    Return a list of color and magnitude for stars in the frac of low
    extinction pixels defined by narrowness of RGB.

    iblue,cnarrow,mnarrow,inarrow,cmap,cstdmap = 
              isolate_low_AV(filename, fraction, mrange, rgbparam)

    Inputs: 

    filename = FITS file of stellar parameters
    fraction = return stars that fall in the fraction of pixels with bluest RGB
    mrange = F160W magnitude range for evaluating color of RGB
    rgbparam = polynomial approximation of color of RGB = [magref,a0,a1,a2]

    Outputs: 

    cnarrow,mnarrow,inarrow = color, magnitude, and indices of stars in the
                        narrowest RGB bins
    ranarrow, decnarrow = ra, dec of stars in the narrowest RGB bins
    rnarrow = major axis radius of each star
    cstdnarrow = local RGB width associated with each star

    """
 
    m1, m2, ra, dec = rbd.read_mag_position_gst(datadir + filename)
    c = np.array(m1 - m2)
    m = np.array(m2)
    r = np.array(get_major_axis(ra, dec))

    # cut out main sequence stars

    crange = [0.3, 2.0]

    indices, rabins, decbins = mfcmd.split_ra_dec(ra, dec, d_arcsec)

    cm,cmlist,cstd,cstdlist = mfcmd.median_rgb_color_map(indices, c, m,
                                                         mrange, crange, rgbparam)
    cstdnodataval = -1
    cmnodataval = -1

    print 'Number of valid areas in color map: ', len(cmlist)

    # get the major axis length for each grid point (don't worry about bin centroiding)

    rarray = np.zeros(cstd.shape)
    for (x,y), value in np.ndenumerate(cm): 
        rarray[x, y] = get_major_axis(rabins[x],decbins[y]).flatten()

    # break list of radii into bins

    rrange = [min(rarray.flatten()), max(rarray.flatten())]
    dr = (rrange[1] - rrange[0]) / nrbins
    rvec = np.array([rrange[0] + i * dr for i in range(int(nrbins) + 1)])
    rgood = [rval for i, rval in enumerate(rarray.flatten()) if cm.flatten()[i] > cmnodataval]
    print 'number of valid r good points', len(rgood)
    
    # find elements with the narrowest color sequence, for each interval in rvec

    mincstd = 0.01   # guarantee a minimum threshold, if cstd = 0 for underpopulated pixels
    cstdthreshvec = 0.0 * np.arange(len(rvec) - 1)
    for j in range(len(rvec)-1):
        cstdtmp = [x for i, x in enumerate(cstdlist) if ((rvec[j] <= rgood[i]) & (rgood[i] < rvec[j+1]))]
        cstdtmp.sort()
        n_cstd_thresh = int(frac * (len(cstdtmp) + 1.))
        if (len(cstdtmp) > 0) & (n_cstd_thresh <= len(cstdtmp)-1):
            cstdthreshvec[j] = cstdtmp[n_cstd_thresh]
            if (cstdthreshvec[j] < mincstd):
                cstdthreshvec[j] = mincstd
        else:
            cstdthreshvec[j] = -1

    print 'rvec:          ', rvec
    print 'cstdthreshvec: ', cstdthreshvec
    #cstdthresh = cstdthreshvec[0]

    #cstdlist.sort()
    #n_cstd_thresh = int(frac * (len(cstdlist) + 1.))
    #cstdthresh = cstdlist[n_cstd_thresh]
    
    ikeep_narrow = []
    cstd_narrow = []
    cm_narrow = []
    
    for j in range(len(rvec)-1):

        for (x,y), value in np.ndenumerate(cm): 
            if ((cstd[x,y] < cstdthreshvec[j]) & (cm[x,y] > cmnodataval) &
                (rvec[j] <= rarray[x,y]) & (rarray[x,y] < rvec[j+1])) :
                ikeep_narrow.extend(indices[x,y])                         # add indices of stars in the bin
                cstd_narrow.extend(cstd[x,y] + np.zeros(len(indices[x,y])))  # tag stars w/ local stddev
                cm_narrow.extend(cm[x,y] + np.zeros(len(indices[x,y])))  # tag stars w/ local median color

    print 'Narrow: Returning ', len(ikeep_narrow),' out of ',len(c),' stars from ',n_cstd_thresh,' bins.'

    cnarrow = np.squeeze(c[ikeep_narrow])
    mnarrow = np.squeeze(m[ikeep_narrow])
    ranarrow = np.squeeze(ra[ikeep_narrow]) 
    decnarrow = np.squeeze(dec[ikeep_narrow])
    rnarrow = np.squeeze(r[ikeep_narrow])
    ikeep_narrow = np.squeeze(ikeep_narrow)
    cstd_narrow = np.squeeze(cstd_narrow)
    cm_narrow = np.squeeze(cm_narrow)

    if savefile:
        
        if savefilename == '':
            savefilename = op.splitext(filename)[0] + '.npz'
        if op.isfile(savefilename):
            # if it does, append some random characters
            print 'Output file ', savefilename, ' exists. Changing filename...'
            savefilenameorig = savefilename
            savefilename = savefiledir + op.splitext(savefilenameorig)[0] + '.' + mfcmd.id_generator(4) + '.npz'
            #print 'New name: ', savefilename
        savefilename = savefiledir + savefilename

        print 'Writing data to ', savefilename
        np.savez(savefilename,
                 cnarrow = cnarrow,
                 mnarrow = mnarrow,
                 ikeep_narrow = ikeep_narrow, 
                 ranarrow = ranarrow,
                 decnarrow = decnarrow,
                 rnarrow = rnarrow,
                 cstd_narrow = cstd_narrow,
                 cm_narrow = cm_narrow)
        #except:
        #    print 'Failed to write ', savefilename
              
    plt.figure(1)
    plt.clf()
    plt.plot(rarray, cstd, ',', color='red')
    plt.plot(rnarrow, cstd_narrow, ',', color='blue')
    plt.axis([min(r), max(r), 0, 0.3])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('RGB width')
    plt.title(filename + ' ' + str(mrange))
    plt.savefig(savefiledir + op.splitext(filename)[0] + '.png', bbox_inches=0)
    
    return c[ikeep_narrow], m[ikeep_narrow], ikeep_narrow, ra[ikeep_narrow], dec[ikeep_narrow], rnarrow, cstd_narrow, cm_narrow

def get_major_axis(ra, dec):

    # conversion from degrees to radians
    radeg  = np.pi / 180.

    # default M31 parameters (see also compleness.py)
    m31ra  = 10.6847929
    m31dec = 41.2690650
    pa = 43.5
    incl = 70.
    m31param = [m31ra, m31dec, pa, incl]

    # useful intermediate quantities
    m31pa  = pa * radeg
    incl   = incl * radeg
    b_over_a = math.cos(incl)
    ecc = math.sqrt(1 - b_over_a)
    
    raoff  = (ra  - m31ra) * math.cos(m31dec * radeg)
    decoff = (dec - m31dec)
    #mindeg = raoff * np.cos(m31pa) - decoff * np.sin(m31pa)
    #majdeg = raoff * np.sin(m31pa) + decoff * np.cos(m31pa)   
    r = np.sqrt((decoff * math.cos(m31pa) + raoff * math.sin(m31pa))**2 +
                (decoff * math.sin(m31pa) - raoff * math.cos(m31pa))**2 / (1.0 - ecc**2))
    
    return r

def make_all_isolate_AV():

    datadir = '../../Data/'
    resultsdir = '../Unreddened/'

    filelist = ['ir-sf-b01-v8-st.fits',
                'ir-sf-b02-v8-st.fits',
                'ir-sf-b04-v8-st.fits',
                'ir-sf-b05-v8-st.fits',
                'ir-sf-b06-v8-st.fits',
                'ir-sf-b08-v8-st.fits',
                'ir-sf-b09-v8-st.fits',
                'ir-sf-b12-v8-st.fits',
                'ir-sf-b14-v8-st.fits',
                'ir-sf-b15-v8-st.fits',
                'ir-sf-b16-v8-st.fits',
                'ir-sf-b17-v8-st.fits',
                'ir-sf-b18-v8-st.fits',
                'ir-sf-b19-v8-st.fits',
                'ir-sf-b21-v8-st.fits',
                'ir-sf-b22-v8-st.fits',
                'ir-sf-b23-v8-st.fits']

    # initialize output vectors
    cnarrow = []
    mnarrow = []
    ranarrow = []
    decnarrow = []
    rnarrow = []
    ikeep_narrow = []
    cstd_narrow = []
    cm_narrow = []

    # set up parameter values
    f = 0.05
    mr = [19.0, 22.0]
    nb = 5

    f = 0.1
    nb = 10

    for filename in filelist:

        savefilename = op.splitext(filename)[0] + '.npz'
        c, m, i, ra, dec, r, cstd, cm =isolate_low_AV(filename = filename, frac=f, mrange=mr, 
                                                      nrbins=nb, 
                                                      savefile=True,
                                                      datadir = datadir,
                                                      savefiledir = resultsdir,
                                                      savefilename = savefilename)        
        cnarrow.extend(c)
        mnarrow.extend(m)
        ranarrow.extend(ra)
        decnarrow.extend(dec)
        rnarrow.extend(r)
        ikeep_narrow.extend(i)
        cstd_narrow.extend(cstd)
        cm_narrow.extend(cm)
        
        print 'Adding ', len(c), ' elements. Total stars: ', len(cnarrow)

    savefilename = resultsdir + 'allbricks.npz'
    np.savez(savefilename,
             cnarrow = np.array(cnarrow),
             mnarrow = np.array(mnarrow),
             ikeep_narrow = np.array(ikeep_narrow), 
             ranarrow = np.array(ranarrow),
             decnarrow = np.array(decnarrow),
             rnarrow = np.array(rnarrow),
             cstd_narrow = np.array(cstd_narrow),
             cm_narrow = np.array(cm_narrow))

    plt.figure(1)
    plt.clf()
    plt.plot(rnarrow, cstd_narrow, ',', color='blue', alpha=0.5)
    plt.axis([min(rnarrow), max(rnarrow), 0, 0.15])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('RGB width')
    plt.title('F160W Range: ' + str(mr))
    plt.savefig(op.splitext(savefilename)[0] + '.rgbwidth.png', bbox_inches=0)

    plt.figure(2)
    plt.clf()
    plt.plot(rnarrow, cm_narrow, ',', color='blue', alpha=0.5)
    plt.axis([min(rnarrow), max(rnarrow), -0.075, 0.075])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('$\Delta$(F110W - F160W)')
    #plt.title('Color Shift Relative to F110W-F160 = '+str(rgbparam[1])+' at F160W = ' + str(rgbparam[0]))
    plt.savefig(op.splitext(savefilename)[0] + '.rgbcolor.png', bbox_inches=0)

    return

def make_all_isolate_AV_for_noise_model():
    """
    identical to make_all_isolate_AV, but uses a much narrower magrange, to avoid biasing
    too much against noisy pixels, and a larger fraction and pixel size, to better populate noise.
    """

    datadir = '../../Data/'
    resultsdir = '../Unreddened/'

    filelist = ['ir-sf-b01-v8-st.fits',
                'ir-sf-b02-v8-st.fits',
                'ir-sf-b04-v8-st.fits',
                'ir-sf-b05-v8-st.fits',
                'ir-sf-b06-v8-st.fits',
                'ir-sf-b08-v8-st.fits',
                'ir-sf-b09-v8-st.fits',
                'ir-sf-b12-v8-st.fits',
                'ir-sf-b14-v8-st.fits',
                'ir-sf-b15-v8-st.fits',
                'ir-sf-b16-v8-st.fits',
                'ir-sf-b17-v8-st.fits',
                'ir-sf-b18-v8-st.fits',
                'ir-sf-b19-v8-st.fits',
                'ir-sf-b21-v8-st.fits',
                'ir-sf-b22-v8-st.fits',
                'ir-sf-b23-v8-st.fits']

    # initialize output vectors
    cnarrow = []
    mnarrow = []
    ranarrow = []
    decnarrow = []
    rnarrow = []
    ikeep_narrow = []
    cstd_narrow = []
    cm_narrow = []

    # set up parameter values
    f = 0.075
    mr = [18.75, 19.75]
    nb = 5
    d_arcsec = 20.

    for filename in filelist:

        savefilename = op.splitext(filename)[0] + '.noise.npz'
        c, m, i, ra, dec, r, cstd, cm = isolate_low_AV(filename = filename, frac=f, mrange=mr, nrbins=nb, 
                                                       d_arcsec = d_arcsec,
                                                       datadir = datadir,
                                                       savefile=True,
                                                       savefiledir = resultsdir,
                                                       savefilename=savefilename)        
        cnarrow.extend(c)
        mnarrow.extend(m)
        ranarrow.extend(ra)
        decnarrow.extend(dec)
        rnarrow.extend(r)
        ikeep_narrow.extend(i)
        cstd_narrow.extend(cstd)
        cm_narrow.extend(cm)
        
        print 'Adding ', len(c), ' elements. Total stars: ', len(cnarrow)

    savefilename = resultsdir + 'allbricks.noise.npz'
    np.savez(savefilename,
             cnarrow = np.array(cnarrow),
             mnarrow = np.array(mnarrow),
             ikeep_narrow = np.array(ikeep_narrow), 
             ranarrow = np.array(ranarrow),
             decnarrow = np.array(decnarrow),
             rnarrow = np.array(rnarrow),
             cstd_narrow = np.array(cstd_narrow),
             cm_narrow = np.array(cm_narrow))

    plt.figure(1)
    plt.clf()
    plt.plot(rnarrow, cstd_narrow, ',', color='blue', alpha=0.5)
    plt.axis([min(rnarrow), max(rnarrow), 0, 0.15])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('RGB width')
    plt.title('F160W Range: ' + str(mr))
    plt.savefig(op.splitext(savefilename)[0] + '.rgbwidth.png', bbox_inches=0)

    plt.figure(2)
    plt.clf()
    plt.plot(rnarrow, cm_narrow, ',', color='blue', alpha=0.5)
    plt.axis([min(rnarrow), max(rnarrow), -0.075, 0.075])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('$\Delta$(F110W - F160W)')
    plt.title('Color Shift Relative to F110W-F160 = '+str(rgbparam[1])+' at F160W = ' + str(rgbparam[0]))
    plt.savefig(op.splitext(savefilename)[0] + '.rgbcolor.png', bbox_inches=0)

    return

def clean_low_AZ_sample(tolerance = 0.005, makenoise=False):

    resultsdir = '../Unreddened/'
    fileroot = resultsdir + 'allbricks'
    if makenoise:
        fileroot = resultsdir + 'allbricks.noise'
    savefilename = fileroot + '.npz'

    dat = np.load(savefilename)
    
    r = dat['rnarrow']
    cstd = dat['cstd_narrow']
    cm = dat['cm_narrow']

    plt.figure(1)
    plt.clf()
    plt.plot(r, cstd, ',', color='black', alpha=0.5)
    plt.axis([0, 1.35, 0, 0.15])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('RGB width')

    # do a first cull on obvious bad regions

    i_bad = np.where(((r>0.58) & (r <1.2) & (cstd > 0.052)) | 
                     ((r > 1.05) & (r < 1.25)) |
                     (r < 0.03) |
                     ((r > 0.39) & (r < 0.44) & (cstd > 0.075)) |
                     ((r > 0.44) & (r < 0.6) & (cstd > 0.055)) |
                     ((r > 1.23) & (cstd > 0.035)) |
                     ((r > 0.32) & (r < 0.39)), 1, 0)
    i_bad_noise = np.where(((r>0.58) & (r <1.2) & (cstd > 0.03)) | 
                           ((r > 1.09) & (r < 1.2) & (cstd > 0.0225)) |
                           (r < 0.03) |
                           ((r > 0.39) & (r < 0.44) & (cstd > 0.0425)) |
                           ((r > 0.44) & (r < 0.6) & (cstd > 0.035)) |
                           ((r > 0.32) & (r < 0.39)), 1, 0)
    if makenoise:
        i_good = np.where(i_bad_noise == 0)
    else:
        i_good = np.where(i_bad == 0)
    plt.plot(r[i_good], cstd[i_good], ',', color='green', alpha=0.5)

    # fit polynomials to good points

    npoly = 8                #...... RGB width
    param = np.polyfit(r[i_good], cstd[i_good], npoly)
    print 'Polynomial: ', param 
    p = np.poly1d(param)

    rp = np.linspace(0, max(r), 100)

    npoly = 6                #..... RGB color
    param_cm = np.polyfit(r[i_good], cm[i_good], npoly)
    print 'Polynomial: ', param_cm

    p_cm = np.poly1d(param_cm)

    # keep everything within a given tolerance of the polynomial

    cm_tolfac = 2.5

    #ikeep = np.where((cstd - p(r) < tolerance) & (r > 0.03))
    ikeep = np.where(((cstd - p(r))    < tolerance) & 
                     ((cm   - p_cm(r)) < cm_tolfac*tolerance) & 
                     (r > 0.03))
    plt.plot(r[ikeep], cstd[ikeep], ',', color='magenta')
    plt.plot(rp, p(rp), color='blue', linewidth=3)

    Acol_AV = 0.33669 - 0.20443
    print 'Tolerance in units of A_V: ', tolerance / Acol_AV
    plt.title('Tolerance = '+ ("%g" % round(tolerance,4)) + ' ($\Delta A_V$ = ' + ("%g" % round(tolerance / Acol_AV,3)) + ')')

    plotfilename = fileroot + '.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)

    # make other diagnostic plots

    plt.figure(2)
    plt.clf()
    plt.plot(r, cm, ',', color='black', alpha=0.5)
    plt.axis([0, 1.35, -0.07, 0.07])
    plt.plot(r[ikeep], cm[ikeep], ',', color='magenta', alpha=0.5)
    plt.plot(rp, p_cm(rp), color='blue', linewidth=3)
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('$\Delta(F110W - F160W)$')
    plt.title('Tolerance = '+ ("%g" % round(cm_tolfac*tolerance,5)) + ' ($\Delta A_V$ = ' + ("%g" % round(cm_tolfac*tolerance / Acol_AV,3)) + ')')
    plotfilename = fileroot + '.meancolor.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(3)
    plt.clf()
    plt.plot(cstd, cm, ',', color='black')
    plt.plot(cstd[ikeep], cm[ikeep], ',', color='magenta')
    #im = plt.scatter(cstd[ikeep], cm[ikeep], c=r[ikeep], s=3, linewidth=0, vmin=0, vmax=1.35)
    #plt.colorbar(im)
    plt.axis([0, 0.15, -0.07, 0.07])
    plt.xlabel('RGB Width')
    plt.ylabel('$\Delta(F110W - F160W)$')
    plotfilename = fileroot + '.meancolorvswidth.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)


    # save clean data to a new file

    newsavefilename = fileroot + '.clean.npz'
    print 'Saving clean data to ', newsavefilename
    np.savez(newsavefilename,
             cnarrow = dat['cnarrow'][ikeep],
             mnarrow = dat['mnarrow'][ikeep],
             ikeep_narrow = dat['ikeep_narrow'][ikeep], 
             ranarrow = dat['ranarrow'][ikeep],
             decnarrow = dat['decnarrow'][ikeep],
             rnarrow = dat['rnarrow'][ikeep],
             cstd_narrow = dat['cstd_narrow'][ikeep],
             cm_narrow = dat['cm_narrow'][ikeep],
             polyparam_cstd = param, polyparam_cm = param_cm)

    return

def read_clean(readnoise = False):

    resultsdir = '../Unreddened/'
    savefilename = resultsdir + 'allbricks.clean.npz'
    if readnoise:
        savefilename = resultsdir + 'allbricks.noise.clean.npz'

    dat = np.load(savefilename)

    c = dat['cnarrow']
    m = dat['mnarrow']
    #ikeep = dat['ikeep_narrow']
    ra = dat['ranarrow']
    dec = dat['decnarrow']
    r = dat['rnarrow']
    cstd = dat['cstd_narrow']
    cm = dat['cm_narrow']

    return c, m, ra, dec, r, cstd, cm

def make_low_AV_cmd(c, m, 
                    mrange = [18.2, 25.], 
                    crange = [0.3,1.3], 
                    deltapix = [0.015,0.15], 
                    masksig=[2.5,3.0]):

    nbins = [int((mrange[1] - mrange[0]) / deltapix[1]),
             int((crange[1] - crange[0]) / deltapix[0])]
    cmd, mboundary, cboundary = np.histogram2d(m, c,
                                               range=[np.sort(mrange), crange], bins=nbins)

    deltapix =  [cboundary[1]-cboundary[0],mboundary[1]-mboundary[0]]
    extent = [cboundary[0], cboundary[-1],
              mboundary[-1], mboundary[0]]

    # Generate mask and measure mean and sigma
    foregroundmask, meancol, sigcol = mfcmd.clean_fg_cmd(cmd, masksig, 
                                                         niter=4, showplot=0)

    return cmd, foregroundmask, meancol, sigcol, cboundary, mboundary, extent


def make_radial_low_AV_cmds(nrgbstars = 2500, nsubstep=3., 
                            mrange = [18.2, 25.],
                            crange = [0.3, 2.5],
                            deltapixorig = [0.015,0.25],
                            mnormalizerange = [19,21.5], 
                            maglimoff = 0.25,
                            nsig_blue_color_cut = 2.0, blue_color_cut_mask_only=False,
                            usemask=True, masksig=[2.5,3.0],
                            makenoise=False, noisemasksig=[4.5,4.5],
                            useq=False, reference_color=1.0,
                            restricted_r_range=''):

    # Define reddening parameters

    Amag_AV = 0.20443
    Acol_AV = 0.33669 - 0.20443
    t = np.arctan(-Amag_AV / Acol_AV)
    reference_color = 1.0

    # read in cleaned, low reddening data

    c, m, ra, dec, r, cstd, cm = read_clean()

    # convert to reddening-free mag, if requested
    if useq:
        #m = m + (c-reference_color)*np.sin(t)/np.cos(t)
        m = m + (c-reference_color)*(-Amag_AV / Acol_AV)

    # sort according to radius

    isort = np.argsort(r)
    c = c[isort]
    m = m[isort]
    r = r[isort]
    ra = ra[isort]
    dec = dec[isort]
    cstd = cstd[isort]
    cm = cm[isort]

    # break into radial bins based on number of stars in normalization range. 
    # Normal brick wide low-AV has ~10K - 20K total, but many of those are in RC.

    istar = np.arange(len(r))
    irgb = np.where((mnormalizerange[0] < m) & (m < mnormalizerange[1]))[0]
    print 'Starting with ', len(r), ' stars total.'
    print 'Number of upper RGB Stars: ', len(irgb)

    nstep = int(nrgbstars / nsubstep)

    nrgblo = np.arange(0, len(irgb)-nstep, nstep)
    nrgbhi = nrgblo + int(nrgbstars)
    nrgbhi = np.where(nrgbhi < len(irgb)-1, nrgbhi, len(irgb)-1) # tidy up ends

    nrlo = istar[irgb[nrgblo]]
    nrhi = istar[irgb[nrgbhi]]

    nrhi = np.where(nrhi < len(istar)-1, nrhi, len(istar)-1)     # tidy up ends again
    nrbins = len(nrhi)

    #print 'nrgblo: ', nrgblo
    #print 'nrgbhi: ', nrgbhi
    #print 'nrlo: ', nrlo
    #print 'nrhi: ', nrhi

    print 'Splitting into ', len(nrhi),' radial bins with ',nrgbstars,' in each upper RGB.'

    # merge adjacent bins, if they're too close together....

    min_dr = 0.01
    nr_orig = len(nrlo)
    meanr = np.array([np.average(r[nrlo[i]:nrhi[i]]) for i in range(len(nrlo))])
    dmeanr = np.array([meanr[i+1] - meanr[i] for i in range(len(nrlo)-1)])
    bad_dr = np.where(dmeanr < min_dr)[0]

    while (len(bad_dr) > 0): # are there any gaps that are too small?

        if (bad_dr[0] + 1 < len(nrlo) - 1):     # and is the gap not in the last, unmergeable bin?

            # if so, merge the first instance with the adjacent cell
            print 'Merging ', bad_dr[0], ' with ', bad_dr[0] + 1, '.  ', len(bad_dr)-1, ' small gaps remaining.'
            nrlo = np.delete(nrlo, bad_dr[0] + 1)   
            nrhi = np.delete(nrhi, bad_dr[0])

            meanr = np.array([np.average(r[nrlo[i]:nrhi[i]]) for i in range(len(nrlo))])
            dmeanr = np.array([meanr[i+1] - meanr[i] for i in range(len(nrlo)-1)])
            bad_dr = np.where(dmeanr < min_dr)[0]

    print 'Reduced number of radial bins from ', nr_orig, ' to ', len(nrlo)

    nrbins = len(nrhi)

    # run once to get shape of CMD
    cmd, fgmask, meancol, sigcol, cboundary, mboundary, extent = \
        make_low_AV_cmd(c, m, 
                        mrange = mrange,
                        crange = crange,
                        deltapix = deltapixorig,
                        masksig = masksig)
    
    # setup interpolation to convert pixel values of meancol into numerical values
    mcen = (mboundary[0:-1] + mboundary[1:]) / 2.0
    ccen = (cboundary[0:-1] + cboundary[1:]) / 2.0
    cinterp  = interp1d(np.arange(len(ccen)), ccen)
    deltapix =  np.array([cboundary[1]-cboundary[0],mboundary[1]-mboundary[0]])

    # initialize storage arrays
    cmd_array = np.zeros([cmd.shape[0], cmd.shape[1], nrbins])
    mask_array = np.zeros([cmd.shape[0], cmd.shape[1], nrbins])
    meancol_array = np.zeros([meancol.shape[0], nrbins])
    sigcol_array = np.zeros([sigcol.shape[0], nrbins])
    meanr_array = np.zeros(nrbins)
    rrange_array = np.zeros([2,nrbins])
    n_array = np.zeros(nrbins)
    maglim_array = np.zeros([2,nrbins])

    # initialize magnitude limit polynomials
    completenessdir = '../../Completeness/'
    m110file = 'completeness_ra_dec.st.F110W.npz'
    m160file = 'completeness_ra_dec.st.F160W.npz'
        
    m110dat = np.load(completenessdir + m110file)
    m110polyparam = m110dat['param']
    m160dat = np.load(completenessdir + m160file)
    m160polyparam = m160dat['param']
        
    p110 = np.poly1d(m110polyparam)
    p160 = np.poly1d(m160polyparam)

    # initialize plot
    plt.figure(1)
    plt.clf()

    #
    if usemask:
        print 'Masking out noise in stddeviation range: ', masksig

    # Loop through bins of radius
    for i in range(len(nrlo)):
        
        cmd, fgmask, meancol, sigcol, cboundary, mboundary, extent = \
            make_low_AV_cmd(c[nrlo[i]:nrhi[i]],
                            m[nrlo[i]:nrhi[i]],
                            mrange = mrange,
                            crange = crange,
                            deltapix = deltapixorig,
                            masksig = masksig)

        # mask out noise, if requested
        if usemask:
            cmd = cmd * fgmask

        # normalize CMD to a constant # of stars for magnitudes in mnormalizerange
        irgb = np.where((mnormalizerange[0] < mboundary) & (mboundary < mnormalizerange[1]))[0]
        rgbrange = [min(irgb), max(irgb)]
        norm = np.sum(cmd[rgbrange[0]:rgbrange[1],:])
        nstars = np.sum(cmd)
        cmd = cmd.astype(float) / float(norm)

        # copy results to appropriate array
        cmd_array[:,:,i] = cmd
        meancol_array[:,i] = cinterp(meancol)
        sigcol_array[:,i] = sigcol * deltapix[0]
        meanr_array[i] = np.average(r[nrlo[i]:nrhi[i]])
        rrange_array[:,i] = [r[nrlo[i]], r[nrhi[i]]]
        n_array[i] = len(r[nrlo[i]:nrhi[i]])

        # calculate corresponding magnitude limits

        maglim110 = p110(meanr_array[i])
        maglim160 = p160(meanr_array[i])
        maglim_array[:,i] = np.array([maglim110, maglim160])

        m110range = [16.0, maglim110 - maglimoff]
        m160range = [18.4, maglim160 - maglimoff]

        # generate mask
        bluecolorlim = cboundary[np.maximum(np.rint(meancol - nsig_blue_color_cut * 
                                                    sigcol).astype(int), 0)]
        if blue_color_cut_mask_only: 
            color_mag_datamask = mfcmd.make_data_mask(cmd, cboundary, 
                                                      mboundary, [0.0, 100.], [0., 100.],
                                                      bluecolorlim)
        else:
            color_mag_datamask = mfcmd.make_data_mask(cmd, cboundary, 
                                                      mboundary, m110range, m160range, 
                                                      bluecolorlim)
        if useq: 

            if blue_color_cut_mask_only: 
                color_mag_datamask = mfcmd.make_data_mask(cmd, cboundary, 
                                                          mboundary, [0.0, 100.], [0., 100.],
                                                          bluecolorlim, useq=[reference_color])
            else:
                color_mag_datamask = mfcmd.make_data_mask(cmd, cboundary, 
                                                          mboundary, m110range, m160range, 
                                                          bluecolorlim, useq=[reference_color])

        mask_array[:,:,i] = color_mag_datamask

        print 'Bin: ', i, '  R: ', ("%g" % round(meanr_array[i],3)), 'maglim', ("%g" % round(maglim110,2)), ("%g" % round(maglim160,2)),' NStars: ',nstars, ' NRGB: ',norm

        #plt.imshow(cmd,  extent=extent, aspect='auto', interpolation='nearest', vmin=0, vmax = 0.1)
        plt.imshow(0.01*color_mag_datamask + cmd,  extent=extent, aspect='auto', interpolation='nearest', vmin=0, vmax = 0.1)
        plt.xlabel('F110W - F160W')
        plt.ylabel('F160W')
        plt.title('Major Axis: '+ ("%g" % np.round(meanr_array[i],3)) + 
                  '  F160W 50% Completeness: ' + ("%g" % np.round(maglim160,2)))
        plt.draw()

    # make a noise model, if requested

    if makenoise:

        noise_smooth_mag = np.array([0.2, 0.5])    # Note: filter is with opposite polarity
        noise_smooth = np.rint(noise_smooth_mag / deltapix)
        #noise_smooth = [3, 10]  # mag, color, in pixels
        print 'Building Noise Model...Smoothing with: ', noise_smooth

        c_n, m_n, ra_n, dec_n, r_n, cstd_n, cm_n = read_clean(readnoise = True)
        # convert to reddening-free mag, if requested
        if useq:
            m_n = m_n + (c_n-reference_color)*(-Amag_AV / Acol_AV)

        isort = np.argsort(r_n)
        c_n = c_n[isort]
        m_n = m_n[isort]
        r_n = r_n[isort]
        ra_n = ra_n[isort]
        dec_n = dec_n[isort]
        cstd_n = cstd_n[isort]
        cm_n = cm_n[isort]

        r_n_range = [min(r_n), max(r_n)]

        # initialize array to save noise model

        noise_array = np.zeros([cmd.shape[0], cmd.shape[1], nrbins])
        noisefrac_array = np.zeros(nrbins)

        # set up function to quickly find indices of stars within given r-range
        i_rinterp  = interp1d(r_n, np.arange(len(r_n)))

        for i in range(len(nrlo)):

            # interpolate to find range of points, enforcing boundaries of interpolation
            r_range = rrange_array[:,i]
            i_lo = np.ceil(i_rinterp(np.maximum(r_range[0], r_n_range[0])))
            i_hi = np.floor(i_rinterp(np.minimum(r_range[1], r_n_range[1])))
            #i_rrange = i_rinterp(r_range)
            #i_lo = np.ceil(i_rrange[0])
            #i_hi = np.floor(i_rrange[1])

            cmd_n, noisemask, meancol_n, sigcol_n, cboundary_n, mboundary_n, extent_n =   \
                make_low_AV_cmd(c_n[i_lo:i_hi],
                                m_n[i_lo:i_hi],
                                mrange = mrange,
                                crange = crange,
                                deltapix = deltapixorig,
                                masksig = noisemasksig)

            nmag = noisemask.shape[0]
            
            # invert sense of mask, and smooth
            noisemask = abs(noisemask - 1)

            # trim blue side
            bluecutmask = 1.0 + np.zeros(noisemask.shape)
            bluecutmask = np.array([np.where((cboundary[:-1] > bluecolorlim[j]), 
                                             1.0, 0.0) for j in range(nmag)])
            noisemask = noisemask * bluecutmask


            noise_model_orig = cmd_n * noisemask
            noise_model = ndimage.filters.uniform_filter(noise_model_orig,
                                                         size=[noise_smooth[1],noise_smooth[0]])

            # calculate fraction in noise model
            color_mag_datamask = mask_array[:,:,i]
            nfg = (cmd_n * color_mag_datamask).sum()
            nnoise = (noise_model * color_mag_datamask).sum()
            frac_noise = nnoise / nfg
            print 'ilo: ', i_lo, ' ihi: ', i_hi, ' Noise fraction: ', frac_noise

            # do a rough normalization (will have to redo after radial interpretation and data mask)
            noise_model = noise_model / float(nnoise)

            noise_array[:,:,i] = noise_model
            noisefrac_array[i] = frac_noise

            plt.imshow(0.005*color_mag_datamask + noise_model,  extent=extent, aspect='auto', 
                       interpolation='nearest', vmin=0, vmax=0.1)
            plt.xlabel('F110W - F160W')
            plt.ylabel('F160W')
            plt.title('Major Axis: '+ ("%g" % np.round(meanr_array[i],3)) + 
                      '  F160W 50% Completeness: ' + ("%g" % np.round(maglim160,2)))
            plt.draw()

    if restricted_r_range != '':

        print 'Trimming down to restricted radial range: ', restricted_r_range
        # fix limits of restricted_r_range
        restricted_r_range = [np.maximum(restricted_r_range[0],np.min(meanr_array)),
                              np.minimum(restricted_r_range[1],np.max(meanr_array))]
        i_rinterp  = interp1d(meanr_array, np.arange(len(meanr_array)))
        i = i_rinterp(restricted_r_range)
        ilo = np.maximum(int(np.floor(i[0])), 0)
        ihi = np.minimum(int(np.ceil(i[1]) + 1), len(meanr_array))
        print 'Grabbing index range: ', [ilo, ihi]
        return cmd_array[:,:,ilo:ihi], mask_array[:,:,ilo:ihi], \
            noise_array[:,:,ilo:ihi], noisefrac_array[ilo:ihi], \
            meanr_array[ilo:ihi], rrange_array[:,ilo:ihi], \
            meancol_array[:,ilo:ihi], sigcol_array[:,ilo:ihi], \
            n_array[ilo:ihi], maglim_array[:,ilo:ihi], mboundary, cboundary
    
    else:

        return cmd_array, mask_array, noise_array, noisefrac_array, \
            meanr_array, rrange_array, meancol_array, sigcol_array, \
            n_array, maglim_array, mboundary, cboundary


def get_radius_range_of_all_bricks():
    """
    Generate an array containing brick file names, and range of major axis length radii in the brick.
    Returns the array and saves it to a file for reference.
    """
        
    datadir = '../../Data/'
    resultsdir = '../Unreddened/'
    radiusfile = 'radius_range_of_bricks.npz'

    filelist = ['ir-sf-b01-v8-st.fits',
                'ir-sf-b02-v8-st.fits',
                'ir-sf-b04-v8-st.fits',
                'ir-sf-b05-v8-st.fits',
                'ir-sf-b06-v8-st.fits',
                'ir-sf-b08-v8-st.fits',
                'ir-sf-b09-v8-st.fits',
                'ir-sf-b12-v8-st.fits',
                'ir-sf-b14-v8-st.fits',
                'ir-sf-b15-v8-st.fits',
                'ir-sf-b16-v8-st.fits',
                'ir-sf-b17-v8-st.fits',
                'ir-sf-b18-v8-st.fits',
                'ir-sf-b19-v8-st.fits',
                'ir-sf-b21-v8-st.fits',
                'ir-sf-b22-v8-st.fits',
                'ir-sf-b23-v8-st.fits']

    r_range_brick_array = np.empty((len(filelist), 7), dtype=object)
    r_range_brick_array[:,0] = filelist
        
    for i, filename in enumerate(filelist): 
        m1, m2, ra, dec = rbd.read_mag_position_gst(datadir + filename)
        r = np.array(get_major_axis(ra, dec))
        r_range_brick_array[i,1:] = [min(r), max(r), min(ra), max(ra), min(dec), max(dec)]

    print r_range_brick_array

    print 'Writing r_range_brick_array to ', radiusfile
    np.savez(radiusfile, 
             r_range_brick_array=r_range_brick_array)

    print 'Global range of ra:  ', min(r_range_brick_array[:,3].flatten()), max(r_range_brick_array[:,4].flatten())
    print 'Global range of dec: ', min(r_range_brick_array[:,5].flatten()), max(r_range_brick_array[:,6].flatten())

    return r_range_brick_array


def brick_files_containing_radii(r, r_range_brick_array):
    """
    Uses the output of "r_range_brick_array = get_radius_range_of_all_bricks()" or
        dat = np.load('radius_range_of_bricks.npz')
        r_range_brick_array = dat['r_range_brick_array']

    Returns a filelist of the brick data files containing stars at that radius
    """
    rr = np.array(r_range_brick_array[:,1:3], dtype=float)
    bricklist = r_range_brick_array[:,0:1].flatten()

    #try:
    #    # if r is an array or list
    #    len(r)
    return np.array([[filename for i, filename in enumerate(bricklist) if ((rr[i,0] <= r[j]) & 
                                                                           (r[j] <= rr[i,1]))] 
                     for j in range(len(r))])
    #except:
    #    return np.array([filename for i, filename in enumerate(bricklist) if ((rr[i,0] <= r) & 
    #                                                                          (r <= rr[i,1]))])




    


        






