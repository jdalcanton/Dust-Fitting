import math
import numpy as np
import read_brick_data as rbd
import makefakecmd as mfcmd
import fit_disk as fitdisk
import os.path as op
import matplotlib.pyplot as plt
import pyfits as pyfits
import pywcs as pywcs
from scipy.interpolate import interp1d
from scipy import ndimage

def isolate_low_AV(filename = 'ir-sf-b17-v8-st.fits', datadir = '../../Data/',
                   frac=0.05, mrange=[19,21.5], mrange_nstar=[18.5,21.0],
                   rgbparam=[22.0, 0.72, -0.13, -0.012], nrbins = 5., 
                   d_arcsec=10.0, 
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

    nstarnarrow = number of upper RGB stars per sqr-arcsec

    """
 
    m1, m2, ra, dec = rbd.read_mag_position_gst(datadir + filename)
    c = np.array(m1 - m2)
    m = np.array(m2)
    r = np.array(get_major_axis(ra, dec))

    # cut out main sequence stars

    crange = [0.3, 2.0]

    indices, rabins, decbins = mfcmd.split_ra_dec(ra, dec, d_arcsec)

    # position of pixel centers
    racenvec  = (rabins[:-1]  +  rabins[1:]) / 2.0
    deccenvec = (decbins[:-1] + decbins[1:]) / 2.0

    # calculate radii at grid points
    decarray, raarray = np.meshgrid(deccenvec, racenvec)
    rarray = get_major_axis(raarray, decarray)

    # calculate RGB properties at grid points, first for full magnitude range,
    # then for the magnitude range used to count the number of stars.
    cm,cmlist,cstd,cstdlist,cmean,cmeanlist,junk1,junk2 = \
        mfcmd.median_rgb_color_map(indices, c, m,
                                   mrange=mrange, crange=crange, 
                                   rgbparam=rgbparam)
    junk1,junk2,junk3,junk4,junk5,junk6,nstar,nstarlist = \
        mfcmd.median_rgb_color_map(indices, c, m,
                                   mrange=mrange_nstar, crange=crange, 
                                   rgbparam=rgbparam)

    # initialize dummy values for regions with no stars.
    cstdnodataval = -1
    cmnodataval = -1
    cmeannodataval = -1
    nstarnodataval = -1

    print 'Number of valid areas in color map: ', len(cmlist)

    # get the major axis length for each grid point (don't worry about bin centroiding)

    rarraytmp = np.zeros(cstd.shape)
    for (x,y), value in np.ndenumerate(cm): 
        rarraytmp[x, y] = get_major_axis(rabins[x],decbins[y]).flatten()

    # break list of radii into bins

    rrange = [min(rarraytmp.flatten()), max(rarraytmp.flatten())]
    dr = (rrange[1] - rrange[0]) / nrbins
    rvec = np.array([rrange[0] + i * dr for i in range(int(nrbins) + 1)])
    rgood = [rval for i, rval in enumerate(rarraytmp.flatten()) 
             if cm.flatten()[i] > cmnodataval]
    print 'number of valid r good points', len(rgood)
    
    # find elements with the narrowest color sequence, for each interval in rvec

    mincstd = 0.01   # guarantee a minimum threshold, if cstd = 0 for underpopulated pixels
    cstdthreshvec = 0.0 * np.arange(len(rvec) - 1)
    for j in range(len(rvec)-1):
        cstdtmp = [x for i, x in enumerate(cstdlist) 
                   if ((rvec[j] <= rgood[i]) & (rgood[i] < rvec[j+1]))]
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
    cmean_narrow = []
    nstar_narrow = []
    
    for j in range(len(rvec)-1):

        for (x,y), value in np.ndenumerate(cm): 
            if ((cstd[x,y] < cstdthreshvec[j]) & (cm[x,y] > cmnodataval) &
                (rvec[j] <= rarraytmp[x,y]) & (rarraytmp[x,y] < rvec[j+1])) :

                ikeep_narrow.extend(indices[x,y])                         # add indices of stars in the bin
                cstd_narrow.extend(cstd[x,y] + np.zeros(len(indices[x,y])))  # tag stars w/ local stddev
                cm_narrow.extend(cm[x,y] + np.zeros(len(indices[x,y])))  # tag stars w/ local median color
                cmean_narrow.extend(cmean[x,y] + np.zeros(len(indices[x,y])))  # tag stars w/ local mean color
                nstar_narrow.extend(nstar[x,y] + np.zeros(len(indices[x,y])))  # tag stars w/ local numberdensity

    print 'Narrow: Returning ', len(ikeep_narrow),' out of ',len(c),' stars from ',n_cstd_thresh,' bins.'

    cnarrow = np.squeeze(c[ikeep_narrow])
    mnarrow = np.squeeze(m[ikeep_narrow])
    ranarrow = np.squeeze(ra[ikeep_narrow]) 
    decnarrow = np.squeeze(dec[ikeep_narrow])
    rnarrow = np.squeeze(r[ikeep_narrow])
    ikeep_narrow = np.squeeze(ikeep_narrow)
    cstd_narrow = np.squeeze(cstd_narrow)
    cm_narrow = np.squeeze(cm_narrow)
    cmean_narrow = np.squeeze(cmean_narrow)
    nstar_narrow = np.squeeze(nstar_narrow)

    # normalize number of stars to a surface density
    nstar        = nstar / (d_arcsec**2)
    nstar_narrow = nstar_narrow / (d_arcsec**2)

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
                 cnarrow = np.array(cnarrow),
                 mnarrow = np.array(mnarrow),
                 ikeep_narrow = np.array(ikeep_narrow), 
                 ranarrow = np.array(ranarrow),
                 decnarrow = np.array(decnarrow),
                 rnarrow = np.array(rnarrow),
                 cstd_narrow = np.array(cstd_narrow),
                 cm_narrow = np.array(cm_narrow),
                 cmean_narrow = np.array(cmean_narrow),
                 nstar_narrow = np.array(nstar_narrow),
                 cmarray = np.array(cm),
                 cmeanarray = np.array(cmean),
                 cstdarray = np.array(cstd),
                 nstararray = np.array(nstar),
                 rabins = np.array(rabins),
                 decbins = np.array(decbins),
                 rarray = np.array(rarray),
                 decarray = np.array(decarray),
                 raarray = np.array(raarray))
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
    
    return c[ikeep_narrow], m[ikeep_narrow], ikeep_narrow, ra[ikeep_narrow], dec[ikeep_narrow], rnarrow, cstd_narrow, cm_narrow, cmean_narrow, nstar_narrow, cm, cmean, cstd, nstar, rabins, decbins, rarray, decarray, raarray


def get_major_axis(ra, dec, m31ra=10.6847929, m31dec = 41.2690650, pa=38.5, incl=74., return_theta=False):

    # conversion from degrees to radians
    radeg  = np.pi / 180.

    # default M31 parameters (see also compleness.py)
    #m31ra  = 10.6847929
    #m31dec = 41.2690650

    # parameters below chosen to match isophotes in 3.6 micron images.
    # See radius_contours_on_irac1_image_log_contours.png
    # made w/ logarithmic contours of smoothed irac1 image (blue), and red overlay
    # contours from radius with m31param = [10.6847929, 41.2690650, 38.5, 74.]
    # radial contour levels at 0.125, 0.25, 0.375, etc out to 1.75
    #pa = 38.5
    #incl = 74.
    m31param = [m31ra, m31dec, pa, incl]

    # useful intermediate quantities
    m31pa  = pa * radeg
    incl   = incl * radeg
    b_over_a = math.cos(incl)
    ecc = math.sqrt(1. - (b_over_a)**2.)
    #print 'B_over_A: ', b_over_a
    #print 'Incl: ', incl
    
    raoff  = (ra  - m31ra) * math.cos(m31dec * radeg)
    decoff = (dec - m31dec)
    #mindeg = raoff * np.cos(m31pa) - decoff * np.sin(m31pa)
    #majdeg = raoff * np.sin(m31pa) + decoff * np.cos(m31pa)   
    r = np.sqrt((decoff * math.cos(m31pa) + raoff * math.sin(m31pa))**2 +
                (decoff * math.sin(m31pa) - raoff * math.cos(m31pa))**2 / (1.0 - ecc**2))

    if (return_theta == True):

        y = decoff * math.cos(m31pa) + raoff * math.sin(m31pa)
        x = decoff * math.sin(m31pa) - raoff * math.cos(m31pa)

        theta = np.arccos(np.sqrt(1.0 - (y / r)**2))

        i_negx = np.where(x < 0)
        theta[i_negx] = (np.pi ) - theta[i_negx]

        return r, theta

    else:

        return r


def test_major_axis(filename = '../Results/FourthRun/ir-sf-b17-v8-st.npz', hz_over_hr=0.2):

    d = np.load(filename)

    ra_g = np.array(d['ra_global'])
    dec_g = np.array(d['dec_global'])
    ra_b = np.array(d['ra_bins'])
    dec_b = np.array(d['dec_bins'])

    dd_g, rr_g = np.meshgrid(dec_g, ra_g)
    rangevec_g = [max(ra_g), min(ra_g), min(dec_g), max(dec_g)]
    dd_b, rr_b = np.meshgrid(dec_b, ra_b)
    rangevec_b = [max(ra_b), min(ra_b), min(dec_b), max(dec_b)]
    r_brickcorner = [min(ra_b), min(ra_b), max(ra_b), max(ra_b), min(ra_b)]
    d_brickcorner = [min(dec_b), max(dec_b), max(dec_b), min(dec_b), min(dec_b)]

    r_g, t_g = get_major_axis(rr_g, dd_g, return_theta=True)
    r_b, t_b = get_major_axis(rr_b, dd_b, return_theta=True)

    incl = 74.
    f_g = 0.5 * (1 - np.tan(incl * np.pi / 180.) * np.cos(t_g) * hz_over_hr)

    print rr_g.shape
    print rr_g[:,::-1].shape
    print ra_g.shape
    print dec_g.shape

    
    #----------------------
    plt.figure(1)
    plt.clf()

    plt.subplot(2,2,1)
    im = plt.imshow(rr_g[:,::-1], origin='lower', extent=rangevec_g, aspect='auto')
    plt.colorbar(im)
    plt.title('RA')
    plt.suptitle(filename)

    plt.subplot(2,2,2)
    im = plt.imshow(dd_g[:,::-1], origin='lower', extent=rangevec_g, aspect='auto')
    plt.colorbar(im)
    plt.title('Dec')

    plt.subplot(2,2,3)
    #im = plt.imshow(r_g[:,::-1], origin='lower', extent=rangevec_g, aspect='auto')
    im = plt.imshow(r_g[:,::-1], origin='lower', extent=rangevec_g, aspect='auto')
    plt.axis(rangevec_g)
    plt.colorbar(im)
    plt.title('Major Axis')
    im = plt.plot(r_brickcorner, d_brickcorner, color='black')
    
    plt.subplot(2,2,4)
    #im = plt.imshow(np.cos(t_g[:,::-1]), origin='lower', extent=rangevec_g, aspect='auto')
    #plt.title(r'cos($\theta$)')
    im = plt.imshow(f_g[:,::-1], origin='lower', extent=rangevec_g, aspect='auto', vmin=0, vmax=1)
    plt.title(r'$f_{red}$')
    plt.axis(rangevec_g)
    plt.colorbar(im)
    im = plt.plot(r_brickcorner, d_brickcorner, color='black')
    
    #----------------------
    plt.figure(2)
    plt.clf()

    plt.subplot(3,1,1)
    im = plt.imshow(rr_b[:,::-1], origin='lower', extent=rangevec_b, aspect='auto')
    plt.colorbar(im)
    plt.title('RA')
    plt.suptitle(filename)

    plt.subplot(3,1,2)
    im = plt.imshow(dd_b[:,::-1], origin='lower', extent=rangevec_b, aspect='auto')
    plt.colorbar(im)
    plt.title('Dec')

    plt.subplot(3,1,3)
    im = plt.imshow(r_b[:,::-1], origin='lower', extent=rangevec_b, aspect='auto')
    plt.colorbar(im)
    plt.title('Major Axis')

    return

def make_all_isolate_AV(plot_results = False):

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
    cmean_narrow = []
    nstar_narrow = []
    cmgridval = [] 
    cmeangridval = [] 
    cstdgridval = [] 
    nstargridval = [] 
    rgridval = [] 
    decgridval = [] 
    ragridval = []

    # set up parameter values
    mr = [19.0, 22.0]
    mr_nstar = [18.5, 21.0]
    f = 0.2
    nb = 20
    d_arcsec = 30.0

    for filename in filelist:

        savefilename = op.splitext(filename)[0] + '.npz'
        c, m, i, ra, dec, r, cstd, cm, cmean, nstar, \
            cmarray, cmeanarray, cstdarray, nstararray, rabins, \
            decbins, rarray, decarray, raarray = \
            isolate_low_AV(filename = filename, frac=f, 
                           mrange=mr, mrange_nstar = mr_nstar,
                           d_arcsec = d_arcsec,
                           nrbins=nb, 
                           savefile=True,
                           datadir = datadir,
                           savefiledir = resultsdir,
                           savefilename = savefilename)        
        ikeep = np.where(nstararray > 0)
        cnarrow.extend(c)
        mnarrow.extend(m)
        ranarrow.extend(ra)
        decnarrow.extend(dec)
        rnarrow.extend(r)
        ikeep_narrow.extend(i)
        cstd_narrow.extend(cstd)
        cm_narrow.extend(cm)
        cmean_narrow.extend(cmean)
        nstar_narrow.extend(nstar)
        cmgridval.extend(cmarray[ikeep])
        cmeangridval.extend(cmeanarray[ikeep])
        cstdgridval.extend(cstdarray[ikeep])
        nstargridval.extend(nstararray[ikeep])
        rgridval.extend(rarray[ikeep])
        decgridval.extend(decarray[ikeep])
        ragridval.extend(raarray[ikeep])
        
        print 'Adding ', len(c), ' elements. Total stars: ', len(cnarrow)
        print 'Adding ', len(cmarray[ikeep]), ' grid points. Total grid points: ', len(cmgridval)

    savefilename = resultsdir + 'allbricks.npz'
    np.savez(savefilename,
             cnarrow = np.array(cnarrow),
             mnarrow = np.array(mnarrow),
             ikeep_narrow = np.array(ikeep_narrow), 
             ranarrow = np.array(ranarrow),
             decnarrow = np.array(decnarrow),
             rnarrow = np.array(rnarrow),
             cstd_narrow = np.array(cstd_narrow),
             cm_narrow = np.array(cm_narrow),
             cmean_narrow = np.array(cmean_narrow),
             nstar_narrow = np.array(nstar_narrow),
             cmgridval = np.array(cmgridval), 
             cmeangridval = np.array(cmeangridval), 
             cstdgridval = np.array(cstdgridval), 
             nstargridval = np.array(nstargridval), 
             rgridval = np.array(rgridval), 
             decgridval = np.array(decgridval), 
             ragridval = np.array(ragridval))


    if (plot_results): 

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
        plt.ylabel('$\Delta$(F110W - F160W) Median')
        plt.savefig(op.splitext(savefilename)[0] + '.rgbcolor.png', bbox_inches=0)

        plt.figure(3)
        plt.clf()
        plt.plot(rnarrow, cmean_narrow, ',', color='blue', alpha=0.5)
        plt.axis([min(rnarrow), max(rnarrow), -0.075, 0.075])
        plt.xlabel('Major Axis Length (degrees)')
        plt.ylabel('$\Delta$(F110W - F160W) Mean')
        plt.savefig(op.splitext(savefilename)[0] + '.rgbcolormean.png', bbox_inches=0)

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
    d_arcsec = 20.
    mr = [18.75, 19.75]

    f = 0.15
    nb = 7

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

def read_allbricks(resultsdir = '../Unreddened/'):

    fileroot = resultsdir + 'allbricks'
    savefilename = fileroot + '.npz'
    dat = np.load(savefilename)
    
    ra = dat['ranarrow']
    dec = dat['decnarrow']
    r = dat['rnarrow']
    cstd = dat['cstd_narrow']
    cm = dat['cmean_narrow']
    cmed = dat['cm_narrow']
    nstar = dat['nstar_narrow']
    ragrid = dat['ragridval']
    decgrid = dat['decgridval']
    rgrid = dat['rgridval']
    cstdgrid = dat['cstdgridval']
    cmgrid = dat['cmeangridval']
    cmedgrid = dat['cmgridval']
    nstargrid = dat['nstargridval']

    return ra, dec, r, cstd, cm, cmed, nstar, \
        ragrid, decgrid, rgrid, cstdgrid, cmgrid, cmedgrid, nstargrid

def return_ra_dec_for_radius(r):

    # conversion from degrees to radians
    radeg  = np.pi / 180.

    # default M31 parameters (see also compleness.py)
    m31ra  = 10.6847929
    m31dec = 41.2690650

    pa = 38.5
    incl = 74.

    # useful intermediate quantities
    m31pa  = pa * radeg
    incl   = incl * radeg
    b_over_a = math.cos(incl)
    ecc = math.sqrt(1. - (b_over_a)**2.)
    a = r
    b = a * b_over_a
    
    phi = radeg * np.arange(0., 360., 0.2)
    rvec = a*b / np.sqrt((b*np.cos(phi))**2 + (a*np.sin(phi))**2)

    dra  = rvec * np.sin(m31pa + phi)
    ddec = rvec * np.cos(m31pa + phi)

    ra  = m31ra + dra / math.cos(m31dec * radeg) 
    dec = m31dec + ddec

    return ra, dec
    
def plot_allbricks_ra_dec(saveplots = True, makenoise = False):

    resultsdir = '../Unreddened/FourthRun15arcsec/'
    fileroot = resultsdir + 'allbricks'
    savefilename = fileroot + '.npz'
    newsavefilename = fileroot + '.clean.npz'
    noise_tweek = 0.0
    if (makenoise == True):
        #noise_tweek = 0.0075
        noise_tweek = 0.015
        print 'Making noise file with more generous selection'
        newsavefilename = fileroot + '.noise.clean.npz'
        #print 'Not saving images'
        #saveplots = False
    
    rplot_vec = [0.25, 0.55, 1.05]

    ra, dec, r, cstd, cm, cmed, nstar, \
        ragrid, decgrid, rgrid, cstdgrid, cmgrid, cmedgrid, nstargrid \
        = read_allbricks(resultsdir = resultsdir)

    nstarmodelgrid = get_nstar_at_ra_dec(ragrid, decgrid, renormalize_to_surfdens=True)
    print ra.shape
    print nstarmodelgrid.shape

    # set selection of low extinction points off of log(nstar) vs rgb
    # width (cstd)  (optimized for 15 arcsec selection)

    a_rgbw = 0.0645
    b_rgbw = 0.0555
    a_rgbw_lo = 0.020
    b_rgbw_lo = 0.0545
    ref_lgn = -0.5
    ref_rgbw = 0.05
    min_radius = 0.1

    # tweak selection to be more generous if selecting noise files
    b_rgbw = b_rgbw + noise_tweek
    b_rgbw_lo = b_rgbw_lo + noise_tweek

    # tweak selection to be a bit more generous when using Draine AV as a cross check
    b_tweek = 0.0075
    b_rgbw = b_rgbw + b_tweek
    b_rgbw_lo = b_rgbw_lo + b_tweek

    # fit polynomial to nstar vs mean color, doing some sensible rejection & iteration
    a_cm = -0.06
    b_cm = 0.03
    ref_cm = 0.03
    max_lgn = 0.5
    i_keep = np.where((cmgrid < ref_cm) & (np.log10(nstargrid) < max_lgn) & 
                      (cmgrid < b_cm + a_cm * (np.log10(nstargrid) - ref_lgn)))
    npoly = 4
    param_n_cm = np.polyfit(np.log10(nstargrid[i_keep]), cmgrid[i_keep], npoly)
    p_n_cm_1 = np.poly1d(param_n_cm)
    i_keep = np.where(np.abs(p_n_cm_1(np.log10(nstargrid)) - cmgrid) < 0.025)
    param_n_cm = np.polyfit(np.log10(nstargrid[i_keep]), cmgrid[i_keep], npoly)
    p_n_cm_2 = np.poly1d(param_n_cm)
    i_keep = np.where(np.abs(p_n_cm_1(np.log10(nstargrid)) - cmgrid) < 0.015)
    param_n_cm = np.polyfit(np.log10(nstargrid[i_keep]), cmgrid[i_keep], npoly)
    p_n_cm = np.poly1d(param_n_cm)
    rgbcm_stddev = np.std(np.abs(p_n_cm_1(np.log10(nstargrid[i_keep])) - cmgrid[i_keep]))
    print 'Dispersion: ', rgbcm_stddev
    cm_stddev_range=[2.0, 5.0]

    # cross-reference with Draine AV map to isolate regions that also have low dust
    # emission

    drainefile = '../draine_M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.AV.fits'
    print 'Opening Draine image...'
    f = pyfits.open(drainefile)
    hdr, draineimg = f[0].header, f[0].data

    # get ra dec of draine image
    wcs = pywcs.WCS(hdr)
    # get Draine pixel locations
    img_coords = wcs.wcs_sky2pix(ra, dec, 1)
    img_coords_grid = wcs.wcs_sky2pix(ragrid, decgrid, 1)

    # grab values at those locations
    draineAV = draineimg[np.rint(img_coords[0]).astype('int'),
                         np.rint(img_coords[1]).astype('int')]
    draineAVgrid = draineimg[np.rint(img_coords_grid[0]).astype('int'),
                             np.rint(img_coords_grid[1]).astype('int')]
    
    # Set Draine A_V limit
    draineratiofix = 2.3
    #draineAVlim = 0.25 * draineratiofix
    desiredAVlim = 0.25
    if (makenoise):
        desiredAVlim *= 1.5
    draineAVlim = desiredAVlim * draineratiofix
    print 'Clipping to regions where A_V < ', desiredAVlim

    # Generate indices for grid points and all stars that meet
    # likely low-extinction criteria.

    i_lowAV_grid = np.where(((cstdgrid < b_rgbw +  a_rgbw * (np.log10(nstargrid) - ref_lgn)) |
                             (cstdgrid < b_rgbw_lo +  a_rgbw_lo * (np.log10(nstargrid) - ref_lgn))) & 
                            (rgrid > min_radius) & 
                            (cmgrid < p_n_cm(np.log10(nstargrid))  
                             + cm_stddev_range[0]*rgbcm_stddev) & 
                            (cmgrid > p_n_cm(np.log10(nstargrid)) 
                                 - cm_stddev_range[1]*rgbcm_stddev) &
                            (draineAVgrid < draineAVlim))
    i_lowAV_stars = np.where(((cstd < b_rgbw +  a_rgbw * (np.log10(nstar) - ref_lgn)) |
                              (cstd < b_rgbw_lo +  a_rgbw_lo * (np.log10(nstar) - ref_lgn))) & 
                             (r > min_radius)  & 
                             (cm < p_n_cm(np.log10(nstar))  
                              + cm_stddev_range[0]*rgbcm_stddev) & 
                             (cm > p_n_cm(np.log10(nstar)) 
                              - cm_stddev_range[1]*rgbcm_stddev) &
                            (draineAV < draineAVlim))

    # save data from low extinction regions to file
    # save clean data to a new file

    dat = np.load(savefilename)
    np.savez(newsavefilename,
             cnarrow = dat['cnarrow'][i_lowAV_stars],
             mnarrow = dat['mnarrow'][i_lowAV_stars],
             ikeep_narrow = dat['ikeep_narrow'][i_lowAV_stars], 
             ranarrow = dat['ranarrow'][i_lowAV_stars],
             decnarrow = dat['decnarrow'][i_lowAV_stars],
             rnarrow = dat['rnarrow'][i_lowAV_stars],
             cstd_narrow = dat['cstd_narrow'][i_lowAV_stars],
             cm_narrow = dat['cm_narrow'][i_lowAV_stars],
             nstarnarrow = dat['nstar_narrow'][i_lowAV_stars],
             ragrid = dat['ragridval'][i_lowAV_grid],
             decgrid = dat['decgridval'][i_lowAV_grid],
             rgrid = dat['rgridval'][i_lowAV_grid],
             cstdgrid = dat['cstdgridval'][i_lowAV_grid],
             cmgrid = dat['cmeangridval'][i_lowAV_grid],
             cmedgrid = dat['cmgridval'][i_lowAV_grid],
             nstargrid = dat['nstargridval'][i_lowAV_grid],
             param_n_cm = param_n_cm,
             rgbcm_stddev = rgbcm_stddev,
             cm_stddev_range = cm_stddev_range,
             a_rgbw = a_rgbw,
             b_rgbw = b_rgbw,
             a_rgbw_lo = a_rgbw_lo,
             b_rgbw_lo = b_rgbw_lo,
             ref_lgn = ref_lgn,
             ref_rgbw = ref_rgbw)
    print 'Saved %d stars of cleaned data to %s' % (len(dat['cnarrow'][i_lowAV_stars]),
                                                     newsavefilename)
    del dat   # clear memory

    # set up vectors for plotting the selection boundaries

    lgn = np.arange(-1.5,0.5,0.01)
    rgbw = b_rgbw +  a_rgbw * (lgn - ref_lgn)
    ref_rgbw = b_rgbw_lo +  a_rgbw_lo * (lgn - ref_lgn)
    rgbw[np.where(rgbw < ref_rgbw)] = ref_rgbw
    rgbcm_0 = b_cm + a_cm * (lgn - ref_lgn)
    rgbcm_0[np.where(rgbcm_0 > ref_cm)] = ref_cm
    rgbcm = p_n_cm(lgn)

    # set up default point size in maps based on number of grid points (s=1 for d_arcsec=10 case)
    print len(nstargrid), ' grid points'
    s_size = 1.0 * (48185. / len(nstargrid))

    # Plot lots of diagnostics for paper

    plt.figure(1)
    plt.clf()
    im = plt.scatter(rgrid, nstargrid, c=cstdgrid, s=s_size, 
                     linewidth=0, cmap='jet', alpha=0.5, vmin=0, vmax=0.15)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0, 1.55, 0, 3.25])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('$\Sigma_{stars}$  (arcsec$^{-2}$)')
    plt.title('RGB Width')
    if (saveplots): 
        plotfilename = fileroot + '.radius_nstars.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(2)
    plt.clf()
    im = plt.scatter(rgrid, np.log10(nstargrid), c=cstdgrid, s=s_size, 
                     linewidth=0, cmap='jet', alpha=0.5, vmin=0, vmax=0.15)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0, 1.55, -1.6, np.log10(3.5)])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('Log$_{10}$ $\Sigma_{stars}$  (arcsec$^{-2}$)')
    plt.title('RGB Width')
    if (saveplots): 
        plotfilename = fileroot + '.radius_lognstars.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(3)
    plt.clf()
    im = plt.scatter(nstargrid, cstdgrid, c=rgrid, s=3, 
                     linewidth=0, cmap='jet', alpha=0.25, vmin=0, vmax=1.55)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0, 3.25, 0.01, 0.21])
    plt.xlabel('$\Sigma_{stars}$  (arcsec$^{-2}$)')
    plt.ylabel('RGB Width')
    plt.title('Major Axis Length (degrees)')
    if (saveplots): 
        plotfilename = fileroot + '.nstars_rgbwidth_radiuscode.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(4)
    plt.clf()
    im = plt.scatter(nstargrid, cstdgrid, c=cmgrid, s=3, 
                     linewidth=0, cmap='seismic', alpha=0.25, vmin=-0.05, vmax=0.05)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0, 3.25, 0.01, 0.21])
    plt.xlabel('$\Sigma_{stars}$  (arcsec$^{-2}$)')
    plt.ylabel('RGB Width')
    plt.title('Mean RGB Color Offset')
    if (saveplots): 
        plotfilename = fileroot + '.nstars_rgbwidth_meancolorcode.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(22)
    plt.clf()
    im = plt.scatter(rgrid, cstdgrid, c=cmgrid, s=3, 
                     linewidth=0, cmap='seismic', alpha=0.25, vmin=-0.05, vmax=0.05)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0, 1.55, 0.01, 0.21])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('RGB Width')
    plt.title('Mean RGB Color Offset')
    # overplot low extinction regions
    plt.plot(rgrid[i_lowAV_grid], cstdgrid[i_lowAV_grid], 'bo', mew=0, markersize=2.5, alpha=1.0)
    if (saveplots): 
        plotfilename = fileroot + '.radius_rgbwidth_meancolorcode.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(5)
    plt.clf()
    im = plt.scatter(np.log10(nstargrid), cstdgrid, c=rgrid, s=3, 
                     linewidth=0, cmap='jet', alpha=0.25, vmin=0, vmax=1.55)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([-1.6, np.log10(3.5), 0.01, 0.21])
    plt.xlabel('Log$_{10}$ $\Sigma_{stars}$  (arcsec$^{-2}$)')
    plt.ylabel('RGB Width')
    plt.title('Major Axis Length (degrees)')
    plt.plot(lgn, rgbw, color='black', linewidth=3)
    if (saveplots): 
        plotfilename = fileroot + '.lognstars_rgbwidth_radiuscode.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(6)
    plt.clf()
    im = plt.scatter(np.log10(nstargrid), cstdgrid, c=cmgrid, s=3, 
                     linewidth=0, cmap='seismic', alpha=0.25, vmin=-0.05, vmax=0.05)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([-1.6, np.log10(3.5), 0.01, 0.21])
    plt.xlabel('Log$_{10}$ $\Sigma_{stars}$  (arcsec$^{-2}$)')
    plt.ylabel('RGB Width')
    plt.title('Mean RGB Color Shift')
    plt.plot(lgn, rgbw, color='black', linewidth=3)
    if (saveplots): 
        plotfilename = fileroot + '.lognstars_rgbwidth_meancolorcode.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(20)
    plt.clf()
    im = plt.scatter(np.log10(nstargrid), cmgrid, c=cstdgrid, s=3, 
                     linewidth=0, cmap='jet', alpha=0.25, vmin=0.01, vmax=0.21)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([-1.6, np.log10(3.5), -0.075, 0.075])
    plt.xlabel('Log$_{10}$ $\Sigma_{stars}$  (arcsec$^{-2}$)')
    plt.ylabel('Mean RGB Color Shift')
    plt.title('RGB Width')
    #plt.plot(lgn, rgbcm_0, color='black', linewidth=1)
    #plt.plot(lgn, rgbcm_1, color='black', linewidth=2)
    #plt.plot(lgn, rgbcm_2, color='black', linewidth=3)
    plt.plot(lgn, rgbcm, color='black', linewidth=3)
    plt.plot(lgn, rgbcm + cm_stddev_range[0]*rgbcm_stddev, 
             color='black', linewidth=1)
    plt.plot(lgn, rgbcm - cm_stddev_range[1]*rgbcm_stddev, 
             color='black', linewidth=1)
    # overplot low extinction regions
    plt.plot(np.log10(nstargrid[i_lowAV_grid]), cmgrid[i_lowAV_grid], 'bo', mew=0, markersize=2.5, alpha=1.0)
    if (saveplots): 
        plotfilename = fileroot + '.lognstars_meancolor_rgbwidthcode.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(7)
    plt.clf()
    im = plt.scatter(ragrid, decgrid, c=cmgrid, s=s_size, 
                     linewidth=0, cmap='seismic', alpha=0.5, vmin=-0.05, vmax=0.05)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('Mean RGB Color Shift')
    for rval in rplot_vec:
        raell, decell = return_ra_dec_for_radius(rval)
        plt.plot(raell, decell, color='black', linewidth=2, linestyle='-')
    if (saveplots): 
        plotfilename = fileroot + '.meancolor_position.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(8)
    plt.clf()
    im = plt.scatter(ragrid, decgrid, c=cmedgrid, s=s_size, 
                     linewidth=0, cmap='seismic', alpha=0.5, vmin=-0.05, vmax=0.05)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('Median RGB Color Shift')
    for rval in rplot_vec:
        raell, decell = return_ra_dec_for_radius(rval)
        plt.plot(raell, decell, color='black', linewidth=2, linestyle='-')
    if (saveplots): 
        plotfilename = fileroot + '.mediancolor_position.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(9)
    plt.clf()
    im = plt.scatter(ragrid, decgrid, c=cstdgrid, s=s_size, 
                     linewidth=0, cmap='jet', alpha=0.5, vmin=0.02, vmax=0.15)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('RGB Width')
    for rval in rplot_vec:
        raell, decell = return_ra_dec_for_radius(rval)
        plt.plot(raell, decell, color='black', linewidth=2, linestyle='-')
    if (saveplots): 
        plotfilename = fileroot + '.rgbwidth_position.png'
        plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(10)
    plt.clf()
    im = plt.scatter(ragrid, decgrid, c=cstdgrid, s=s_size, 
                     linewidth=0, cmap='PuRd', alpha=0.5, vmin=0.025, vmax=0.15)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('RGB Width w/ Low $A_V$')
    # overplot low extinction regions
    plt.plot(ragrid[i_lowAV_grid], decgrid[i_lowAV_grid], 'bo', mew=0, markersize=1., alpha=1.0)
    for rval in rplot_vec:
        raell, decell = return_ra_dec_for_radius(rval)
        plt.plot(raell, decell, color='black', linewidth=2, linestyle='-')
    if (saveplots): 
        plotfilename = fileroot + '.rgbwidth_position_lowAV.png'
        plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(11)
    plt.clf()
    im = plt.scatter(ragrid, decgrid, c=nstargrid, s=s_size, 
                     linewidth=0, cmap='jet', alpha=0.5, vmin=0, vmax=3)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('$\Sigma_{stars}$  (arcsec$^{-2}$)')
    for rval in rplot_vec:
        raell, decell = return_ra_dec_for_radius(rval)
        plt.plot(raell, decell, color='black', linewidth=2, linestyle='-')
    if (saveplots): 
        plotfilename = fileroot + '.nstar_position.png'
        plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(21)
    plt.clf()
    im = plt.scatter(ragrid, decgrid, c=np.log10(nstargrid), s=s_size, 
                     linewidth=0, cmap='gist_ncar', alpha=0.5, vmin=-1.2, vmax=np.log10(3.0))
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('Log$_{10} \Sigma_{stars}$  (arcsec$^{-2}$)')
    for rval in rplot_vec:
        raell, decell = return_ra_dec_for_radius(rval)
        plt.plot(raell, decell, color='black', linewidth=2, linestyle='-')
    if (saveplots): 
        plotfilename = fileroot + '.lognstar_position.png'
        plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(22)
    plt.clf()
    im = plt.scatter(ragrid, decgrid, c=np.log10(nstarmodelgrid), s=s_size, 
                     linewidth=0, cmap='gist_ncar', alpha=0.5, vmin=-1.2, vmax=np.log10(3.0))
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('Model Log$_{10} \Sigma_{stars}$  (arcsec$^{-2}$)')
    for rval in rplot_vec:
        raell, decell = return_ra_dec_for_radius(rval)
        plt.plot(raell, decell, color='black', linewidth=2, linestyle='-')
    if (saveplots): 
        plotfilename = fileroot + '.lognstarmodel_position.png'
        plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(14)
    plt.clf()
    im = plt.scatter(cstdgrid, cmgrid, c=rgrid, s=3, 
                     linewidth=0, cmap='jet', alpha=0.45, vmin=0, vmax=1.55)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0.01, 0.15, -0.075, 0.075])
    plt.xlabel('RGB Width')
    plt.ylabel('Mean Color Offset')
    plt.title('Major Axis Length (degrees)')
    # overplot low extinction regions
    plt.plot(cstdgrid[i_lowAV_grid], cmgrid[i_lowAV_grid], 'bo', mew=0, markersize=2.5, alpha=1.0)
    if (saveplots): 
        plotfilename = fileroot + '.rgbwidth_meancolor_radiuscode.png'
        plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(15)
    plt.clf()
    im = plt.scatter(cstdgrid, cmgrid, c=np.log10(nstargrid), s=3, 
                     linewidth=0, cmap='jet', alpha=0.45, vmin=-1.6, vmax=0.5)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0.01, 0.15, -0.075, 0.075])
    plt.xlabel('RGB Width')
    plt.ylabel('Mean Color Offset')
    plt.title('Log$_{10}$ $\Sigma_{stars}$  (arcsec$^{-2}$)')
    # overplot low extinction regions
    plt.plot(cstdgrid[i_lowAV_grid], cmgrid[i_lowAV_grid], 'bo', mew=0, markersize=2.5, alpha=1.0)
    if (saveplots): 
        plotfilename = fileroot + '.rgbwidth_meancolor_nstarcode.png'
        plt.savefig(plotfilename, bbox_inches=0)
    
    plt.figure(12)
    plt.clf()
    i = np.arange(len(r[i_lowAV_stars]))
    plt.plot(np.sort(r[i_lowAV_stars]), i)
    plt.axis=[0, 1.55, 0, len(i_lowAV_stars)]
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('Number of Stars in Clean Sample')
    plotfilename = fileroot + '.nstars_cumulative_radius.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)


    plt.figure(13)
    plt.clf()
    i = np.arange(len(np.log10(nstar[i_lowAV_stars])))
    plt.plot(np.sort(np.log10(nstar[i_lowAV_stars])), i)
    plt.axis=[-1.6, np.log10(3.5), 0, len(i_lowAV_stars)]
    plt.xlabel('Log$_{10}$ $\Sigma_{stars}$  (arcsec$^{-2}$)')
    plt.ylabel('Number of Stars in Clean Sample')
    plotfilename = fileroot + '.nstars_cumulative_nstars.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)

def clean_low_AZ_sample(tolerance = 0.0025, color_tolerance=0.02, makenoise=False):

    resultsdir = '../Unreddened/'
    fileroot = resultsdir + 'allbricks'
    savefilename = fileroot + '.npz'
    dat = np.load(savefilename)
    
    if makenoise:
        fileroot = resultsdir + 'allbricks.noise'
        savefilename = fileroot + '.npz'

    r = dat['rnarrow']
    cstd = dat['cstd_narrow']
    #cm = dat['cm_narrow']
    cm = dat['cmean_narrow']

    plt.figure(1)
    plt.clf()
    plt.plot(r, cstd, ',', color='black', alpha=0.5)
    plt.axis([0, 1.55, 0, 0.15])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('RGB width')

    # do a first cull on obvious bad regions

    #i_bad = np.where(((r>0.58) & (r <1.2) & (cstd > 0.052)) | 
    #                 ((r > 1.05) & (r < 1.25)) |
    #                 (r < 0.05) |
    #                 ((r > 0.39) & (r < 0.44) & (cstd > 0.075)) |
    #                 ((r > 0.44) & (r < 0.6) & (cstd > 0.055)) |
    #                 ((r > 1.23) & (cstd > 0.035)) |
    #                 ((r > 0.32) & (r < 0.39)), 1, 0)
    #i_bad_noise = np.where(((r>0.58) & (r <1.2) & (cstd > 0.03)) | 
    #                       ((r > 1.09) & (r < 1.2) & (cstd > 0.0225)) |
    #                       (r < 0.05) |
    #                       ((r > 0.39) & (r < 0.44) & (cstd > 0.0425)) |
    #                       ((r > 0.44) & (r < 0.6) & (cstd > 0.035)) |
    #                       ((r > 0.32) & (r < 0.39)), 1, 0)
    i_bad = np.where((r < 0.05) |
                     ((r > 0.41) & (r < 0.61) & (cstd > 0.085)) |
                     ((r > 0.6) & (r < 0.65)) |
                     ((r>0.6) & (r <1.2) & (cstd > 0.07)) |
                     ((r > 0.6) & (r < 0.7) & (cstd > 0.065)) | 
                     ((r > 0.77) & (r < 1.1) & (cstd > 0.052)) | 
                     ((r > 0.85) & (r < 0.95) & (cstd > 0.045)) | 
                     ((r > 0.95) & (r < 1.03)) |
                     ((r > 1.07) & (r < 1.25)) |
                     ((r > 1.15) & (cstd > 0.04)) |
                     ((r > 1.25) & (cstd > 0.035)) |
                     (r > 1.4), 1, 0)
    i_bad_noise = np.where(((r>0.7) & (r <1.1) & (cstd > 0.07)) | 
                           ((r > 0.9) & (r < 1.3) & (cstd > 0.05)) |
                           (r < 0.05) |
                           ((r > 0.47) & (r < 0.65) & (cstd > 0.08)) |
                           ((r > 0.32) & (r < 0.39)), 1, 0)
    i_bad_noise = i_bad
    if makenoise:
        i_good = np.where(i_bad_noise == 0)
    else:
        i_good = np.where(i_bad == 0)
    plt.plot(r[i_good], cstd[i_good], ',', color='green', alpha=0.5)

    # fit polynomials to good points

    npoly = 4                #...... RGB width
    param = np.polyfit(r[i_good], cstd[i_good], npoly)
    print 'Polynomial: ', param 
    p = np.poly1d(param)

    rp = np.linspace(0, max(r), 100)

    npoly = 4                #..... RGB color
    param_cm = np.polyfit(r[i_good], cm[i_good], npoly)
    print 'Polynomial: ', param_cm

    p_cm = np.poly1d(param_cm)

    # keep everything within a given tolerance of the polynomial

    #ikeep = np.where((cstd - p(r) < tolerance) & (r > 0.03))
    ikeep = np.where(((cstd - p(r))    < tolerance) & 
                     ((cm   - p_cm(r)) < color_tolerance) & 
                     (r > 0.03))
    plt.plot(r[ikeep], cstd[ikeep], ',', color='magenta')
    plt.plot(rp, p(rp), color='blue', linewidth=3)

    # repeat polynomial fit and culling 

    npoly = 4                #...... RGB width
    param = np.polyfit(r[ikeep], cstd[ikeep], npoly)
    print 'Polynomial: ', param 
    p = np.poly1d(param)

    rp = np.linspace(0, max(r), 100)

    npoly = 4                #..... RGB color
    param_cm = np.polyfit(r[ikeep], cm[ikeep], npoly)
    print 'Polynomial: ', param_cm

    p_cm = np.poly1d(param_cm)

    # keep everything within a given tolerance of the second polynomials. FINAL!
    #  SHUT OFF EVERYTHING ABOVE POLYNOMIAL!

    final_tolerance = 0.0

    ikeep = np.where((cstd - p(r) < final_tolerance) &  
                     ((cm   - p_cm(r)) < color_tolerance) & (r > 0.05))
    #ikeep = np.where(((cstd - p(r))    < tolerance) & 
    #                 ((cm   - p_cm(r)) < cm_tolfac*tolerance) & 
    #                 (r > 0.03))
    plt.plot(r[ikeep], cstd[ikeep], ',', color='red')
    plt.plot(rp, p(rp), color='yellow', linewidth=3)

    # label and title plot...

    Acol_AV = 0.33669 - 0.20443
    print 'Tolerance in units of A_V: ', tolerance / Acol_AV
    plt.title('Tolerance = '+ ("%g" % round(tolerance,4)) + ' ($\Delta A_V$ = ' + ("%g" % round(tolerance / Acol_AV,3)) + ')')

    plotfilename = fileroot + '.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)

    # make other diagnostic plots

    plt.figure(5)
    plt.clf()
    im = plt.scatter(r, cstd, c=cm, s=1, linewidth=0, cmap='seismic', alpha=0.1, vmin=-0.05, vmax=0.05)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0, 1.55, 0, 0.15])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('RGB Width')
    plt.plot(rp, p(rp) + final_tolerance, ',', color='black', linewidth=3, linestyle='-')
    plotfilename = fileroot + '.meancolor.clean.colorcoded.png'
    plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(2)
    plt.clf()
    plt.plot(r, cm, ',', color='black', alpha=0.5)
    plt.axis([0, 1.55, -0.07, 0.07])
    plt.plot(r[ikeep], cm[ikeep], ',', color='magenta', alpha=0.5)
    plt.plot(rp, p_cm(rp), color='blue', linewidth=3)
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('$\Delta(F110W - F160W)$')
    plt.title('Tolerance = '+ ("%g" % round(color_tolerance,5)) + ' ($\Delta A_V$ = ' + ("%g" % round(color_tolerance / Acol_AV,3)) + ')')
    plotfilename = fileroot + '.meancolor.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(3)
    plt.clf()
    plt.plot(cstd, cm, ',', color='black')
    plt.plot(cstd[ikeep], cm[ikeep], ',', color='magenta')
    plt.axis([0, 0.15, -0.07, 0.07])
    plt.xlabel('RGB Width')
    plt.ylabel('$\Delta(F110W - F160W)$')
    plotfilename = fileroot + '.meancolorvswidth.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)

    plt.figure(4)
    plt.clf()
    i = np.arange(len(r[ikeep]))
    plt.plot(np.sort(r[ikeep]), i)
    plt.axis=[0, 1.55, 0, len(ikeep)]
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('Number of Stars in Clean Sample')
    plotfilename = fileroot + '.nstars.clean.png'
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
    #resultsdir = '../Unreddened/FourthRun15arcsec/'
    savefilename = resultsdir + 'allbricks.clean.npz'
    if readnoise:
        savefilename = resultsdir + 'allbricks.noise.clean.npz'
    
    print 'Reading clean data from ', savefilename

    dat = np.load(savefilename)

    c = dat['cnarrow']
    m = dat['mnarrow']
    #ikeep = dat['ikeep_narrow']
    ra = dat['ranarrow']
    dec = dat['decnarrow']
    r = dat['rnarrow']
    cstd = dat['cstd_narrow']
    cm = dat['cm_narrow']
    nstar = dat['nstarnarrow']

    return c, m, ra, dec, r, cstd, cm, nstar

def make_low_AV_cmd(c, m, 
                    mrange = [18.2, 25.], 
                    crange = [0.3,1.3], 
                    deltapix = [0.015,0.15], 
                    masksig=[2.5,3.0]):

    nbins = [int((mrange[1] - mrange[0]) / deltapix[1]),
             int((crange[1] - crange[0]) / deltapix[0])]
    if (len(c) > 0): 
        cmd, mboundary, cboundary = np.histogram2d(m, c,
                                               range=[np.sort(mrange), crange], bins=nbins)
    else:
        # use dummy values to force an empty grid to return
        cmd, mboundary, cboundary = np.histogram2d([0], [0],
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
                            maglimoff = [0.0, 0.25],
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
            print 'Merging ', bad_dr[0], ' with ', bad_dr[0] + 1, '.  ', \
                len(bad_dr)-1, ' small gaps remaining.'
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
    m110file = 'completeness_ra_dec.st.F110W.radius.npz'
    m160file = 'completeness_ra_dec.st.F160W.radius.npz'
        
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

        maglim110 = p110(meanr_array[i]) - maglimoff[0]
        maglim160 = p160(meanr_array[i]) - maglimoff[1]
        maglim_array[:,i] = np.array([maglim110, maglim160])

        m110range = [16.0, maglim110]
        m160range = [18.4, maglim160]

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
            if (nfg == 0): 
                frac_noise = 0.0
            else:
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

def make_nstar_selected_low_AV_cmds(nrgbstars = 2750, nsubstep=6., 
                            mrange = [18.2, 25.],
                            crange = [0.3, 2.5],
                            deltapixorig = [0.015,0.25],
                            mnormalizerange = [19,21.5], 
                            maglimoff = [0.0, 0.5],
                            nsig_blue_color_cut = 2.0, blue_color_cut_mask_only=False,
                            usemask=True, masksig=[2.5,3.0],
                            makenoise=False, noisemasksig=[4.5,3.5],
                            useq=False, reference_color=1.0,
                            restricted_n_range=''):

    # Define reddening parameters

    Amag_AV = 0.20443
    Acol_AV = 0.33669 - 0.20443
    t = np.arctan(-Amag_AV / Acol_AV)
    reference_color = 1.0

    # read in cleaned, low reddening data

    c, m, ra, dec, r, cstd, cm, nstar = read_clean()

    # convert to reddening-free mag, if requested
    if useq:
        #m = m + (c-reference_color)*np.sin(t)/np.cos(t)
        m = m + (c-reference_color)*(-Amag_AV / Acol_AV)

    # sort according to increasing local stellar density, as tracked by nstar

    isort = np.argsort(nstar)
    c = c[isort]
    m = m[isort]
    r = r[isort]
    ra = ra[isort]
    dec = dec[isort]
    cstd = cstd[isort]
    cm = cm[isort]
    nstar = np.log10(nstar[isort])

    # break into bins of different nstar based on number of stars in normalization range. 
    # Normal brick wide low-AV has ~10K - 20K total, but many of those are in RC.

    istar = np.arange(len(nstar))
    irgb = np.where((mnormalizerange[0] < m) & (m < mnormalizerange[1]))[0]
    print 'Starting with ', len(nstar), ' stars total.'
    print 'Number of upper RGB Stars: ', len(irgb)

    nstep = int(nrgbstars / nsubstep)

    nrgblo = np.arange(0, len(irgb)-nstep, nstep)
    nrgbhi = nrgblo + int(nrgbstars)
    nrgbhi = np.where(nrgbhi < len(irgb)-1, nrgbhi, len(irgb)-1) # tidy up ends

    nnstarlo = istar[irgb[nrgblo]]
    nnstarhi = istar[irgb[nrgbhi]]

    nnstarhi = np.where(nnstarhi < len(istar)-1, nnstarhi, len(istar)-1)     # tidy up ends again
    nnstarbins = len(nnstarhi)

    #print 'nrgblo: ', nrgblo
    #print 'nrgbhi: ', nrgbhi
    #print 'nnstarlo: ', nnstarlo
    #print 'nnstarhi: ', nnstarhi

    print 'Splitting into ', len(nnstarhi),' bins of nstar with ',nrgbstars,' in each upper RGB.'

    # merge adjacent bins, if they're too close together....

    min_nstar = 0.05
    nnstar_orig = len(nnstarlo)
    meannstar = np.array([np.average(nstar[nnstarlo[i]:nnstarhi[i]]) 
                          for i in range(len(nnstarlo))])
    dmeannstar = np.array([meannstar[i+1] - meannstar[i] 
                           for i in range(len(nnstarlo)-1)])
    bad_dnstar = np.where(dmeannstar < min_nstar)[0]

    # are there any gaps that are too small?
    while (len(bad_dnstar) > 0): 

        # and is the gap not in the last, unmergeable bin?
        #if (bad_dnstar[0] + 1 < len(nnstarlo) - 1):
        if (bad_dnstar[0] < len(nnstarlo) - 1):

            # if so, merge the first instance with the adjacent cell
            print 'Merging ', bad_dnstar[0], ' with ', bad_dnstar[0] + 1, '.  ', \
                len(bad_dnstar)-1, ' small gaps remaining.'
            nnstarlo = np.delete(nnstarlo, bad_dnstar[0] + 1)   
            nnstarhi = np.delete(nnstarhi, bad_dnstar[0])

            meannstar = np.array([np.average(nstar[nnstarlo[i]:nnstarhi[i]]) 
                                  for i in range(len(nnstarlo))])
            dmeannstar = np.array([meannstar[i+1] - meannstar[i] 
                                   for i in range(len(nnstarlo)-1)])
            bad_dnstar = np.where(dmeannstar < min_nstar)[0]

        else:
            
            bad_dnstar = bad_dnstar[:-1]

    print 'Reduced number of bins of nstar from ', nnstar_orig, ' to ', len(nnstarlo)

    nnstarbins = len(nnstarhi)

    # run once to get shape of CMD array
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
    deltapix =  np.array([cboundary[1] - cboundary[0], 
                          mboundary[1] - mboundary[0]])

    # initialize storage arrays
    cmd_array = np.zeros([cmd.shape[0], cmd.shape[1], nnstarbins])
    mask_array = np.zeros([cmd.shape[0], cmd.shape[1], nnstarbins])
    meancol_array = np.zeros([meancol.shape[0], nnstarbins])
    sigcol_array = np.zeros([sigcol.shape[0], nnstarbins])
    meanlgnstar_array = np.zeros(nnstarbins)
    nstarrange_array = np.zeros([2,nnstarbins])
    num_per_cmd_array = np.zeros(nnstarbins)
    maglim_array = np.zeros([2,nnstarbins])

    # initialize magnitude limit polynomials
    completenessdir = '../../Completeness/'
    m110file = 'completeness_ra_dec.st.F110W.nstar.npz'
    m160file = 'completeness_ra_dec.st.F160W.nstar.npz'
        
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

    # Loop through bins of log10(nstar)
    for i in range(len(nnstarlo)):
        
        cmd, fgmask, meancol, sigcol, cboundary, mboundary, extent = \
            make_low_AV_cmd(c[nnstarlo[i]:nnstarhi[i]],
                            m[nnstarlo[i]:nnstarhi[i]],
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
        nstars_per_cmd = np.sum(cmd)
        cmd = cmd.astype(float) / float(norm)

        # copy results to appropriate array
        cmd_array[:,:,i] = cmd
        meancol_array[:,i] = cinterp(meancol)
        sigcol_array[:,i] = sigcol * deltapix[0]
        meanlgnstar_array[i] = np.average(nstar[nnstarlo[i]:nnstarhi[i]])
        nstarrange_array[:,i] = [nstar[nnstarlo[i]], nstar[nnstarhi[i]]]
        num_per_cmd_array[i] = len(nstar[nnstarlo[i]:nnstarhi[i]])

        # calculate corresponding magnitude limits

        maglim110 = p110(meanlgnstar_array[i]) - maglimoff[0]
        maglim160 = p160(meanlgnstar_array[i]) - maglimoff[1]
        maglim_array[:,i] = np.array([maglim110, maglim160])

        m110range = [16.0, maglim110]
        m160range = [18.4, maglim160]

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

        print 'Bin: ', i, '  logN: ', ("%g" % round(meanlgnstar_array[i],3)), 'maglim', ("%g" % round(maglim110,2)), ("%g" % round(maglim160,2)),' Nstars_Per_Cmd: ',nstars_per_cmd, ' NRGB: ',norm

        #plt.imshow(cmd,  extent=extent, aspect='auto', interpolation='nearest', vmin=0, vmax = 0.1)
        plt.imshow(0.01*color_mag_datamask + cmd,  extent=extent, aspect='auto', interpolation='nearest', vmin=0, vmax = 0.1)
        plt.xlabel('F110W - F160W')
        plt.ylabel('F160W')
        plt.title('Log$_{10}$ N$_{star}$: '+ ("%g" % np.round(meanlgnstar_array[i],3)) + 
                  '  F160W 50% Completeness: ' + ("%g" % np.round(maglim160,2)))
        plt.draw()

    # make a noise model, if requested

    # initialize noise array to prevent griping on return when not calculating
    # noise model.

    noise_array = np.zeros([cmd.shape[0], cmd.shape[1], nnstarbins])
    noisefrac_array = np.zeros(nnstarbins)

    if makenoise:

        noise_smooth_mag = np.array([0.2, 0.5])    # Note: filter is with opposite polarity
        noise_smooth = np.rint(noise_smooth_mag / deltapix)
        #noise_smooth = [3, 10]  # mag, color, in pixels
        print 'Building Noise Model...Smoothing with: ', noise_smooth

        c_n, m_n, ra_n, dec_n, r_n, cstd_n, cm_n, nstar_n = read_clean(readnoise = True)
        # convert to reddening-free mag, if requested
        if useq:
            m_n = m_n + (c_n-reference_color)*(-Amag_AV / Acol_AV)

        isort = np.argsort(nstar_n)
        c_n = c_n[isort]
        m_n = m_n[isort]
        r_n = r_n[isort]
        ra_n = ra_n[isort]
        dec_n = dec_n[isort]
        cstd_n = cstd_n[isort]
        cm_n = cm_n[isort]
        nstar_n = np.log10(nstar_n[isort])

        nstar_n_range = [min(nstar_n), max(nstar_n)]

        # initialize array to save noise model

        noise_array = np.zeros([cmd.shape[0], cmd.shape[1], nnstarbins])
        noisefrac_array = np.zeros(nnstarbins)

        # set up function to quickly find indices of stars within given n-range
        i_nstarinterp  = interp1d(nstar_n, np.arange(len(nstar_n)))

        for i in range(len(nnstarlo)):

            # interpolate to find range of points, enforcing boundaries of interpolation
            nstar_range = nstarrange_array[:,i]
            #print nstar_range[0], nstar_n_range[0]
            #print nstar_range[1], nstar_n_range[1]
            nlo_val = np.maximum(nstar_range[0], nstar_n_range[0])+0.000000001
            nhi_val = np.minimum(nstar_range[1], nstar_n_range[1])
            #print nlo_val, nhi_val
            i_lo = np.ceil(i_nstarinterp(nlo_val))
            i_hi = np.floor(i_nstarinterp(nhi_val))
            #print 'ilo: ', i_lo, ' ihi: ', i_hi

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
            if (nfg == 0): 
                frac_noise = 0
            else:
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
            plt.title('Log$_{10} N_{star}$: '+ ("%g" % np.round(meanlgnstar_array[i],3)) + 
                      '  F160W 50% Completeness: ' + ("%g" % np.round(maglim160,2)))
            plt.draw()

    if restricted_n_range != '':

        print 'Trimming down to restricted nstar range: ', restricted_n_range
        # fix limits of restricted_n_range
        restricted_n_range = [np.maximum(restricted_n_range[0],np.min(meanlgnstar_array)),
                              np.minimum(restricted_n_range[1],np.max(meanlgnstar_array))]
        i_nstarinterp  = interp1d(meanlgnstar_array, np.arange(len(meanlgnstar_array)))
        i = i_nstarinterp(restricted_n_range)
        ilo = np.maximum(int(np.floor(i[0])), 0)
        ihi = np.minimum(int(np.ceil(i[1]) + 1), len(meanlgnstar_array))
        print 'Grabbing index range: ', [ilo, ihi]
        return cmd_array[:,:,ilo:ihi], mask_array[:,:,ilo:ihi], \
            noise_array[:,:,ilo:ihi], noisefrac_array[ilo:ihi], \
            meanlgnstar_array[ilo:ihi], nstarrange_array[:,ilo:ihi], \
            meancol_array[:,ilo:ihi], sigcol_array[:,ilo:ihi], \
            num_per_cmd_array[ilo:ihi], maglim_array[:,ilo:ihi], mboundary, cboundary
    
    else:

        return cmd_array, mask_array, noise_array, noisefrac_array, \
            meanlgnstar_array, nstarrange_array, meancol_array, sigcol_array, \
            num_per_cmd_array, maglim_array, mboundary, cboundary

def plot_RGB_locii(plotfileroot='../Unreddened/lowAV_RGB'):

    cmda,ma,na,nfa,mnsa,nsra,mca,sca,npca,mla,mb,cb = \
        make_nstar_selected_low_AV_cmds(makenoise=False)

    # initialize magnitude limit polynomials
    completenessdir = '../../Completeness/'
    m160file = 'completeness_ra_dec.st.F160W.nstar.npz'
    m160dat = np.load(completenessdir + m160file)
    m160polyparam = m160dat['param']
    p160 = np.poly1d(m160polyparam)

    mvec = (mb[:-1] + mb[1:])/2.
    k = np.where(mvec < 24)
    print len(mvec)
    print mca.shape

    n_lines = mca.shape[1]

    plt.figure(1)
    plt.clf()
    
    # http://stackoverflow.com/questions/4805048/how-to-get-different-lines-for-different-plots-in-a-single-figure
    colormap = plt.cm.gist_ncar_r
    plt.gca().set_color_cycle([colormap(i) 
                               for i in np.linspace(0.05, 0.9, n_lines)])

    labels = []
    for i in np.arange(n_lines):
        
        # set up magnitude limit

        j = n_lines - i - 1
        mlim160 = p160(mnsa[j])
        ikeep = np.where(mvec <= mlim160)[0]

        plt.plot(mca[ikeep,j], mvec[ikeep], linewidth=3)
        #labels.append(r'$log_{10}N=%5.2f$' % (mnsa[i]))
        labels.append(r'$%5.2f$' % (mnsa[j]))

    plt.xlabel('F110W - F160W')
    plt.ylabel('F160W')
    plt.axis([0.5, 1.25, 25, 18.])
    plt.legend(labels, loc='lower right', labelspacing=0, 
               ncol=2, columnspacing=1.0)
    plt.savefig(plotfileroot+'.locii.png', bbox_inches=0)

    plt.figure(2)
    plt.clf()
    
    # http://stackoverflow.com/questions/4805048/how-to-get-different-lines-for-different-plots-in-a-single-figure
    colormap = plt.cm.gist_ncar_r
    plt.gca().set_color_cycle([colormap(i) 
                               for i in np.linspace(0.05, 0.9, n_lines)])

    labels = []
    for i in np.arange(n_lines):
        
        # set up magnitude limit

        j = n_lines - i - 1
        mlim160 = p160(mnsa[j])
        ikeep = np.where(mvec <= mlim160)[0]

        plt.plot(mvec[ikeep], sca[ikeep,j], linewidth=3)
        #labels.append(r'$log_{10}N=%5.2f$' % (mnsa[i]))
        labels.append(r'$%5.2f$' % (mnsa[j]))

    plt.xlabel('F160W')
    plt.ylabel(r"$\sigma_{F110W-F160W}$")
    plt.axis([18., 25., 0, 0.19])
    plt.legend(labels, loc='upper left', labelspacing=0, 
               ncol=4, columnspacing=1.0)

    Amag_AV = 0.20443
    AV = 0.25
    mref = 24.0
    sigref = 0.0
    arrowhead = 0.01
    plt.arrow(mref, sigref, 0*AV, Amag_AV*AV-arrowhead, width=0.05, color='r', 
              head_length=arrowhead)
    plt.annotate(r"$A_V=0.25$", xy=(24.1,0.025), xytext=None )

    plt.savefig(plotfileroot+'.width.png', bbox_inches=0)

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

def get_nstar_at_ra_dec(ra, dec, renormalize_to_surfdens=True):

    # use results from fit_disk.py to return estimate of local nstar
    # based on sum of gaussian model

    ra0  = 10.6847929
    dec0 = 41.2690650
    ncomp = 12
    p12 = [ra0, dec0, 
           5.18815078e+01,   8.12854891e+01,   1.15172061e+02,   7.99289784e-02,   
           3.87304417e+01,   8.47519489e+01,   1.84756225e+02,   8.16379274e-02,   
           3.37469947e+01,   4.09064728e+01,   8.27126913e+01,   9.37481884e-02,   
           3.20786219e+01,   4.55816926e+01,   1.26139485e+02,   1.14824313e-01,   
           5.32233413e+01,   4.22564593e+01,   8.12427291e+01,   1.49078468e-01,   
           4.64183669e+01,   7.72900759e+01,   5.48387216e+01,   1.96137141e-01,   
           4.08805925e+01,   4.64752825e+01,   3.01155728e+01,   2.33612213e-01,   
           5.36751315e+01,   7.66603019e+01,   1.32822154e+02,   4.21999937e-01,   
           4.47164325e+01,   6.99611650e+01,   1.81829413e+02,   5.69677738e-01,   
           3.13267388e+01,   7.26107049e+01,   4.15825866e+01,   6.32949036e-01,
           3.52155138e+01,   8.14328369e+01,   1.62254720e+02,   6.44182046e-01,   
           4.24411911e+01,   8.28341003e+01,   5.99901012e+01,   7.44109603e-01]   
    p = np.array(p12)

    npar = 4
    i = np.arange(0,ncomp)
    i_n0   = 4 + i*npar
    d_arcsec = 15.0

    if (renormalize_to_surfdens):
        p[i_n0] = p[i_n0] / d_arcsec**2

    return fitdisk.nstar_multigauss(np.array(p), ra, dec, ncomp=ncomp)



    

