import math
import numpy as np
import read_brick_data as rbd
import makefakecmd as mfcmd
import os.path as op
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def isolate_low_AV(filename = 'ir-sf-b17-v8-st.fits', datadir = '../../Data/',
                             frac=0.05, mrange=[19,21.5],
                             rgbparam=[22.0, 0.72, -0.13, -0.012], nrbins = 5., 
                             d_arcsec=10, savefile = True, savefiledir = '../Unreddened/'):
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

    cstdthreshvec = 0.0 * np.arange(len(rvec) - 1)
    for j in range(len(rvec)-1):
        cstdtmp = [x for i, x in enumerate(cstdlist) if ((rvec[j] <= rgood[i]) & (rgood[i] < rvec[j+1]))]
        cstdtmp.sort()
        n_cstd_thresh = int(frac * (len(cstdtmp) + 1.))
        if (len(cstdtmp) > 0) & (n_cstd_thresh <= len(cstdtmp)-1):
            cstdthreshvec[j] = cstdtmp[n_cstd_thresh]
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
        
        savefilename = savefiledir + op.splitext(filename)[0] + '.npz'
        if op.isfile(savefilename):
            # if it does, append some random characters
            print 'Output file ', savefilename, ' exists. Changing filename...'
            savefilenameorig = savefilename
            savefilename = op.splitext(savefilenameorig)[0] + '.' + mfcmd.id_generator(4) + '.npz'
            #print 'New name: ', savefilename

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

    for filename in filelist:

        c, m, i, ra, dec, r, cstd, cm = isolate_low_AV(filename = filename, frac=f, mrange=mr, nrbins=nb, 
                                                       savefile=True,
                                                       datadir = datadir,
                                                       savefiledir = resultsdir)        
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
    plt.title('Color Shift Relative to F110W-F160 = '+str(rgbparam[1])+' at F160W = ' + str(rgbparam[0]))
    plt.savefig(op.splitext(savefilename)[0] + '.rgbcolor.png', bbox_inches=0)

    return

def clean_low_AZ_sample(tolerance = 0.005):

    resultsdir = '../Unreddened/'
    savefilename = resultsdir + 'allbricks.npz'

    dat = np.load(savefilename)
    
    r = dat['rnarrow']
    cstd = dat['cstd_narrow']
    cm = dat['cm_narrow']

    plt.figure(1)
    plt.clf()
    plt.plot(r, cstd, ',', color='red', alpha=0.5)
    plt.axis([0, 1.35, 0, 0.15])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('RGB width')

    # do a first cull on obvious bad regions

    i_bad = np.where(((r>0.58) & (r <1.2) & (cstd > 0.052)) | 
                     ((r > 1.09) & (r < 1.2)) |
                     (r < 0.03) |
                     ((r > 0.39) & (r < 0.44) & (cstd > 0.075)) |
                     ((r > 0.44) & (r < 0.6) & (cstd > 0.055)) |
                     ((r > 1.23) & (cstd > 0.035)) |
                     ((r > 0.32) & (r < 0.39)), 1, 0)
    i_good = np.where(i_bad == 0)
    plt.plot(r[i_good], cstd[i_good], ',', color='blue', alpha=0.5)

    # fit a polynomial to good points
    npoly = 8
    param = np.polyfit(r[i_good], cstd[i_good], npoly)
    print 'Polynomial: ', param 

    p = np.poly1d(param)
    rp = np.linspace(0, max(r), 100)
    plt.plot(rp, p(rp), color='green')

    plotfilename = resultsdir + 'allbricks.clean.png'
    plt.savefig(plotfilename, bbox_inches=0)


    # keep everything within a given tolerance of the polynomial

    ikeep = np.where((cstd - p(r) < tolerance) & (r > 0.03))
    plt.plot(r[ikeep], cstd[ikeep], ',', color='magenta')

    Acol_AV = 0.33669 - 0.20443
    print 'Tolerance in units of A_V: ', tolerance / Acol_AV
    plt.title('Tolerance = '+ ("%g" % round(tolerance,4)) + ' ($\Delta A_V$ = ' + ("%g" % round(tolerance / Acol_AV,3)) + ')')

    # save clean data to a new file

    newsavefilename = resultsdir + 'allbricks.clean.npz'
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
             polyparam = param)

    return

def read_clean():

    resultsdir = '../Unreddened/'
    savefilename = resultsdir + 'allbricks.clean.npz'
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
                    deltapix = [0.015,0.15]):

    nbins = [int((mrange[1] - mrange[0]) / deltapix[1]),
             int((crange[1] - crange[0]) / deltapix[0])]
    cmd, mboundary, cboundary = np.histogram2d(m, c,
                                               range=[np.sort(mrange), crange], bins=nbins)

    deltapix =  [cboundary[1]-cboundary[0],mboundary[1]-mboundary[0]]
    extent = [cboundary[0], cboundary[-1],
              mboundary[-1], mboundary[0]]

    nmag = cmd.shape[0]
    ncol = cmd.shape[1]

    nsig_clip = 2.5
    nanfix = 0.0000000000001
    niter = 5

    mask = 1.0 + np.zeros(cmd.shape)
    for j in range(niter):

        # Get mean color and width of each line

        meancol = np.array([sum(np.arange(ncol) * cmd[i,:] * mask[i,:]) / 
                            sum(cmd[i,:] * (mask[i,:] + nanfix)) for i in range(nmag)])
        sigcol = np.array([np.sqrt(sum(cmd[i,:]*mask[i,:] *
                                    (range(ncol) - meancol[i])**2) / 
                                sum(cmd[i,:]*mask[i,:] + nanfix)) 
                           for i in range(nmag)])

        # Mask the region near the RGB

        mask = np.array([np.where(abs(range(ncol) - meancol[i]) < 
                                  nsig_clip*sigcol[i], 1.0, 0.0) 
                         for i in range(nmag)])

    return cmd, meancol, sigcol, cboundary, mboundary, extent


def make_radial_low_AV_cmds(nrgbstars = 4000, nsubstep=3., mnormalizerange = [19,21.5])

    c, m, ra, dec, r, cstd, cm = read_clean()

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

    nsubstep = 3. # number of substeps (i.e. overlapping bins, centers shifted by nstar/nubstep)
    nstep = int(nrgbstars / nsubstep)

    nrgblo = np.arange(0, len(irgb)-nstep, nstep)
    nrgbhi = nrgblo + int(nrgbstars)
    nrgbhi = np.where(nrgbhi < len(irgb)-1, nrgbhi, len(irgb)-1) # tidy up ends

    nrlo = istar[irgb[nrgblo]]
    nrhi = istar[irgb[nrgbhi]]

    nrhi = np.where(nrhi < len(istar)-1, nrhi, len(istar)-1)
    nrbins = len(nrhi)

    print 'nrgblo: ', nrgblo
    print 'nrgbhi: ', nrgbhi
    print 'nrlo: ', nrlo
    print 'nrhi: ', nrhi

    print 'Splitting into ', len(nrhi),' radial bins with ',nrgbstars,' in each upper RGB.'

    #

    # run once to get shape of CMD
    cmd, meancol, sigcol, cboundary, mboundary, extent = make_low_AV_cmd(c,m)
    
    # setup interpolation to convert pixel values of meancol into numerical values
    mcen = (mboundary[0:-1] + mboundary[1:]) / 2.0
    ccen = (cboundary[0:-1] + cboundary[1:]) / 2.0
    cinterp  = interp1d(np.arange(len(ccen)), ccen)
    deltapix =  [cboundary[1]-cboundary[0],mboundary[1]-mboundary[0]]

    # initialize storage arrays
    cmd_array = np.zeros([cmd.shape[0], cmd.shape[1], nrbins])
    meancol_array = np.zeros([meancol.shape[0], nrbins])
    sigcol_array = np.zeros([sigcol.shape[0], nrbins])
    meanr_array = np.zeros(nrbins)
    rrange_array = np.zeros([2,nrbins])
    n_array = np.zeros(nrbins)
    maglim_array = np.zeros([2,nrbins])

    # initialize magnitude limit polynomials
    completenessdir = '../../Completeness/'
    m110file = 'completeness_ra_dec.gst.F110W.npz'
    m160file = 'completeness_ra_dec.gst.F160W.npz'
        
    m110dat = np.load(completenessdir + m110file)
    m110polyparam = m110dat['param']
    m160dat = np.load(completenessdir + m160file)
    m160polyparam = m160dat['param']
        
    p110 = np.poly1d(m110polyparam)
    p160 = np.poly1d(m160polyparam)

    # initialize plot
    plt.figure(1)
    plt.clf()

    for i in range(len(nrlo)):
        
        cmd, meancol, sigcol, cboundary, mboundary, extent = make_low_AV_cmd(c[nrlo[i]:nrhi[i]],
                                                                             m[nrlo[i]:nrhi[i]])

        # normalize CMD to a constant # of stars for magnitudes in mnormalizerange
        irgb = np.where((mnormalizerange[0] < mboundary) & (mboundary < mnormalizerange[1]))[0]
        rgbrange = [min(irgb), max(irgb)]
        norm = np.sum(cmd[rgbrange[0]:rgbrange[1],:])
        #norm = np.sum(np.array([cmd[j,:] for j in range(len(mcen)-1) if ((mnormalizerange[0] < mboundary[j]) & 
        #                                                            (mboundary[j+1] < mnormalizerange[1]))]))
        nstars = np.sum(cmd)
        cmd = cmd.astype(float) / float(norm)

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

        print 'Bin: ', i, '  R: ', ("%g" % round(meanr_array[i],3)), 'maglim', ("%g" % round(maglim110,2)), ("%g" % round(maglim160,2)),' NStars: ',nstars, ' NRGB: ',norm


        # make a mask at the same radius, using completeness information in ../../Completeness
        if makemask:

            print 'making mask....'


        plt.imshow(cmd,  extent=extent, aspect='auto', interpolation='nearest', vmin=0, vmax = 0.1)
        plt.xlabel('F110W - F160W')
        plt.ylabel('F160W')
        plt.title('Major Axis: '+str(meanr_array[i]))
        plt.draw()

    return cmd_array, meanr_array, rrange_array, meancol_array, sigcol_array, n_array, maglim_array, mboundary, cboundary
        
        






