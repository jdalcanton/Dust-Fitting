import pylab
import math
import numpy as np
import scipy.special as spec
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import read_brick_data as rbd
import makefakecmd as mfc
import isolatelowAV as iAV
from scipy.ndimage import filters as filt
import scipy.special as special
from random import sample
import random as random
import copy as copy
import ezfig  # morgan's plotting code
import pyfits as pyfits
import pywcs as pywcs
import makefakecmd as mfc
import aplpy as aplpy
from astropy.io import fits
from astropy import wcs

def merge_results(savefilelist=['ir-sf-b14-v8-st.npz', 'newfit_test.npz'], 
                  resultsdir='../Results/', fileextension='',
                  mergefileroot='merged'):

    savefilelist = ['ir-sf-b02-v8-st', 'ir-sf-b04-v8-st', 'ir-sf-b05-v8-st', 
                    'ir-sf-b06-v8-st', 'ir-sf-b08-v8-st', 'ir-sf-b09-v8-st', 
                    'ir-sf-b12-v8-st', 'ir-sf-b14-v8-st', 'ir-sf-b15-v8-st', 
                    'ir-sf-b16-v8-st', 'ir-sf-b17-v8-st', 'ir-sf-b18-v8-st', 
                    'ir-sf-b19-v8-st', 'ir-sf-b21-v8-st', 'ir-sf-b22-v8-st', 
                    'ir-sf-b23-v8-st']
    savefilelist = [s + fileextension + '.npz' for s in savefilelist]
    mergefile = resultsdir + mergefileroot + fileextension + '.npz'
    pngfileroot = resultsdir + mergefileroot + fileextension

    # initialize ra-dec grid

    dat = np.load(resultsdir + savefilelist[0])
    ra_global = dat['ra_global'].flatten()
    dec_global = dat['dec_global'].flatten()

    nx = len(ra_global) - 1
    ny = len(dec_global) - 1

    print 'ddec: ', (dec_global[1]-dec_global[0])*3600.
    print 'dra:  ', (ra_global[1]-ra_global[0])*3600.

    nz_bestfit = 3
    nz_sigma = nz_bestfit * 3
    nz_derived = 8
    nz_derived_sigma = nz_derived * 3
    nz_quality = 2
    bestfit_values = np.zeros([nx, ny, nz_bestfit])
    percentile_values = np.zeros([nx, ny, nz_sigma])
    quality_values = np.zeros([nx, ny, nz_quality])
    derived_values = np.zeros([nx, ny, nz_derived])
    derived_percentile_values = np.zeros([nx, ny, nz_derived_sigma])

    # loop through list of files

    for i, savefile in enumerate(savefilelist):

        print 'Merging ', resultsdir + savefilelist[i]

        dat = np.load(resultsdir + savefilelist[i])
        bf = dat['bestfit_values']
        p  = dat['percentile_values']
        q  = dat['quality_values']
        der = dat['derived_values']
        derp = dat['derived_percentile_values']
        if (len(p.shape) == 4):
            p = p[0,:,:,:]
        if (len(q.shape) == 4):
            q = q[0,:,:,:]
        ra_g  = dat['ra_global'].flatten()
        dec_g = dat['dec_global'].flatten()

        try:
            ra_local = dat['ra_local']
            dec_local = dat['dec_local']
        except:
            ra_local = dat['ra_bins']         # to deal with old file that had different name...
            dec_local = dat['dec_bins']

        dat.close()

        nx, ny = bf[:,:,0].shape

        # verify that the global arrays equal the master

        if (np.array_equal(ra_g, ra_global) & np.array_equal(dec_g, dec_global)):

            # find starting location for copy
            i_x0 = np.where(ra_local[0]  == ra_global)[0][0]
            i_y0 = np.where(dec_local[0] == dec_global)[0][0]

            # copy to the right location

            #bestfit_values[i_x:i_x + nx, i_y:i_y + ny, :] = bf
            #percentile_values[i_x:i_x + nx, i_y:i_y + ny, :] = p
            #quality_values[i_x:i_x + nx, i_y:i_y + ny, :] = q
            
            # don't copy -666, and check that nstars in quality values is higher than existing one 
            # (i.e., for overlaps).

            for i_x in range(len(bf[:,0,0])):

                for i_y in range(len(bf[0,:,0])):

                    if (quality_values[i_x0 + i_x, i_y0 + i_y, 1] < q[i_x, i_y, 1]):

                        bestfit_values[   i_x0 + i_x, i_y0 + i_y, :] = bf[i_x, i_y, :]
                        percentile_values[i_x0 + i_x, i_y0 + i_y, :] = p[i_x, i_y, :]
                        quality_values[   i_x0 + i_x, i_y0 + i_y, :] = q[i_x, i_y, :]
                        derived_values[   i_x0 + i_x, i_y0 + i_y, :] = der[i_x, i_y, :]
                        derived_percentile_values[   i_x0 + i_x, i_y0 + i_y, :] = derp[i_x, i_y, :]
        
        else:

            print 'Global ra and dec do not agree for ', resultsdir + savefilelist[i]

    # make mask of likely bad fits
    datamask = np.where(bestfit_values[:,:,1] > 0, 1., 0.)
    datamask_bool = np.where(bestfit_values[:,:,1] > 0, True, False)

    likelihood_cut = -6.4
    median_filt_cut = 7

    bf_smooth = filt.median_filter(bestfit_values, size=(3,3,1))
    bad_pix_mask = np.where(bestfit_values[:,:,1] / bf_smooth[:,:,1] > median_filt_cut, 0., 1.)
    bestfit_values_clean = bestfit_values.copy()
    bestfit_values_clean[:,:,0] = bestfit_values[:,:,0] * bad_pix_mask + bf_smooth[:,:,0]*(1.-bad_pix_mask)
    bestfit_values_clean[:,:,1] = bestfit_values[:,:,1] * bad_pix_mask + bf_smooth[:,:,1]*(1.-bad_pix_mask)
    bestfit_values_clean[:,:,2] = bestfit_values[:,:,2] * bad_pix_mask + bf_smooth[:,:,2]*(1.-bad_pix_mask)

    der_smooth = filt.median_filter(derived_values, size=(3,3,1))
    bad_pix_mask = np.where(derived_values[:,:,1] / der_smooth[:,:,1] > median_filt_cut, 0., 1.)
    derived_values_clean = derived_values.copy()
    derived_values_clean[:,:,0] = derived_values[:,:,0] * bad_pix_mask + der_smooth[:,:,0]*(1.-bad_pix_mask)
    derived_values_clean[:,:,1] = derived_values[:,:,1] * bad_pix_mask + der_smooth[:,:,1]*(1.-bad_pix_mask)
    derived_values_clean[:,:,2] = derived_values[:,:,2] * bad_pix_mask + der_smooth[:,:,2]*(1.-bad_pix_mask)

    # helped a bit, but not much...
    #bad_pix_mask = np.where((bestfit_values[:,:,2] < 0.45 - 0.45*bestfit_values[:,:,1]) |
    #                     (quality_values[:,:,0] < likelihood_cut), 0., 1.)
    # following cut out way too many stars
    #bad_pix_mask = np.where(((bestfit_values[:,:,1] < percentile_values[:,:,3]) |
    #                      (bestfit_values[:,:,1] > percentile_values[:,:,5])),
    #                     0., 1.)
    # better, on the right track, but cut some real stuff as well...
    #bad_pix_mask = np.where(((percentile_values[:,:,5] - percentile_values[:,:,3]) > 
    #                      3.0 - 0.5 * bestfit_values),
    #                     0., 1.)
        
    np.savez(mergefile,
          bestfit_values = bestfit_values,
          percentile_values = percentile_values,
          quality_values = quality_values,
          derived_values = derived_values,
          derived_percentile_values = derived_percentile_values,
          bad_pix_mask = bad_pix_mask,
          bestfit_values_clean = bestfit_values_clean,
          dervied_values_clean = derived_values_clean,
          ra_bins = ra_global,
          dec_bins = dec_global,
          savefilelist = savefilelist)

    plt.figure(6)
    plt.clf()
    plt.imshow(bestfit_values[::-1,::-1,1].T,vmin=0,vmax=4,cmap='hot')

    plt.figure(7)
    plt.clf()
    im = plt.imshow(bestfit_values_clean[::-1,::-1,1].T,vmin=0,vmax=4,cmap='hot')
    plt.colorbar(im)
    
    plt.figure(8)
    plt.clf()
    im = plt.imshow(bestfit_values_clean[::-1,::-1,0].T,vmin=0,vmax=1,cmap='seismic')
    plt.colorbar(im)

    mfc.plot_bestfit_results(results_file = mergefile, brickname=mergefileroot, pngroot=pngfileroot)

    return bestfit_values, percentile_values, quality_values, bad_pix_mask, bestfit_values_clean


def plot_f_red_theory():

    xmax = 5.
    x = np.linspace(-xmax, xmax, 1000)
    sech_func = stats.hypsecant()
    
    beta_vec = np.linspace(0,1,5)
    offset_vec = 0.0*beta_vec

    plt.figure(1, figsize=(6,9))
    plt.clf()

    for i, beta in enumerate(beta_vec):

        if (beta != 0): 

            L = (2.0**beta * (sech_func.pdf(x / beta))**beta)

        else:

            L = np.exp(-abs(x))

        Lint = np.cumsum(L)

        # normalize...
        L = L / (Lint[-1] * (x[1] - x[0]))
        Lint = Lint / Lint[-1]

        # rescale to common half-light scale height
        Linterp = interp.interp1d(Lint,x)
        x0 = Linterp([0.75])[0]

        # renormalize PDF
        L = L * x0

        plt.subplot(2,1,1)
        plt.plot(x / x0, L)
        plt.axis([-3,3,0,0.4])

        plt.subplot(2,1,2)
        plt.plot(x / x0, Lint)
        plt.axis([-3,3,0,1])

        Linterp = interp.interp1d(Lint,x/x0)
        refx = 0.2
        x1 = Linterp([1-refx])[0]
        offset_vec[i] = x1
        print 'Beta: ', beta,'  X-rescale: ',x0, '  Reference Fraction: ', refx, ' z / z_{1/2}: ', x1
        plt.plot([-3,3], [refx,refx], color='black', alpha=0.3)
        plt.plot([-3,3], [1-refx,1-refx], color='black', alpha=0.3)

        plt.xlabel('$z / z_{1/2}$')

            
    # make plot of tipping angle for various r/z_1/2

    r_over_z = np.linspace(2,20,1000)

    theta = (offset_vec.mean() / r_over_z) * (180. / np.pi)

    plt.figure(2)
    plt.clf()
    
    plt.plot(r_over_z, theta, color='black')
    plt.xlabel('$r / z_{1/2}$')
    plt.ylabel(r'$ \theta $')

def plot_errors(dat, indices='', fignum='', clearfig=False, *args, **kwargs):

    if (fignum != ''):

        plt.figure(fignum, figsize = (14, 10))

    if (clearfig):
            
        plt.clf()

    f            = dat['bestfit_values'][:,:,0]
    A_V         = dat['bestfit_values'][:,:,1]
    #sig_A_V = dat['bestfit_values'][:,:,2]
    #sig = dat['bestfit_values'][:,:,2] * dat['bestfit_values'][:,:,1]
    sig = dat['bestfit_values'][:,:,2]
    sig_A_V = dat['bestfit_values'][:,:,2] / dat['bestfit_values'][:,:,1]
    stdvec = (dat['percentile_values'][:,:,[2,5,8]] - dat['percentile_values'][:,:,[0,3,6]]) / 2.0
    medvec = dat['percentile_values'][:,:,[1,4,7]]
    f_std = stdvec[:,:,0]
    f_med = medvec[:,:,0]
    A_V_std = stdvec[:,:,1]
    A_V_med = medvec[:,:,1]
    sig_std = stdvec[:,:,2]
    sig_med = medvec[:,:,2]
    #medvec = np.reshape(dat['percentile_values'][:,:,[1,4,7]], (-1, 3))
        
    if (indices != ''):

        f            = f[indices[0],indices[1]]
        A_V         = A_V[indices[0],indices[1]]
        sig_A_V = sig_A_V[indices[0],indices[1]]
        sig = sig[indices[0],indices[1]]
        f_std = f_std[indices[0],indices[1]]
        f_med = f_med[indices[0],indices[1]]
        A_V_std = A_V_std[indices[0],indices[1]]
        A_V_med = A_V_med[indices[0],indices[1]]
        sig_std = sig_std[indices[0],indices[1]]
        sig_med = sig_med[indices[0],indices[1]]

    # test plot

    plt.subplot(3,3,1)
    plt.plot(A_V, A_V_std, ',', **kwargs)
    plt.axis([-0.05, 4, -0.05, 4])
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta  A_V$')

    plt.subplot(3,3,2)
    plt.plot(A_V, A_V_med - A_V, ',', **kwargs)
    plt.axis([-0.05, 4, -4, 4])
    plt.xlabel('$A_V$')
    plt.ylabel(' Median $A_V$ - Bestfit $A_V$')

    plt.subplot(3,3,3)
    plt.plot(A_V_std, abs(A_V_med - A_V), ',', **kwargs)
    plt.plot([0,0],[4,4], color='red')
    plt.axis([-0.05, 4, -0.05, 4])
    plt.xlabel(' $\Delta  A_V$')
    plt.ylabel(' |Median $A_V$ - Bestfit $A_V$|')

    plt.subplot(3,3,4)
    plt.plot(f, f_std, ',', **kwargs)
    plt.axis([0, 0.8, 0, 0.3])
    plt.xlabel('$f_{red}$')
    plt.ylabel(' $\Delta  f_{red}$')

    plt.subplot(3,3,5)
    plt.plot(f, f_med - f, ',', **kwargs)
    plt.axis([0, 0.8, -0.8, 0.8])
    plt.xlabel('$f_{red}$')
    plt.ylabel(' Median $f_{red}$ - Bestfit $f_{red}$')

    plt.subplot(3,3,6)
    plt.plot(f_std, abs(f_med - f), ',', **kwargs)
    plt.plot([0,0],[4,4], color='red')
    plt.axis([-0.01, 0.25, -0.01, 0.5])
    plt.xlabel(' $\Delta  f_{red}$')
    plt.ylabel(' |Median $f_{red}$ - Bestfit $f_{red}$|')

    plt.subplot(3,3,7)
    plt.plot(sig, sig_std, ',', **kwargs)
    plt.axis([-0.05, 2, -0.05, 1])
    plt.xlabel('$\sigma / A_V$')
    plt.ylabel('$\Delta(\sigma / A_V)$')

    plt.subplot(3,3,8)
    plt.plot(sig, sig_med - sig, ',', **kwargs)
    plt.axis([-0.05, 2, -1, 1])
    plt.xlabel('$\sigma / A_V$')
    plt.ylabel(' Median $\sigma / A_V$ - Bestfit $\sigma / A_V$')

    plt.subplot(3,3,9)
    plt.plot(sig_std, abs(sig_med - sig), ',', **kwargs)
    plt.plot([0,0],[4,4], color='red')
    plt.axis([-0.05, 1, -0.05, 1])
    plt.xlabel('$\Delta(\sigma / A_V)$')
    plt.ylabel(' |Median $\sigma / A_V$ - Bestfit $\sigma / A_V$|')

    plt.draw()

def compare_results(resultsdir = '../Results/', makeplots=True, usefracstats=False):

    datfiles_25pc = ['ir-sf-b12-v8-st.dc0.015.dm02.da6.6.npz',
                     'ir-sf-b12-v8-st.dc0.015.dm03.da6.6.npz',
                     'ir-sf-b12-v8-st.dc0.02.dm02.da6.6.npz',
                     'ir-sf-b12-v8-st.dc0.025.dm01.da6.6.npz',
                     'ir-sf-b12-v8-st.dc0.025.dm02.da6.6.npz',
                     'ir-sf-b12-v8-st.maglim0.25shift.dc0.025.dm02.da6.6.npz',
                     'ir-sf-b12-v8-st.maglim0.5shift.dc0.025.dm02.da6.6.npz',
                     'ir-sf-b12-v8-st.dc0.025.dm03.da6.6.npz',
                     'ir-sf-b12-v8-st.dc0.03.dm02.da6.6.npz',
                     'ir-sf-b12-v8-st.dc0.035.dm02.da6.6.npz']
    dc_str   = ['0.015', '0.015', '0.02', '0.025', '0.025',  '0.025', '0.025', '0.025', '0.03', '0.035']
    dm_str   = ['0.2',   '0.3',   '0.2',  '0.1',   '0.2',    '0.2',   '0.2',   '0.3',   '0.2',  '0.2'  ]
    dmag_str = ['0',     '0',     '0',    '0',     '0',      '0.25',  '0.5',   '0',     '0',    '0'    ]
                
    datfiles_20pc = ['ir-sf-b12-v8-st.dc0.025.dm02.da5.3.npz',
                     'ir-sf-b12-v8-st.dc0.035.dm02.da5.3.npz']

    nameroot = ['A_V', 'f', 'sig_A_V']

    d = dict()
    d_f = dict()
    d_A_V = dict()
    d_sig_A_V = dict()
    d_sig = dict()
    d_err = dict()
    d_bad = dict()

    for i, resultsfile in enumerate(datfiles_25pc):

        dat = np.load(resultsdir + resultsfile)
        mod_str = dc_str[i] + '_' + dm_str[i] + '_' + dmag_str[i]
        d[mod_str] = dat
        d_f['f_' + mod_str]            = dat['bestfit_values'][:,:,0]
        d_A_V['A_V_' + mod_str]        = dat['bestfit_values'][:,:,1].flatten()
        d_sig_A_V['sig_A_V_' + mod_str] = dat['bestfit_values'][:,:,2].flatten()
        d_sig['sig_' + mod_str] = dat['bestfit_values'][:,:,2].flatten() * dat['bestfit_values'][:,:,1].flatten()
        d_err['std_' + mod_str] = np.reshape((dat['percentile_values'][:,:,[2,5,8]] - 
                                              dat['percentile_values'][:,:,[0,3,6]]) / 2.0, (-1, 3))
        d_err['med_' + mod_str] = np.reshape(dat['percentile_values'][:,:,[1,4,7]], (-1, 3))
        #d_f['f_' + mod_str]            = [f for f in dat['bestfit_values'][:,:,0].flatten() if f > 0]
        #d_A_V['A_V_' + mod_str]        = [A for A in dat['bestfit_values'][:,:,1].flatten() if A > 0]
        #d_sig_A_V['sig_A_V' + mod_str] = [s for s in dat['bestfit_values'][:,:,2].flatten() if s > 0]

        # initialize error plots
        #plot_errors(dat, fignum=i+1, alpha=0.1, color='blue', clearfig=True)
        #plt.suptitle(resultsfile)

        # initialize dictionary of bad values.
        d_bad[i] = np.empty((2,0), dtype=int)

    # calculate stats between pairs

    AV_lim = 0.3
    for j in range(len(dc_str)):

        for i in range(len(dc_str)):

            if i > j:

                mod_str_i = dc_str[i] + '_' + dm_str[i] + '_' + dmag_str[i]
                mod_str_j = dc_str[j] + '_' + dm_str[j] + '_' + dmag_str[j]
            
                f_diff = d_f['f_' + mod_str_j].flatten() - d_f['f_' + mod_str_i].flatten()
                A_V_diff = d_A_V['A_V_' + mod_str_j].flatten() - d_A_V['A_V_' + mod_str_i].flatten()
                
                f_avg = (d_f['f_' + mod_str_j].flatten() + d_f['f_' + mod_str_i].flatten()) / 2.0
                A_V_avg = (d_A_V['A_V_' + mod_str_j].flatten() + d_A_V['A_V_' + mod_str_i].flatten()) / 2.0

                i_good= np.where((d_A_V['A_V_' + mod_str_i].flatten() > 0) & (d_A_V['A_V_' + mod_str_j].flatten() > 0))
                i_lo = np.where((d_A_V['A_V_' + mod_str_i].flatten() < AV_lim) & 
                                (d_A_V['A_V_' + mod_str_i].flatten() > 0) & (d_A_V['A_V_' + mod_str_j].flatten() > 0))
                i_hi = np.where((d_A_V['A_V_' + mod_str_i].flatten() >= AV_lim) & 
                                (d_A_V['A_V_' + mod_str_i].flatten() > 0) & (d_A_V['A_V_' + mod_str_j].flatten() > 0))

                # calculate stats using interquartile widths and medians for robustness
                percvec = [16.0, 50.0, 84.0]  # 1 sigma
                percvec = [5., 50.0, 95.]   # Inner 90%
                percvec = [2.4, 50.0, 97.6]   # 2 sigma
                # straight difference
                f_diff_perc_good = np.percentile(f_diff[i_good],percvec)
                A_V_diff_perc_good = np.percentile(A_V_diff[i_good],percvec)
                # fractional difference
                if usefracstats:

                    f_diff_perc_good = np.percentile(f_diff[i_good] / f_avg[i_good], percvec)
                    A_V_diff_perc_good = np.percentile(A_V_diff[i_good] / A_V_avg[i_good], percvec)

                    f_diff_perc_lo = np.percentile(f_diff[i_lo] / f_avg[i_lo], percvec)
                    A_V_diff_perc_lo = np.percentile(A_V_diff[i_lo] / A_V_avg[i_lo], percvec)

                    f_diff_perc_hi = np.percentile(f_diff[i_hi] / f_avg[i_hi], percvec)
                    A_V_diff_perc_hi = np.percentile(A_V_diff[i_hi] / A_V_avg[i_hi], percvec)

                else: 

                    f_diff_perc_good = np.percentile(f_diff[i_good], percvec)
                    A_V_diff_perc_good = np.percentile(A_V_diff[i_good], percvec)

                    f_diff_perc_lo = np.percentile(f_diff[i_lo], percvec)
                    A_V_diff_perc_lo = np.percentile(A_V_diff[i_lo], percvec)

                    f_diff_perc_hi = np.percentile(f_diff[i_hi], percvec)
                    A_V_diff_perc_hi = np.percentile(A_V_diff[i_hi], percvec)

                f_diff_std_good = (f_diff_perc_good[2] - f_diff_perc_good[0]) / 4.
                A_V_diff_std_good = (A_V_diff_perc_good[2] - A_V_diff_perc_good[0]) / 4.
                
                f_diff_std_lo = (f_diff_perc_lo[2] - f_diff_perc_lo[0]) / 4.
                A_V_diff_std_lo = (A_V_diff_perc_lo[2] - A_V_diff_perc_lo[0]) / 4.
                
                f_diff_std_hi = (f_diff_perc_hi[2] - f_diff_perc_hi[0]) / 4.
                A_V_diff_std_hi = (A_V_diff_perc_hi[2] - A_V_diff_perc_hi[0]) / 4.
                
                print dc_str[i], dc_str[j], dm_str[i], dm_str[j], dmag_str[i], dmag_str[j], \
                    f_diff_perc_good[1], f_diff_perc_lo[1], f_diff_perc_hi[1], \
                    A_V_diff_perc_good[1], A_V_diff_perc_lo[1], A_V_diff_perc_hi[1], \
                    f_diff_std_good, f_diff_std_lo, f_diff_std_hi, \
                    A_V_diff_std_good, A_V_diff_std_lo, A_V_diff_std_hi, \
                    f_diff[i_good].mean(), f_diff[i_lo].mean(), f_diff[i_hi].mean(), \
                    A_V_diff[i_good].mean(), A_V_diff[i_lo].mean(), A_V_diff[i_hi].mean(), \
                    f_diff[i_good].std(), f_diff[i_lo].std(), f_diff[i_hi].std(), \
                    A_V_diff[i_good].std(), A_V_diff[i_lo].std(), A_V_diff[i_hi].std()

                # isolate pixels with bad agreement

                nsig = 4.5
                d1 = d[mod_str_i]
                d2 = d[mod_str_j]
                i_bad = np.array(np.where(((abs((d1['bestfit_values'][:,:,0]-d2['bestfit_values'][:,:,0]) - 
                                                f_diff_perc_good[1]) > nsig * f_diff_std_good) | 
                                           (abs((d1['bestfit_values'][:,:,1]-d2['bestfit_values'][:,:,1]) - 
                                                A_V_diff_perc_good[1]) > nsig * A_V_diff_std_good)) &
                                          (d1['bestfit_values'][:,:,0] > 0) & (d2['bestfit_values'][:,:,0] > 0)))
                print i, j, '  Number outliers: ', len(i_bad[0])

                # save bad values
                d_bad[i] = np.append(d_bad[i], i_bad, axis=1)
                d_bad[j] = np.append(d_bad[j], i_bad, axis=1)

                # overlay outliers
                #plt.figure(i+1)
                #plot_errors(d1, indices=i_bad, fignum=i+1, clearfig=False, alpha=1.0)
                #plt.figure(j+1)
                #plot_errors(d2, indices=i_bad, fignum=j+1, clearfig=False, alpha=1.0)

    # figure out which bad pixels appeared in multiple comparisons

    shiftfac = 100000
    multithresh = 3
    for i in range(len(dc_str)):

        i_b = shiftfac * d_bad[i][0] + d_bad[i][1]
        i_uniq = np.unique(i_b, return_inverse=True, return_index=True)
        n_uniq = np.array([len(np.where(i_uniq[2] == j)[0]) for j in range(len(i_uniq[0]))])
        i_multi = np.where(n_uniq > multithresh)[0]
        i_b_uniq = i_uniq[0][i_multi]
        i_b_0 = np.array((i_b_uniq / shiftfac).astype(int))
        i_b_1 = i_b_uniq - np.array((i_b_uniq / shiftfac).astype(int))
        print 'Orig: ', len(d_bad[i][0]), '# unique: ', len(i_uniq[0]), ' # multiple: ', len(i_multi),'  len(i_b_0): ',len(i_b_0)
        d_bad[i] = np.array([i_b_0, i_b_1])

        mod_str_i = dc_str[i] + '_' + dm_str[i] + '_' + dmag_str[i]
        d1 = d[mod_str_i]
        plot_errors(d1, fignum=i+1, clearfig=True, alpha=0.1, color='blue')
        plot_errors(d1, indices=i_bad, fignum=i+1, clearfig=False, alpha=1.0, color='red')

    if makeplots:

        # plot correlations

        plt.figure(1, figsize=(13,9))
        plt.clf()
        f_keylist = ['f_' + dc_str[i] + '_' + dm_str[i] + '_' + dmag_str[i] for i in range(len(dc_str))]
        ezfig.plotCorr(d_f, f_keylist, axisvec=[0, 0.6, 0, 0.6], alphaval=0.1)
        plt.draw()
        plt.savefig(resultsdir + 'corr_f.png',bbox_inches=0)
        
        plt.figure(2, figsize=(13,9))
        plt.clf()
        A_V_keylist = ['A_V_' + dc_str[i] + '_' + dm_str[i] + '_' + dmag_str[i] for i in range(len(dc_str))]
        ezfig.plotCorr(d_A_V, A_V_keylist, axisvec=[0, 4, 0, 4], alphaval=0.1)
        plt.draw()
        plt.savefig(resultsdir + 'corr_A_V.png',bbox_inches=0)
        
        plt.figure(3, figsize=(13,9))
        plt.clf()
        sig_A_V_keylist = ['sig_A_V_' + dc_str[i] + '_' + dm_str[i] + '_' + dmag_str[i] for i in range(len(dc_str))]
        ezfig.plotCorr(d_sig_A_V, sig_A_V_keylist, axisvec=[0, 1.75, 0, 2.5], alphaval=0.1)
        plt.draw()
        plt.savefig(resultsdir + 'corr_sig_A_V.png',bbox_inches=0)
        
        plt.figure(4, figsize=(13,9))
        plt.clf()
        sig_keylist = ['sig_' + dc_str[i] + '_' + dm_str[i] + '_' + dmag_str[i] for i in range(len(dc_str))]
        ezfig.plotCorr(d_sig, sig_keylist, axisvec=[0, 2.5, 0, 2.5], alphaval=0.1)
        plt.draw()
        plt.savefig(resultsdir + 'corr_sig.png',bbox_inches=0)
        
        # plot difference between two
        
        plt.figure(5, figsize=(13,9))
        plt.clf()
        ezfig.plotCorr(d_f, f_keylist, axisvec=[0, 0.6, -0.2, 0.2], alphaval=0.1, plotdiff=True)
        plt.draw()
        plt.savefig(resultsdir + 'corr_f_diff.png',bbox_inches=0)
        
        plt.figure(6, figsize=(13,9))
        plt.clf()
        ezfig.plotCorr(d_A_V, A_V_keylist, axisvec=[0, 4, -0.25, 0.25], alphaval=0.1, plotdiff=True)
        plt.draw()
        plt.savefig(resultsdir + 'corr_A_V_diff.png',bbox_inches=0)
        
        plt.figure(7, figsize=(13,9))
        plt.clf()
        ezfig.plotCorr(d_sig_A_V, sig_A_V_keylist, axisvec=[0, 1.75, -0.5, 0.5], alphaval=0.1, plotdiff=True)
        plt.draw()
        plt.savefig(resultsdir + 'corr_sig_A_V_diff.png',bbox_inches=0)
        
        plt.figure(8, figsize=(13,9))
        plt.clf()
        ezfig.plotCorr(d_sig, sig_keylist, axisvec=[0, 2.5, -0.25, 0.25], alphaval=0.1, plotdiff=True)
        plt.draw()
        plt.savefig(resultsdir + 'corr_sig_diff.png',bbox_inches=0)

    return d_bad

def reparse_data_file(datafile = 'ir-sf-b12-v8-st.fits', 
                      resultsfile = 'ir-sf-b12-v8-st.dc0.02.dm02.da6.6.npz', 
                      datadir='../../Data/', resultsdir='../Results/'):
    """
    Read extinction mapping results file, and data file it was generated from.
    Recreate indices map for ra, dec bin used.
    """

    #  read original stellar data

    m1, m2, ra, dec, nmatch, sharp1, sharp2, crowd1, crowd2, snr1, snr2  = rbd.read_mag_position_gst(datadir + datafile, return_nmatch=True, return_quality=True)
    m = np.array(m2)
    c = np.array(m1 - m2)
    ra = np.array(ra)
    dec = np.array(dec)
    nmatch = np.array(nmatch)

    # read extinction fitting results

    dat = np.load(resultsdir + resultsfile)
    ra_bins = dat['ra_bins']
    dec_bins = dat['dec_bins']

    f       = dat['bestfit_values'][:,:,0]
    A_V     = dat['bestfit_values'][:,:,1]
    sig_A_V = dat['bestfit_values'][:,:,2]

    # split stellar data into ra-dec bins

    indices, ra_bins, dec_bins = mfc.split_ra_dec(ra, dec, 
                                                  ra_bins=ra_bins,
                                                  dec_bins=dec_bins)

    return m, c, ra, dec, nmatch, indices, ra_bins, dec_bins, f, A_V, sig_A_V, sharp1, sharp2, crowd1, crowd2, snr1, snr2

def compare_overlaps(datafile = 'ir-sf-b12-v8-st.fits', 
                     resultsfile = 'ir-sf-b12-v8-st.dc0.02.dm02.da6.6.npz', 
                     datadir='../../Data/', resultsdir='../Results/'):

    m, c, ra, dec, nmatch, indices, ra_bins, dec_bins, f, A_V, sig_A_V, \
        sharp1, sharp2, crowd1, crowd2, snr1, snr2 = \
        reparse_data_file(datafile, resultsfile, datadir, resultsdir)

    rangevec = [max(dec_bins), min(dec_bins), max(ra_bins), min(ra_bins)]

    # calculate mean number of matches per bin
    n_per_bin = np.median([len(indices[x,y]) 
                           for (x,y),v in np.ndenumerate(indices) 
                           if (len(indices[x,y]) > 1)])
    n_thresh = n_per_bin - 7*np.sqrt(n_per_bin)
    
    # initialize number of matches

    n_matches = np.empty( indices.shape, dtype=float )
    emptyval = -666
    
    for (i,j), value in np.ndenumerate(indices):
        
        if (len(indices[i,j]) > n_thresh):
            
            n_matches[i, j] = np.mean(nmatch[indices[i, j]])
            
        else:
            
            n_matches[i, j] = emptyval
            
    # evaluate likely bad fits
            
    datamask = np.where(A_V > 0, 1., 0.)
    datamask_bool = np.where(A_V > 0, True, False)

    A_V_smooth = filt.median_filter(A_V, size=(3))
    A_V_deviance = A_V / A_V_smooth
    A_V_offset = (A_V - A_V_smooth) / A_V_smooth
    
    # group CMDs in groups of bad deviance

    i_good = []
    i_bad_2 = []
    i_bad_4 = []

    for (i,j), value in np.ndenumerate(indices):
     
        # if there's data and A_V is low
        if (A_V[i,j] > 0) & (A_V_smooth[i,j] < 0.2):

            if (A_V_deviance[i,j] < 2): 
                i_good.extend(indices[i,j])

            if (A_V_deviance[i,j] > 2.5) & (A_V_deviance[i,j] < 4): 
                i_bad_2.extend(indices[i,j])

            if (A_V_deviance[i,j] >= 4): 
                i_bad_4.extend(indices[i,j])

    n_good = len(i_good)
    n_bad_2 = len(i_bad_2)
    n_bad_4 = len(i_bad_4)
    print 'i_good: ', len(i_good)
    print 'i_bad_2: ', len(i_bad_2)
    print 'i_bad_4: ', len(i_bad_4)

    # make density maps of cmds.
    crange=[0.0, 3.] 
    mrange=[24.5, 17.5]
    nbins=[15,30]
    h_good, xedges, yedges = np.histogram2d(m[i_good], c[i_good], normed=True,
                                            range=[np.sort(mrange), crange], bins=nbins)
    h_bad_2, xedges, yedges = np.histogram2d(m[i_bad_2], c[i_bad_2], normed=True,
                                             range=[np.sort(mrange), crange], bins=nbins)
    h_bad_4, xedges, yedges = np.histogram2d(m[i_bad_4], c[i_bad_4], normed=True,
                                             range=[np.sort(mrange), crange], bins=nbins)
    h_rat_2 = h_bad_2 / h_good
    h_rat_4 = h_bad_4 / h_good
    cmd_extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]

    # plot stuff

    plt.figure(1)
    plt.clf()
     
    plt.subplot(2,2,1)
    im = plt.imshow(n_matches, vmin=1, vmax=2.5, interpolation='nearest', 
               aspect = 'auto', extent=rangevec)
    plt.colorbar(im)
    plt.title('Number of Matches')

    plt.subplot(2,2,2)
    im = plt.imshow(A_V_deviance, vmin=0, vmax=10, interpolation='nearest', 
               aspect = 'auto', extent=rangevec)
    plt.colorbar(im)
    plt.title('$A_V$ / $A_{V,smooth}$')
    
    plt.subplot(2,2,3)
    im = plt.imshow(A_V_offset, vmin=-0.5, vmax=0.5, interpolation='nearest', 
               aspect = 'auto', extent=rangevec)
    plt.colorbar(im)
    plt.title('$A_V$ - $A_{V,smooth}$')
    
    plt.subplot(2,2,4)
    im = plt.imshow(A_V, vmin=0, vmax=4, interpolation='nearest', 
               aspect = 'auto', extent=rangevec)
    plt.colorbar(im)
    plt.title('$A_V$')

    plt.figure(2)
    plt.clf()

    plt.subplot(2,2,1)
    plt.plot(n_matches, A_V_deviance, ',', alpha=0.05, color='b')
    plt.axis([0.5, 2, 0, 10])
    plt.xlabel('Average Number of Matches')
    plt.ylabel('$A_V$ / $A_{V,smooth}$')
    
    plt.subplot(2,2,2)
    plt.plot(n_matches, A_V_offset, ',', alpha=0.05, color='b')
    plt.axis([0.5, 2, -2, 2])
    plt.xlabel('Average Number of Matches')
    plt.ylabel('$(A_V - A_{V,smooth}) / A_{V,smooth}$')
    
    plt.subplot(2,2,3)
    plt.plot(A_V_smooth, A_V_deviance, ',', alpha=0.05, color='b')
    plt.axis([0.0, 2, 0, 10])
    plt.xlabel('$A_{V,smooth}$')
    plt.ylabel('$A_V$ / $A_{V,smooth}$')
    
    plt.subplot(2,2,4)
    plt.plot(A_V_smooth, A_V_offset, ',', alpha=0.05, color='b')
    plt.axis([0.0, 2, -2, 2])
    plt.xlabel('$A_{V,smooth}$')
    plt.ylabel('$(A_V - A_{V,smooth}) / A_{V,smooth}$')
    
    # CMDs
    plt.figure(3)
    plt.clf()

    alpha = 0.025
    plt.subplot(2,2,1)
    i_subsample = sample(i_good, n_bad_2)
    plt.plot(c[i_subsample],m[i_subsample],',',alpha=alpha,color='b')
    #plt.plot(c[i_good],m[i_good],',',alpha=alpha,color='b')
    plt.axis([0, 3, 24.5, 17])

    plt.subplot(2,2,3)
    i_subsample = sample(i_good, n_bad_2)
    plt.plot(c[i_good],m[i_good],',',alpha=0.1,color='b')
    plt.axis([0, 3, 24.5, 17])

    plt.subplot(2,2,2)
    plt.plot(c[i_bad_2],m[i_bad_2],',',alpha=alpha,color='b')
    plt.axis([0, 3, 24.5, 17])

    plt.subplot(2,2,4)
    plt.plot(c[i_bad_4],m[i_bad_4],',',alpha=alpha,color='b')
    plt.axis([0, 3, 24.5, 17])

    ####  CMD histograms
    plt.figure(4)
    plt.clf()
    
    plt.subplot(2,3,1)
    plt.imshow(np.log(h_good),  extent=cmd_extent, origin='upper', aspect='auto', 
               interpolation='nearest', vmin=-5, vmax=1)
    plt.xlabel('F110W - F160W')
    plt.ylabel('F160W')
    plt.title('$A_V$ / $A_{V,smooth}$ < 2')

    plt.subplot(2,3,2)
    plt.imshow(np.log(h_bad_2),  extent=cmd_extent, origin='upper', aspect='auto', 
               interpolation='nearest', vmin=-5, vmax=1)
    plt.xlabel('F110W - F160W')
    plt.ylabel('F160W')
    plt.title('$2.5 < A_V$ / $A_{V,smooth} < 4$')

    plt.subplot(2,3,3)
    plt.imshow(np.log(h_bad_4),  extent=cmd_extent, origin='upper', aspect='auto', 
               interpolation='nearest', vmin=-5, vmax=1)
    plt.xlabel('F110W - F160W')
    plt.ylabel('F160W')
    plt.title('$4 < A_V$ / $A_{V,smooth}$')

    plt.subplot(2,3,5)
    im = plt.imshow(h_rat_2,  extent=cmd_extent, origin='upper', aspect='auto', 
               interpolation='nearest', vmin=0, vmax = 2.0, cmap='seismic')
    plt.colorbar(im)
    plt.xlabel('F110W - F160W')
    plt.ylabel('F160W')
    plt.title('$(2.5 < A_V/A_{V,smooth} < 4)$ / $(A_V / A_{V,smooth} < 2)$')


    plt.subplot(2,3,6)
    im = plt.imshow(h_rat_4,  extent=cmd_extent, origin='upper', aspect='auto', 
               interpolation='nearest', vmin=0, vmax = 2.0, cmap='seismic')
    plt.colorbar(im)
    plt.xlabel('F110W - F160W')
    plt.ylabel('F160W')
    plt.title('$(4 < A_V/A_{V,smooth})$ / $(A_V / A_{V,smooth} < 2)$')
    plt.draw()

    ####  plot quality flags 
    plt.figure(5)
    plt.clf()

    
    # define outliers "o" and inliers 'i'
    ig_o = np.array(i_good)[np.where((c[i_good] > 1) & (m[i_good] > 21.5))[0]]
    ib2_o = np.array(i_bad_2)[np.where((c[i_bad_2] > 1) & (m[i_bad_2] > 21.5))[0]]
    ib4_o = np.array(i_bad_4)[np.where((c[i_bad_4] > 1) & (m[i_bad_4] > 21.5))[0]]
    ig_i = np.array(i_good)[np.where((c[i_good] < 1) & (m[i_good] > 21.5))[0]]
    ib2_i = np.array(i_bad_2)[np.where((c[i_bad_2] < 1) & (m[i_bad_2] > 21.5))[0]]
    ib4_i = np.array(i_bad_4)[np.where((c[i_bad_4] < 1) & (m[i_bad_4] > 21.5))[0]]
    snr = np.sqrt(snr1**2 + snr2**2)
    crowd = np.sqrt(crowd1**2 + crowd2**2)

    plt.subplot(3,3,1)
    plt.plot(sharp1[ib2_i],sharp2[ib2_i], ',', color='b', alpha=0.05)
    plt.plot(sharp1[ib4_i],sharp2[ib4_i], ',', color='b', alpha=0.05)
    plt.plot(sharp1[ib2_o],sharp2[ib2_o], ',', color='r')
    plt.plot(sharp1[ib4_o],sharp2[ib4_o], ',', color='r')
    plt.xlabel('Sharp 1')
    plt.ylabel('Sharp 2')
    
    plt.subplot(3,3,2)
    plt.plot(snr1[ib2_i],sharp1[ib2_i], ',', color='b', alpha=0.05)
    plt.plot(snr1[ib4_i],sharp1[ib4_i], ',', color='b', alpha=0.05)
    plt.plot(snr1[ib2_o],sharp1[ib2_o], ',', color='r')
    plt.plot(snr1[ib4_o],sharp1[ib4_o], ',', color='r')
    plt.xlabel('SNR 1')
    plt.ylabel('Sharp 1')
    
    plt.subplot(3,3,3)
    plt.plot(snr2[ib2_i], sharp2[ib2_i],',', color='b', alpha=0.05)
    plt.plot(snr2[ib4_i], sharp2[ib4_i],',', color='b', alpha=0.05)
    plt.plot(snr2[ib2_o], sharp2[ib2_o],',', color='r')
    plt.plot(snr2[ib4_o], sharp2[ib4_o],',', color='r')
    plt.xlabel('SNR 2')
    plt.ylabel('Sharp 2')
    
    plt.subplot(3,3,4)
    plt.plot(crowd1[ig_i],crowd2[ig_i], ',', color='b', alpha=0.05)
    plt.plot(crowd1[ib2_i],crowd2[ib2_i], ',', color='b', alpha=0.05)
    plt.plot(crowd1[ib4_i],crowd2[ib4_i], ',', color='b', alpha=0.05)
    plt.plot(crowd1[ig_o],crowd2[ig_o], ',', color='r')
    plt.plot(crowd1[ib2_o],crowd2[ib2_o], ',', color='r')
    plt.plot(crowd1[ib4_o],crowd2[ib4_o], ',', color='r')
    plt.axis([0, 2, 0, 2])
    plt.xlabel('Crowd 1')
    plt.ylabel('Crowd 2')
    
    plt.subplot(3,3,5)
    plt.plot(sharp1[ib2_i],crowd[ib2_i], ',', color='b', alpha=0.05)
    plt.plot(sharp1[ib4_i],crowd[ib4_i], ',', color='b', alpha=0.05)
    plt.plot(sharp1[ib2_o],crowd[ib2_o], ',', color='r')
    plt.plot(sharp1[ib4_o],crowd[ib4_o], ',', color='r')
    plt.axis([-0.05, 0.3, 0, 2])
    plt.xlabel('Sharp 1')
    plt.ylabel('Crowd')
    
    plt.subplot(3,3,6)
    plt.plot(snr1[ib2_i],crowd1[ib2_i], ',', color='b', alpha=0.05)
    plt.plot(snr1[ib4_i],crowd1[ib4_i], ',', color='b', alpha=0.05)
    plt.plot(snr1[ib2_o],crowd1[ib2_o], ',', color='r')
    plt.plot(snr1[ib4_o],crowd1[ib4_o], ',', color='r')
    plt.xlabel('SNR 1')
    plt.ylabel('Crowd 1')
    
    plt.subplot(3,3,7)
    plt.plot(sharp1[ib2_i],sharp1[ib2_i]-sharp2[ib2_i], ',', color='b', alpha=0.05)
    plt.plot(sharp1[ib4_i],sharp1[ib4_i]-sharp2[ib4_i], ',', color='b', alpha=0.05)
    plt.plot(sharp1[ib2_o],sharp1[ib2_o]-sharp2[ib2_o], ',', color='r')
    plt.plot(sharp1[ib4_o],sharp1[ib4_o]-sharp2[ib4_o], ',', color='r')
    plt.axis([-0.05, 0.3, -0.25, 0.3])
    plt.xlabel('Sharp 1')
    plt.ylabel('Sharp 1 - Sharp 2')
    
    plt.subplot(3,3,8)
    plt.plot(sharp2[ig_i],sharp1[ig_i]-sharp2[ig_i], ',', color='b', alpha=0.05)
    plt.plot(sharp2[ib2_i],sharp1[ib2_i]-sharp2[ib2_i], ',', color='b', alpha=0.05)
    plt.plot(sharp2[ib4_i],sharp1[ib4_i]-sharp2[ib4_i], ',', color='b', alpha=0.05)
    plt.plot(sharp2[ig_o],sharp1[ig_o]-sharp2[ig_o], ',', color='r')
    plt.plot(sharp2[ib2_o],sharp1[ib2_o]-sharp2[ib2_o], ',', color='r')
    plt.plot(sharp2[ib4_o],sharp1[ib4_o]-sharp2[ib4_o], ',', color='r')
    plt.axis([-0.05, 0.3, -0.25, 0.1])
    plt.xlabel('Sharp 2')
    plt.ylabel('Sharp 1 - Sharp 2')
    
    plt.subplot(3,3,9)
    plt.plot(crowd[ig_i],sharp1[ig_i]-sharp2[ig_i], ',', color='b', alpha=0.05)
    plt.plot(crowd[ib2_i],sharp1[ib2_i]-sharp2[ib2_i], ',', color='b', alpha=0.05)
    plt.plot(crowd[ib4_i],sharp1[ib4_i]-sharp2[ib4_i], ',', color='b', alpha=0.05)
    plt.plot(crowd[ig_o],sharp1[ig_o]-sharp2[ig_o], ',', color='r')
    plt.plot(crowd[ib2_o],sharp1[ib2_o]-sharp2[ib2_o], ',', color='r')
    plt.plot(crowd[ib4_o],sharp1[ib4_o]-sharp2[ib4_o], ',', color='r')
    plt.xlabel('Crowd')
    plt.ylabel('Sharp 1 - Sharp 2')
    plt.axis([0, 3, -0.25, 0.1])

    
    return i_good, i_bad_2, i_bad_4

# not certain of astrometry currently
# dat = np.load('../Results/merged.npz')
# AV = dat['bestfit_values_clean'][:,:,1]
# ra_bins = dat['ra_bins']
# dec_bins = dat['dec_bins']

def make_fits_image(fitsfilename, array, rabins, decbins, badval=-666, replaceval=0):

    f = pyfits.PrimaryHDU(data = array.T)
    hdr = f.header
    print 'Array: ', array.shape
    print 'RAbins, Decbins: ', rabins.shape, decbins.shape
    print 'NAXIS1, NAXIS2: ', hdr['NAXIS1'], hdr['NAXIS2']

    # replace bad values, if requested
    if (badval != ''):
        array = np.where(array == badval, replaceval, array)

    # default M31 parameters (from generate_global_ra_dec_grid)
    m31ra  = 10.6847929
    m31dec = 41.2690650    # use as tangent point

    # delta-degrees
    dra = (rabins[1] - rabins[0]) * np.cos(np.math.pi * m31dec / 180.0)
    ddec = decbins[1] - decbins[0]

    # middle of the image
    m31ra_mid = (rabins[-1] + rabins[0]) / 2.0
    m31dec_mid = (decbins[-1] + decbins[0]) / 2.0
    m31ra = m31ra_mid
    m31dec = m31dec_mid
    dra = (rabins[1] - rabins[0]) * np.cos(np.math.pi * m31dec / 180.0)
    ddec = decbins[1] - decbins[0]

    # position of pixel centers
    racenvec  = (rabins[0:-2]  +  rabins[1:-1]) / 2.0
    deccenvec = (decbins[0:-2] + decbins[1:-1]) / 2.0

    ra0 = m31ra
    dec0 = m31dec

    # interpolate to find pixel location of tangent point
    #    issues: FITS: pixels start at 1  (but correction applied to transpose?)
    #    not sure if ref pix number is for center of pixel or corner.
    #
    # if ra, dec at pixel corner
    i_ra = np.arange(0.,len(rabins)).astype(float)
    i_dec = np.arange(0.,len(decbins)).astype(float)
    x0 = np.interp(ra0, rabins, i_ra) + 1.0
    y0 = np.interp(dec0, decbins, i_dec) + 1.0
    print 'Array dimension: ', array.shape
    print 'Length in RA: ', rabins.shape
    print 'Length in Dec: ', decbins.shape

    print 'Reference RA, Dec: ', ra0, dec0
    print 'Reference X, Y:    ', x0, y0
    print 'dRA, dDec:         ', dra, ddec

    hdr.update('CTYPE1', 'RA---CAR')
    hdr.update('CTYPE2', 'DEC--CAR')
    #hdr.update('CTYPE1', 'RA---TAN')
    #hdr.update('CTYPE2', 'DEC--TAN')
    hdr.update('CRPIX1', x0,   'x reference pixel')
    hdr.update('CRPIX2', y0,   'y reference pixel')
    hdr.update('CRVAL1', ra0,  'RA  for first pixel')
    hdr.update('CRVAL2', dec0, 'Dec for first pixel')
    hdr.update('CD1_1',  dra,  'd(RA*cos(Dec))/dx')
    hdr.update('CD1_2',  0,   'd(RA*cos(Dec))/dy')
    hdr.update('CD2_1',  0,    'd(Dec)/dx')
    hdr.update('CD2_2',  ddec, 'd(Dec)/dy')
    #hdr.update('CD1_1',  0,  'd(RA*cos(Dec))/dx')
    #hdr.update('CD1_2',  dra,   'd(RA*cos(Dec))/dy')
    #hdr.update('CD2_1',  ddec,    'd(Dec)/dx')
    #hdr.update('CD2_2',  0, 'd(Dec)/dy')
    # to get Wells convention working...
    hdr.update('CDELT1', dra, 'd(RA*cos(Dec))/dx')
    hdr.update('CDELT2', ddec, 'd(Dec)/dy')
    ############
    #hdr.update('RA_TAN', m31ra, 'Tangent Point')
    #hdr.update('DEC_TAN', m31dec, 'Tangent Point')
    
    pyfits.writeto(fitsfilename, array.T, header=hdr)

    return

def make_radius_image(fitsfilename, rabins, decbins):

    # position of pixel centers
    racenvec  = (rabins[0:-2]  +  rabins[1:-1]) / 2.0
    deccenvec = (decbins[0:-2] + decbins[1:-1]) / 2.0

    dec, ra = np.meshgrid(deccenvec, racenvec)
    r = iAV.get_major_axis(ra, dec)

    make_fits_image(fitsfilename, r, rabins, decbins)
    #make_fits_image('ra_image.fits', ra, rabins, decbins)

    return r, ra, dec

def get_draine_at_lowAV(ratiofix = 2.3):

    drainefile = '../draine_M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.AV.fits'
    unreddenedfile = '../Unreddened/FourthRun15arcsec/allbricks.clean.npz'
    allunreddenedfile = '../Unreddened/FourthRun15arcsec/allbricks.npz'

    # read Draine file data
    print 'Opening Draine image...'
    f = pyfits.open(drainefile)
    hdr, draineimg = f[0].header, f[0].data

    # get ra dec of draine image
    wcs = pywcs.WCS(hdr)

    # read low reddening regions
    data = np.load(unreddenedfile)
    ra = data['ragrid']
    dec = data['decgrid']
    nstar = data['nstargrid']
    radius = data['rgrid']
    cmean = data['cmgrid']
    cstd = data['cstdgrid']

    # read low reddening regions
    dataall = np.load(allunreddenedfile)
    raall = dataall['ragridval']
    decall = dataall['decgridval']
    nstarall = dataall['nstargridval']
    radiusall = dataall['rgridval']
    cstdall = dataall['cstdgridval']
    cmeanall = dataall['cmeangridval']

    # get Draine pixel locations
    img_coords = wcs.wcs_sky2pix(ra, dec, 1)
    img_coords_all = wcs.wcs_sky2pix(raall, decall, 1)

    # grab values at those locations
    loAV = draineimg[np.rint(img_coords[0]).astype('int'),np.rint(img_coords[1]).astype('int')]
    loAVall = draineimg[np.rint(img_coords_all[0]).astype('int'),np.rint(img_coords_all[1]).astype('int')]
    
    # plot results

    plt.figure(1)
    plt.clf()
    plt.plot(np.log10(nstar), loAV, ',', color='blue')
    plt.xlabel(r'$\log_{10}N_{star}$')
    plt.ylabel('$A_{V,emission}$')

    plt.figure(2)
    plt.clf()
    plt.plot(radius, loAV, ',', color='blue')
    plt.xlabel('Radius (Degrees)')
    plt.ylabel('$A_{V,emission}$')

    plt.figure(3)
    plt.clf()
    plt.plot(np.log10(nstar), loAV / ratiofix, ',', color='blue')
    plt.xlabel(r'$\log_{10}N_{star}$')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)

    plt.figure(4)
    plt.clf()
    plt.plot(radius, loAV / ratiofix, ',', color='blue')
    plt.xlabel('Radius (Degrees)')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)

    plt.figure(5)
    plt.clf()
    plt.plot(radiusall, loAVall / ratiofix, ',', color='blue')
    plt.xlabel('Radius (Degrees)')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)

    plt.figure(6)
    plt.clf()
    im = plt.scatter(radiusall, loAVall / ratiofix, c=cstdall, s=7, linewidth=0, alpha=0.4, vmin=0.02, vmax=0.15)
    plt.colorbar(im)
    plt.xlabel('Radius (Degrees)')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)
    plt.suptitle('RGB Width')

    plt.figure(7)
    plt.clf()
    im = plt.scatter(np.log10(nstarall), loAVall / ratiofix, c=cstdall, s=7, linewidth=0, alpha=0.4, vmin=0.02, vmax=0.15)
    plt.colorbar(im)
    plt.xlabel(r'$\log_{10}N_{star}$')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)
    plt.suptitle('RGB Width')

    plt.figure(8)
    plt.clf()
    im = plt.scatter(radiusall, loAVall / ratiofix, c=cmeanall, s=7, linewidth=0, alpha=0.4, vmin=-0.1, vmax=0.01)
    plt.colorbar(im)
    plt.xlabel('Radius (Degrees)')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)
    plt.suptitle('Mean RGB color')

    plt.figure(9)
    plt.clf()
    im = plt.scatter(np.log10(nstarall), loAVall / ratiofix, c=cmeanall, s=7, linewidth=0, alpha=0.4, vmin=-0.1, vmax=0.01)
    plt.colorbar(im)
    plt.xlabel(r'$\log_{10}N_{star}$')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)
    plt.suptitle('Mean RGB color')

    plt.figure(10)
    plt.clf()
    im = plt.scatter(radius, loAV / ratiofix, c=cmean, s=7, linewidth=0, alpha=1.0, vmin=-0.1, vmax=0.01)
    plt.colorbar(im)
    plt.xlabel('Radius (Degrees)')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)
    plt.suptitle('Mean RGB color')

    plt.figure(11)
    plt.clf()
    im = plt.scatter(np.log10(nstar), loAV / ratiofix, c=cmean, s=7, linewidth=0, alpha=1.0, vmin=-0.1, vmax=0.01)
    plt.colorbar(im)
    plt.xlabel(r'$\log_{10}N_{star}$')
    plt.ylabel('$A_{V,emission} / %4.2f$' % ratiofix)
    plt.suptitle('Mean RGB color')

def derive_draine_bias_ratio(fileroot='merged', resultsdir='../Results/',
                             cleanstr = '_clean', imgexten='.png'):

    drainefile = '../draine_M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.AV.fits'
    draineresolution = 24.9   # FWHM
    AVresolution = 6.645      # FWHM
    output_smooth_AV_root = resultsdir + fileroot + '_interleaved_draine_smoothed.AV'
    output_smooth_meanAV_root = resultsdir + fileroot + '_interleaved_draine_smoothed.meanAV'
    output_smooth_AV_file = output_smooth_AV_root + '.fits'
    output_smooth_meanAV_file = output_smooth_meanAV_root + '.fits'
    output_chi2_AV_file = output_smooth_AV_root + '.AV_chi2' + imgexten
    output_chi2_meanAV_file = output_smooth_AV_root + '.meanAV_chi2' + imgexten
        
    # median extinction
    arrayname = 'bestfit_values' + cleanstr
    arraynum = 1
    a, ra_bins, dec_bins = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                                           arrayname=arrayname, 
                                           arraynum=arraynum)
    AV = a

    print 'Size of extinction map: ', AV.shape

    # mean extinction
    if (fileroot == 'merged'):
        arrayname = 'dervied_values' + cleanstr  #  note mispelling!
    else:
        arrayname = 'derived_values' + cleanstr
    arraynum = 0
    a, ra_bins, dec_bins = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                                           arrayname=arrayname, 
                                           arraynum=arraynum)
    meanAV = a

    # mask indicating no data regions for original input

    maskorig = np.where(AV > 0, 1.0, 0.0)

    # run the smoothing algorithm on both A_V maps

    # read from existing files
    print 'Opening existing smoothed images ', output_smooth_AV_file,' and ',output_smooth_meanAV_file
    f = pyfits.open(drainefile)
    draineimg = f[0].data
    f = pyfits.open(drainefile)
    hdr, draineimg = f[0].header, f[0].data
    f = pyfits.open(output_smooth_AV_file)
    AV_img = f[0].data
    f = pyfits.open(output_smooth_meanAV_file)
    meanAV_img = f[0].data

    # get ra dec of draine image
    f = pyfits.open(drainefile)
    hdr = f[0].header
    wcs = pywcs.WCS(hdr)
    i_dec, i_ra = np.meshgrid(np.arange(draineimg.shape[1]), np.arange(draineimg.shape[0]))
    i_coords = np.array([[i_dec[i,j],i_ra[i,j]] for (i,j),val in np.ndenumerate(i_ra)])
    print 'draineimg.shape: ', draineimg.shape
    print 'i_dec.shape: ', i_dec.shape
    print 'i_coords.shape: ', i_coords.shape
    # solve for RA, dec at those coords
    img_coords = wcs.wcs_pix2sky(i_coords, 1)
    img_coords = np.reshape(img_coords,(i_ra.shape[0],i_ra.shape[1],2))
    ra  = img_coords[:,:,0]
    dec = img_coords[:,:,1]

    # initialize machinery for solving for bias

    mask = np.where(AV_img > 0, 1.0, 0.0)
    i_unmasked = np.where(mask > 0)
    i_masked = np.where(mask == 0)

    AVratio = draineimg / AV_img 
    meanAVratio = draineimg / meanAV_img
    # mask out no data  regions (robust to divide by zero)
    AVratio[i_masked] = 0.0
    meanAVratio[i_masked] = 0.0

    # set minimum AV to use in the fit

    AVlim = 0.5

    i_good = np.where(AV_img > AVlim)

    # Estimate uncertainty in ratio at each A_V

    print 'Calculating uncertainties...'
    AVbins = np.linspace(0.25, 4.0, 31)
    i_AVbins = np.digitize(AV_img[i_good], AVbins) - 1
    AVindexvec = np.empty(len(AVbins), dtype=object)
    print 'Len(AVindexvec): ', len(AVindexvec)
    for i in range(len(AVbins)):
        AVindexvec[i] = []
    for i in range(len(i_good[0])):
        AVindexvec[i_AVbins[i]].append(i)

    # Calculate dispersion in each bin
    AVdispersionbinned = np.zeros(len(AVbins)-1)
    meanAVdispersionbinned = np.zeros(len(AVbins)-1)
    for i in range(len(AVdispersionbinned)):
        if len(AVindexvec[i]) > 1:
            AVdispersionbinned[i] = np.std(AVratio[i_good[0][AVindexvec[i]],
                                                   i_good[1][AVindexvec[i]]])
            meanAVdispersionbinned[i] = np.std(meanAVratio[i_good[0][AVindexvec[i]],
                                                           i_good[1][AVindexvec[i]]])

    # fit a polynomial to the dispersion vector
    npoly = 5
    AVbincen = (AVbins[:-1] + AVbins[1:]) / 2.0
    i_ok = np.where(AVdispersionbinned > 0)
    AV_dispersion_param = np.polyfit(AVbincen[i_ok], 
                                     AVdispersionbinned[i_ok], npoly)
    meanAV_dispersion_param = np.polyfit(AVbincen[i_ok], 
                                         meanAVdispersionbinned[i_ok], npoly)
    p_AVratio_dispersion = np.poly1d(AV_dispersion_param)
    p_meanAVratio_dispersion = np.poly1d(meanAV_dispersion_param)

    # Populate dispersion vector
    AVratiodispersion = np.zeros(AVratio.shape)
    meanAVratiodispersion = np.zeros(AVratio.shape)
    AVratiodispersion[i_good] = p_AVratio_dispersion(AV_img[i_good])
    meanAVratiodispersion[i_good] = p_meanAVratio_dispersion(AV_img[i_good])

    plt.figure(1)
    plt.clf()
    plt.plot(AV_img[i_good], AVratiodispersion[i_good], ',', color='blue') 
    plt.plot(AVbincen, AVdispersionbinned, '*', color='red') 
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\widetilde{A_V}$ Ratio')
    
    plt.figure(2)
    plt.clf()
    plt.plot(AV_img[i_good], meanAVratiodispersion[i_good], ',', color='blue') 
    plt.plot(AVbincen, meanAVdispersionbinned, '*', color='red') 
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel(r'$\langle A_V \rangle$ Ratio')
    
    # Generate grid of possible bias, ratio corrections, first for medain AV, then mean

    nbias = 101.
    nR = 76.
    biasrange = [-0.25, 1.25]
    Rrange = [0.5, 3.0]
    biasvec = np.linspace(biasrange[0], biasrange[1], nbias)
    Rvec = np.linspace(Rrange[0], Rrange[1], nR)

    #  lgnstar vec [-1.07342892 -0.85899891 -0.77736519 -0.69294421 -0.60640575 -0.5248202
    #   -0.45251709 -0.38013821 -0.28686579 -0.18048609]

    # First median AV....

    chi2vec = np.zeros((nbias, nR))
    for i, R in enumerate(Rvec):

        for j, bias in enumerate(biasvec):

            diff = AVratio[i_good] - R*(1. + bias/AV_img[i_good])
            chi2 = (diff / AVratiodispersion[i_good])**2

            chi2vec[j, i] = np.sum(chi2) / len(chi2)

    minchi2 = np.min(chi2vec)
    i_minchi2 = np.where(chi2vec == minchi2)   # there's a way to do this with np.argmin...
    minRglobal, minbiasglobal = Rvec[i_minchi2[1][0]], biasvec[i_minchi2[0][0]]

    print 'Best Fit for whole distribution using median AV: ',minRglobal, minbiasglobal
    
    print 'Plotting Global Chi2'

    dchi_max = 1.0

    print 'Increasing font size...'
    font = {'weight': '500',
            'size': '18'}
    plt.rc('font', **font)

    plt.figure(30)
    plt.close()
    plt.figure(30, figsize=(12,10))
    im = plt.imshow(chi2vec,  vmax=np.min(chi2vec) + dchi_max,
                    extent=[Rrange[0], Rrange[1], biasrange[0], biasrange[1]], 
                    aspect='auto', origin='lower',
                    cmap='gist_ncar', interpolation='nearest')
    plt.colorbar(im)
    plt.axis([Rrange[0], Rrange[1], biasrange[0], biasrange[1]])
    plt.plot(minRglobal, minbiasglobal, '*', markersize=20, color='white')
    plt.xlabel('$A_V$ Ratio (Emission / CMD)')
    plt.ylabel(r'$\widetilde{A_V}$ Bias')
    plt.annotate('Fitting $\widetilde{A_V} > %4.2f$' % AVlim,
                     xy=(0.95, 0.90), fontsize=20, horizontalalignment='right',
                     xycoords = 'axes fraction')
    plt.savefig(output_chi2_AV_file, bbox_inches=0)
    
    # ...then mean AV....

    chi2vec = np.zeros((nbias, nR))
    for i, R in enumerate(Rvec):

        for j, bias in enumerate(biasvec):

            diff = meanAVratio[i_good] - R*(1. + bias/meanAV_img[i_good])
            chi2 = (diff / meanAVratiodispersion[i_good])**2

            chi2vec[j, i] = np.sum(chi2) / len(chi2)

    minchi2 = np.min(chi2vec)
    i_minchi2 = np.where(chi2vec == minchi2)   # there's a way to do this with np.argmin...
    minRglobal, minbiasglobal = Rvec[i_minchi2[1][0]], biasvec[i_minchi2[0][0]]

    print 'Best Fit for whole distribution using mean AV: ',minRglobal, minbiasglobal
    
    print 'Plotting Global Chi2'

    dchi_max = 1.0

    plt.figure(3)
    plt.close()
    plt.figure(3, figsize=(12,10))
    im = plt.imshow(chi2vec,  vmax=np.min(chi2vec) + dchi_max,
                    extent=[Rrange[0], Rrange[1], biasrange[0], biasrange[1]], 
                    aspect='auto', origin='lower',
                    cmap='gist_ncar', interpolation='nearest')
    plt.colorbar(im)
    plt.axis([Rrange[0], Rrange[1], biasrange[0], biasrange[1]])
    plt.plot(minRglobal, minbiasglobal, '*', markersize=20, color='white')
    plt.xlabel('$A_V$ Ratio (Emission / CMD)')
    plt.ylabel(r'$\langle A_V \rangle$ Bias')
    plt.annotate('Fitting $\widetilde{A_V} > %4.2f$' % AVlim,
                     xy=(0.95, 0.90), fontsize=20, horizontalalignment='right',
                     xycoords = 'axes fraction')
    plt.savefig(output_chi2_meanAV_file, bbox_inches=0)

    # restore font size
    print 'Restoring original font defaults...'
    plt.rcdefaults()

    
    # set up info to analyze bias as a function of ln-nstar

    nlgnstarbins = 16.
    nsubplot = 4
    nlgnstarbins = 25.
    nsubplot = 5
    #nlgnstarbins = 36.
    #nsubplot = 6
    lgnstar = np.log10(iAV.get_nstar_at_ra_dec(ra, dec, renormalize_to_surfdens=True))
    minlgnstar = min(lgnstar[i_good])
    maxlgnstar = 0   # artificial, but inner parts are crap
    lgnstaredgevec = np.linspace(minlgnstar, maxlgnstar, nlgnstarbins+1)
    lgnstarvec = (lgnstaredgevec[:-1] + lgnstaredgevec[1:]) / 2.0
    chi2array = np.zeros((nbias, nR, nlgnstarbins))
    minRvec = np.zeros(nlgnstarbins)
    minbiasvec = np.zeros(nlgnstarbins)

    for k, lgnstarval in enumerate(lgnstarvec):

        i_keep = np.where((lgnstaredgevec[k] < lgnstar) &
                          (lgnstar <= lgnstaredgevec[k+1]) &
                          (AV_img > AVlim))

        chi2tmp = np.zeros((nbias, nR))

        if (len(i_keep) > 0):

            for i, R in enumerate(Rvec):
            
                for j, bias in enumerate(biasvec):
                
                    diff = meanAVratio[i_keep] - R*(1. + bias/meanAV_img[i_keep])
                    chi2 = (diff / meanAVratiodispersion[i_keep])**2
                
                    chi2tmp[j, i] = np.sum(chi2) / len(chi2)
                    chi2array[j, i, k] = np.sum(chi2) / len(chi2)
                    
            minchi2 = np.min(chi2tmp)
            i_minchi2 = np.where(chi2tmp == minchi2)   # there's a way to do this with np.argmin...
            minR, minbias = Rvec[i_minchi2[1][0]], biasvec[i_minchi2[0][0]]
            minRvec[k] = minR
            minbiasvec[k] = minbias
            print 'Best Fit for lgn ',lgnstarval,' is ',minR, minbias

        else:

            minRvec[k] = 0
            minbiasvec[k] = 0
            print 'No data in fitting range for lgn ',lgnstarval,' is ',minR, minbias

        
    # plot ratio, bias chi2 distributions as a function of lgnstar

    print 'Plotting Figure 4'
    plt.figure(4)
    plt.close()
    plt.figure(4, figsize=(13,10))

    for i, lgnstarval in enumerate(lgnstarvec):

        plt.subplot(nsubplot, nsubplot, i + 1)
        im = plt.imshow(chi2array[:,:,i], 
                        vmax=np.min(chi2array[:,:,i]) + dchi_max,
                        extent=[Rrange[0], Rrange[1], biasrange[0], biasrange[1]], 
                        aspect='auto', origin='lower',
                        cmap='gist_ncar', interpolation='nearest')
        plt.plot(minRglobal, minbiasglobal, '*', markersize=20, color='red')
        plt.plot(minRvec[i], minbiasvec[i], '*', markersize=20, color='white')

        plt.annotate(r'$\log_{10} N_{star} = %5.2f$' % lgnstarvec[i], 
                     xy=(0.95, 0.90), fontsize=10, horizontalalignment='right',
                     xycoords = 'axes fraction')

    plt.suptitle('Fitting $\widetilde{A_V} > %5.3f$' % AVlim)
 
   # plot ratio, bias solutions distributions as a function of lgnstar

    print 'Plotting Figure 5'
    plt.figure(5)
    plt.close()
    plt.figure(5, figsize=(13,10))

    AVvec = np.linspace(0,5,50)
    greyval = '#B3B3B3'

    for i, lgnstarval in enumerate(lgnstarvec):

        i_keep = np.where((lgnstaredgevec[i] < lgnstar) &
                          (lgnstar <= lgnstaredgevec[i+1]) &
                          (AV_img > AVlim))
        i_all = np.where((lgnstaredgevec[i] < lgnstar) &
                         (lgnstar <= lgnstaredgevec[i+1]) &
                         (AV_img > 0.25))

        plt.subplot(nsubplot, nsubplot, i + 1)
        plt.plot(meanAV_img[i_all], meanAVratio[i_all], ',', color=greyval) 
        plt.plot(meanAV_img[i_keep], meanAVratio[i_keep], ',', color='black') 
        plt.plot(AVvec, (1. + minbiasglobal/AVvec) * minRglobal, color='blue')
        plt.plot(AVvec, (1. + minbiasvec[i]/AVvec) * minRvec[i], color='red', )
        plt.plot(AVvec, 2.4 + 0.0*AVvec, color='green')
        plt.axis([0, 1.025*max(meanAV_img[i_keep]), 0, 7])
        plt.annotate(r'$\log_{10} N_{star} = %5.2f$' % lgnstarvec[i], 
                     xy=(0.95, 0.90), fontsize=10, horizontalalignment='right',
                     xycoords = 'axes fraction')
        plt.annotate(r'${\rm Bias} = %4.2f$' % minbiasvec[i], 
                     xy=(0.95, 0.15), fontsize=10, horizontalalignment='right',
                     xycoords = 'axes fraction')
        plt.annotate(r'${\rm Ratio} = %4.2f$' % minRvec[i], 
                     xy=(0.95, 0.05), fontsize=10, horizontalalignment='right',
                     xycoords = 'axes fraction')


    print 'Plotting Figure 6'
    plt.figure(6)
    plt.close()
    plt.figure(6, figsize=(10,10))
    plt.plot(lgnstarvec, minbiasvec)
    plt.xlabel(r'$\log_{10} N_{star}$')
    plt.ylabel('Bias')

    print 'Plotting Figure 7'
    plt.figure(7)
    plt.close()
    plt.figure(7, figsize=(10,10))
    plt.plot(lgnstarvec, minRvec)
    plt.xlabel(r'$\log_{10} N_{star}$')
    plt.ylabel('Ratio')

    # adjust images by the global solutions

    AV_fix = maskorig * (AV + minbiasglobal)
    meanAV_fix = maskorig * (meanAV + minbiasglobal)
    AV_img_fix = mask * (AV_img + minbiasglobal)
    meanAV_img_fix = mask * (meanAV_img + minbiasglobal)
    draineimg_fix = mask * (draineimg / minRglobal)
    meanAVratio_fix = mask * (draineimg_fix / meanAV_img_fix)

    # regenerate smoothed image with debiased input AV maps, for a few different 
    # smoothing parameters to test if residuals go down.

    test_resolution = 'False'
    if (test_resolution == 'True'):

        wcs, im_coords, draineimg2, meanAV_img_08 = compare_img_to_AV(meanAV_fix, ra_bins, dec_bins, drainefile, 
                                                                      crop='True',
                                                                      scaleimgfactor = 1.0,
                                                                      resolution_in_arcsec=draineresolution*0.8, 
                                                                      AV_resolution_in_arcsec=AVresolution, 
                                                                      outputAVfile='')
        wcs, im_coords, draineimg2, meanAV_img_09 = compare_img_to_AV(meanAV_fix, ra_bins, dec_bins, drainefile, 
                                                                      crop='True',
                                                                      scaleimgfactor = 1.0,
                                                                      resolution_in_arcsec=draineresolution*0.9, 
                                                                      AV_resolution_in_arcsec=AVresolution, 
                                                                      outputAVfile='')
        wcs, im_coords, draineimg2, meanAV_img_11 = compare_img_to_AV(meanAV_fix, ra_bins, dec_bins, drainefile, 
                                                                      crop='True',
                                                                      scaleimgfactor = 1.0,
                                                                      resolution_in_arcsec=draineresolution*1.1, 
                                                                      AV_resolution_in_arcsec=AVresolution, 
                                                                      outputAVfile='')
        wcs, im_coords, draineimg2, meanAV_img_12 = compare_img_to_AV(meanAV_fix, ra_bins, dec_bins, drainefile, 
                                                                      crop='True',
                                                                      scaleimgfactor = 1.0,
                                                                      resolution_in_arcsec=draineresolution*1.2, 
                                                                      AV_resolution_in_arcsec=AVresolution, 
                                                                      outputAVfile='')
        wcs, im_coords, draineimg2, meanAV_img_13 = compare_img_to_AV(meanAV_fix, ra_bins, dec_bins, drainefile, 
                                                                      crop='True',
                                                                      scaleimgfactor = 1.0,
                                                                      resolution_in_arcsec=draineresolution*1.3, 
                                                                      AV_resolution_in_arcsec=AVresolution, 
                                                                      outputAVfile='')
        wcs, im_coords, draineimg2, meanAV_img_14 = compare_img_to_AV(meanAV_fix, ra_bins, dec_bins, drainefile, 
                                                                      crop='True',
                                                                      scaleimgfactor = 1.0,
                                                                      resolution_in_arcsec=draineresolution*1.4, 
                                                                      AV_resolution_in_arcsec=AVresolution, 
                                                                      outputAVfile='')
        wcs, im_coords, draineimg2, meanAV_img_15 = compare_img_to_AV(meanAV_fix, ra_bins, dec_bins, drainefile, 
                                                                      crop='True',
                                                                      scaleimgfactor = 1.0,
                                                                      resolution_in_arcsec=draineresolution*1.5, 
                                                                      AV_resolution_in_arcsec=AVresolution, 
                                                                      outputAVfile='')
        i_unmasked_orig = i_unmasked
        mask = np.where(meanAV_img_12 > 0.25, 1.0, 0.0)  # set highish threshold to avoid bias offset
        i_unmasked = np.where(mask > 0)
        i_masked = np.where(mask == 0)
        rat_08 = mask * draineimg2 / meanAV_img_08
        rat_09 = mask * draineimg2 / meanAV_img_09
        #rat_10 = np.where(meanAV_img > 0.25, 1.0, 0.0) * draineimg / meanAV_img_fix
        rat_11 = mask * draineimg2 / meanAV_img_11
        rat_12 = mask * draineimg2 / meanAV_img_12
        rat_13 = mask * draineimg2 / meanAV_img_13
        rat_14 = mask * draineimg2 / meanAV_img_14
        rat_15 = mask * draineimg2 / meanAV_img_15
        
        print ' Smooth Fac     Mean    Median    Stddev  '
        print '0.8', np.mean(rat_08[i_unmasked]), np.median(rat_08[i_unmasked]), np.std(rat_08[i_unmasked])
        print '0.9', np.mean(rat_09[i_unmasked]), np.median(rat_09[i_unmasked]), np.std(rat_09[i_unmasked])
        #print '1.0', np.mean(rat_10[i_unmasked_orig]), np.median(rat_10[i_unmasked_orig]), np.std(rat_10[i_unmasked_orig])
        print '1.1', np.mean(rat_11[i_unmasked]), np.median(rat_11[i_unmasked]), np.std(rat_11[i_unmasked])
        print '1.2', np.mean(rat_12[i_unmasked]), np.median(rat_12[i_unmasked]), np.std(rat_12[i_unmasked])
        print '1.3', np.mean(rat_13[i_unmasked]), np.median(rat_13[i_unmasked]), np.std(rat_13[i_unmasked])
        print '1.4', np.mean(rat_14[i_unmasked]), np.median(rat_14[i_unmasked]), np.std(rat_14[i_unmasked])
        print '1.5', np.mean(rat_15[i_unmasked]), np.median(rat_15[i_unmasked]), np.std(rat_15[i_unmasked])
        
        plt.figure(8)
        plt.clf
        plt.imshow(rat_08 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')
        
        plt.figure(8)
        plt.clf
        plt.imshow(rat_08 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')
        
        plt.figure(9)
        plt.clf
        plt.imshow(rat_09 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')
        
        plt.figure(10)
        plt.clf
        #plt.imshow(rat_10 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')
        
        plt.figure(11)
        plt.clf
        plt.imshow(rat_11 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')
        
        plt.figure(12)
        plt.clf
        plt.imshow(rat_12 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')
        
        plt.figure(13)
        plt.clf
        plt.imshow(rat_13 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')
        
        plt.figure(14)
        plt.clf
        plt.imshow(rat_15 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')
        
        plt.figure(16)
        plt.clf
        plt.imshow(rat_16 / minRglobal, vmin=0, vmax=2, cmap='seismic', interpolation='nearest')

    return meanAV_img, draineimg, meanAVratio, meanAV_img_fix, draineimg_fix, meanAVratio_fix

    # Scatter plot of ratio, color coded by lg10 Nstar

    plt.figure(21)
    plt.close()
    plt.figure(21, figsize=(10,10))
    im = plt.plot(meanAV_img[i_loAV], draineimg[i_loAV] / meanAV_img[i_loAV], ',', 
                  color=greyval)
    # shuffling such a large array is too slow!
    #i_hiAV_shuffle = np.array(copy.deepcopy(i_hiAV))
    #random.shuffle(i_hiAV_shuffle)
    im = plt.scatter(meanAV_img[i_hiAV], 
                     draineimg[i_hiAV] / meanAV_img[i_hiAV], 
                     c=lgnstar[i_hiAV], 
                     cmap='jet_r', s=7, linewidth=0, vmin=-1.4, vmax=0.4)
    plt.colorbar(im)
    plt.xlabel(r'$\langle A_V \rangle$')
    plt.ylabel(r'$A_{V,emission}  /  \langle A_V \rangle$')
    plt.axis([0, 3.5, 0, 7])


    # plot bias results

    plt.figure(10)
    plt.close()
    plt.figure(10, figsize=(10,10))
    
    for i, AVlim in enumerate(Rvec):
        plt.plot(biasvec, (stddevvec / meanvec)[:,i], linewidth=1+2*i)
        plt.plot(minbiasvec[i], (stddevvec / meanvec)[i_minbiasvec[i],i], 'o', color='black')

    plt.xlabel('Bias')
    plt.ylabel(r'$\sigma_{\rm Ratio}  /  {\rm Ratio}$')

    plt.figure(11)
    plt.close()
    plt.figure(11, figsize=(10,10))
    
    for i, AVlim in enumerate(Rvec):
        plt.plot(biasvec, medianvec[:,i], linewidth=1+2*i)
        plt.plot(minbiasvec[i], medianvec[i_minbiasvec[i],i], 'o', color='black')
        print 'AVlim: ',AVlim, ' Minimum Stddev/Ratio at bias = ', biasvec[i_minstd],'  Ratio: ',medianvec[i_minbiasvec[i],i]

    plt.xlabel('Bias')
    plt.ylabel('Median Ratio')

    return

def plot_final_draine_compare(fileroot='merged', resultsdir='../Results/',
                              cleanstr = '_clean', imgexten='.png',
                              write_fits = 'False', write_ratio_fits = 'True',
                              biasfix = 0.26, ratiofix=2.37,
                              biasfixmean = 0.275, ratiofixmean=2.20,
                              smooth_img=0):

    drainefile = '../draine_M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.AV.fits'
    draineresolution = 24.9   # FWHM
    AVresolution = 6.645      # FWHM
    output_smooth_AV_root = resultsdir + fileroot + '_interleaved_draine_smoothed.AV'
    output_smooth_meanAV_root = resultsdir + fileroot + '_interleaved_draine_smoothed.meanAV'
    output_smooth_AV_file = output_smooth_AV_root + '.fits'
    output_smooth_meanAV_file = output_smooth_meanAV_root + '.fits'
    output_smooth_AV_ratio_file = output_smooth_AV_root + '.ratio.fits'
    output_smooth_meanAV_ratio_file = output_smooth_meanAV_root + '.ratio.fits'
    output_smooth_AV_ratiocorr_file = output_smooth_AV_root + '.ratiocorr.fits'
    output_smooth_meanAV_ratiocorr_file = output_smooth_meanAV_root + '.ratiocorr.fits'
        
    # median extinction
    arrayname = 'bestfit_values' + cleanstr
    arraynum = 1
    a, ra_bins, dec_bins = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                                           arrayname=arrayname, 
                                           arraynum=arraynum)
    AV = a
    print 'Size of extinction map: ', AV.shape

    # mean extinction
    if (fileroot == 'merged'):
        arrayname = 'dervied_values' + cleanstr  #  note mispelling!
    else:
        arrayname = 'derived_values' + cleanstr
    arraynum = 0
    a, ra_bins, dec_bins = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                                           arrayname=arrayname, 
                                           arraynum=arraynum)
    meanAV = a

    # run the smoothing algorithm on both A_V maps

    if (write_fits == 'True'):

        wcs, im_coords, draineimg, AV_img = compare_img_to_AV(AV, ra_bins, dec_bins, drainefile, 
                                                              crop='True',
                                                              scaleimgfactor = 1.0,
                                                              resolution_in_arcsec=draineresolution, 
                                                              AV_resolution_in_arcsec=AVresolution, 
                                                              outputAVfile=output_smooth_AV_file)
        wcs, im_coords, draineimg, meanAV_img = compare_img_to_AV(meanAV, ra_bins, dec_bins, drainefile, 
                                                                  crop='True',
                                                                  scaleimgfactor = 1.0,
                                                                  resolution_in_arcsec=draineresolution, 
                                                                  AV_resolution_in_arcsec=AVresolution, 
                                                                  outputAVfile=output_smooth_meanAV_file)

    # read from fits files (guarantees consistency -- compare_img_to_AV returns cropped version, but FITS=full)
    f = pyfits.open(drainefile)
    draineimg = f[0].data
    hdr = f[0].header
    wcs = pywcs.WCS(hdr)
    f = pyfits.open(drainefile)
    hdr, draineimg = f[0].header, f[0].data
    f = pyfits.open(output_smooth_AV_file)
    AV_img = f[0].data
    f = pyfits.open(output_smooth_meanAV_file)
    meanAV_img = f[0].data

    # smooth images to reduce noise, if requested
    if (smooth_img > 0):

        print 'Boxcar smoothing images to reduce noise, using width ', int(smooth_img)
        draineimg = scipy.stsci.convolve.boxcar(draineimg, (int(smooth_img), int(smooth_img)))
        AV_img = scipy.stsci.convolve.boxcar(AV_img, (int(smooth_img), int(smooth_img)))
        meanAV_img = scipy.stsci.convolve.boxcar(meanAV_img, (int(smooth_img), int(smooth_img)))

    # get ra dec of draine image

    i_dec, i_ra = np.meshgrid(np.arange(draineimg.shape[1]), np.arange(draineimg.shape[0]))
    i_coords = np.array([[i_dec[i,j],i_ra[i,j]] for (i,j),val in np.ndenumerate(i_ra)])
    print 'draineimg.shape: ', draineimg.shape
    print 'i_dec.shape: ', i_dec.shape
    print 'i_coords.shape: ', i_coords.shape
    # solve for RA, dec at those coords
    img_coords = wcs.wcs_pix2sky(i_coords, 1)
    img_coords = np.reshape(img_coords,(i_ra.shape[0],i_ra.shape[1],2))
    ra  = img_coords[:,:,0]
    dec = img_coords[:,:,1]

    lgnstar = np.log10(iAV.get_nstar_at_ra_dec(ra, dec, renormalize_to_surfdens=True))

    # select regions for analysis

    mask = np.where(AV_img > 0, 1.0, 0.0)
    i_masked = np.where(mask == 0)
    i_unmasked = np.where(mask > 0)
    mask_loAV = np.where(AV_img > 0.25, 1.0, 0.0)
    i_loAV = np.where((mask_loAV == 0) & (mask > 0))
    i_hiAV = np.where(mask_loAV > 0)

    AVratio = draineimg / AV_img 
    meanAVratio = draineimg / meanAV_img
    AVratiocorr = (draineimg / ratiofix) / (AV_img + biasfix)
    meanAVratiocorr = (draineimg / ratiofixmean) / (meanAV_img + biasfixmean)
    # clean up masked regions -- do replacement to deal with NaN's in divisions.
    AVratio[i_masked] = 0.0
    meanAVratio[i_masked] = 0.0
    AVratiocorr[i_masked] = 0.0
    meanAVratiocorr[i_masked] = 0.0

    # write ratio fits files

    if (write_ratio_fits == 'True'):
        print ' Writing ratio maps'
        pyfits.writeto(output_smooth_AV_ratio_file, AVratio, header=hdr)
        pyfits.writeto(output_smooth_meanAV_ratio_file, meanAVratio, header=hdr)
        pyfits.writeto(output_smooth_AV_ratiocorr_file, AVratiocorr, header=hdr)
        pyfits.writeto(output_smooth_meanAV_ratiocorr_file, meanAVratiocorr, header=hdr)

    # set minimum AV to use in plots

    AVlim = 0.325

    i_good = np.where(AV_img > AVlim)

    ################################################
    # Make plots

    # set image region

    region1 = [11.05, 41.32, 0.47, 0.45]   # brick 5 + SF ring
    region2 = [11.29, 41.8,  0.55, 0.5]    # brick 9 + brick 15 region
    region3 = [11.585, 42.1, 0.55, 0.5]
    allregion = [11.33, 41.72, 0.92, 1.28]   # brick 5 + SF ring

    # Redefine fits files, in case write_fits='False'

    output_smooth_AV_file = output_smooth_AV_root + '.fits'
    output_smooth_meanAV_file = output_smooth_meanAV_root + '.fits'

    # image of smoothed AV

    gc = aplpy.FITSFigure(output_smooth_AV_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=4, cmap='hot', 
                       interpolation='nearest', aspect='auto')

    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$A_V$')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])
    filename = output_smooth_AV_root + '.region1' + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of meanAV smoothed

    gc = aplpy.FITSFigure(output_smooth_meanAV_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=4, cmap='hot', 
                       interpolation='nearest', aspect='auto')

    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$A_V$')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])
    filename = output_smooth_meanAV_root + '.region1' + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of smoothed AV ratio

    ratiomax = 10

    gc = aplpy.FITSFigure(output_smooth_AV_ratio_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=ratiomax, cmap='spectral', 
                       interpolation='nearest', aspect='auto')
    # add AV level contours
    gc.show_contour(output_smooth_AV_file, levels=[1.0], convention='wells',
                    colors='black', linewidths=4)


    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text(r'$A_{V,emission} / \widetilde{A_V}$')

    r = allregion
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region1'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region2
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region2'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region3
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region3'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of smoothed meanAV ratio

    gc = aplpy.FITSFigure(output_smooth_meanAV_ratio_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=ratiomax, cmap='spectral', 
                       interpolation='nearest', aspect='auto')
    # add AV level contours
    gc.show_contour(output_smooth_meanAV_file, levels=[1.0], convention='wells',
                    colors='black', linewidths=4)


    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text(r'$A_{V,emission} / \langle A_V \rangle$')

    r = allregion
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region1'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region2
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region2'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region3
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region3'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of smoothed AV corrected ratio

    gc = aplpy.FITSFigure(output_smooth_AV_ratiocorr_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=2, cmap='seismic', 
                       interpolation='nearest', aspect='auto')
    # add AV level contours
    gc.show_contour(output_smooth_AV_file, levels=[1.0], convention='wells',
                    colors='black', linewidths=4)


    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text(r'$(A_{V,emission} / %4.2f) / (\widetilde{A_V} + %4.2f)$' % (ratiofix, biasfix))

    r = allregion
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region1'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region2
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region2'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region3
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region3'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of smoothed meanAV ratio

    gc = aplpy.FITSFigure(output_smooth_meanAV_ratiocorr_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=2, cmap='seismic', 
                       interpolation='nearest', aspect='auto')
    # add AV level contours
    gc.show_contour(output_smooth_meanAV_file, levels=[1.0], convention='wells',
                    colors='black', linewidths=4)

    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text(r'$(A_{V,emission} / %4.2f) / (\langle A_V \rangle + %4.2f)$' % (ratiofixmean, biasfixmean))

    r = allregion
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region1'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region2
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region2'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region3
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region3'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    ################################################
    # adopt larger fonts and set alpha value

    print 'Increasing font size...'
    
    font = {'weight': '500',
            'size': '18'}
    plt.rc('font', **font)
    
    alpha = 1.0
    greyval = '#B3B3B3'
    plotfigsize = (10.0,10.0)

    # Correlation plot AV  (ghost out low AV points)

    plt.figure(11, figsize=plotfigsize)
    plt.close()
    plt.figure(11, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(AV_img[i_loAV], draineimg[i_loAV], ',', color=greyval, alpha=alpha)
    im = plt.plot(AV_img[i_hiAV], draineimg[i_hiAV], ',', color='black', alpha=alpha)
    plt.xlabel(r'$\widetilde{A_V}$')
    plt.ylabel(r'$A_{V,emission}$')
    plt.axis([0, 3.5, -0.25, 12])
    
    exten = '.correlation'
    savefile = output_smooth_AV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # Correlation plot meanAV (ghost out low AV points)

    plt.figure(12, figsize=plotfigsize)
    plt.close()
    plt.figure(12, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(meanAV_img[i_loAV], draineimg[i_loAV], ',', color=greyval, alpha=alpha)
    im = plt.plot(meanAV_img[i_hiAV], draineimg[i_hiAV], ',', color='black', alpha=alpha)
    plt.xlabel(r'$\langle A_V \rangle$')
    plt.ylabel(r'$A_{V,emission}$')
    plt.axis([0, 3.5, -0.25, 12])
    
    exten = '.correlation'
    savefile = output_smooth_meanAV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # corrected Correlation plot AV  (ghost out low AV points)

    plt.figure(21, figsize=plotfigsize)
    plt.close()
    plt.figure(21, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(AV_img[i_loAV] + biasfix, draineimg[i_loAV] / ratiofix, ',', color=greyval, alpha=alpha)
    im = plt.plot(AV_img[i_hiAV] + biasfix, draineimg[i_hiAV] / ratiofix, ',', color='black', alpha=alpha)
    plt.plot([-1,10], [-1,10], color='red', linewidth=4)
    plt.xlabel(r'$\widetilde{A_V} + %4.2f$' % biasfix)
    plt.ylabel(r'$A_{V,emission} / %4.2f$' % ratiofix)
    plt.axis([-0.25, 3.5, -0.25, 3.5])
    
    exten = '.correlationcorr'
    savefile = output_smooth_AV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # corrected Correlation plot AV  (ghost out low AV points)

    plt.figure(22, figsize=plotfigsize)
    plt.close()
    plt.figure(22, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(meanAV_img[i_loAV] + biasfixmean, draineimg[i_loAV] / ratiofixmean, ',', color=greyval, alpha=alpha)
    im = plt.plot(meanAV_img[i_hiAV] + biasfixmean, draineimg[i_hiAV] / ratiofixmean, ',', color='black', alpha=alpha)
    plt.plot([-1,10], [-1,10], color='red', linewidth=4)
    plt.xlabel(r'$\langle A_V \rangle + %4.2f$' % biasfixmean)
    plt.ylabel(r'$A_{V,emission} / %4.2f$' % ratiofixmean)
    plt.axis([-0.25, 3.5, -0.25, 3.5])
    
    exten = '.correlationcorr'
    savefile = output_smooth_meanAV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # Ratio plot AV  (ghost out low AV points)

    AVvec = np.linspace(0.001,10,100)
    ratiovec = ratiofix * (1.0 + biasfix/AVvec)

    plt.figure(13, figsize=plotfigsize)
    plt.close()
    plt.figure(13, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(AV_img[i_loAV], draineimg[i_loAV] / AV_img[i_loAV], ',', color=greyval, alpha=alpha)
    im = plt.plot(AV_img[i_hiAV], draineimg[i_hiAV] / AV_img[i_hiAV], ',', color='black', alpha=alpha)
    plt.plot(AVvec, ratiovec, color='red', linewidth=4)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$A_{V,emission}  /  \widetilde{A_V}$')
    plt.axis([0, 3.5, 0, 7])
    
    exten = '.ratio'
    savefile = output_smooth_AV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # Ratio plot meanAV  (ghost out low AV points)

    ratiovec = ratiofixmean * (1.0 + biasfixmean/AVvec)

    plt.figure(14, figsize=plotfigsize)
    plt.close()
    plt.figure(14, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(meanAV_img[i_loAV], draineimg[i_loAV] / meanAV_img[i_loAV], ',', 
                  color=greyval, alpha=alpha)
    im = plt.plot(meanAV_img[i_hiAV], draineimg[i_hiAV] / meanAV_img[i_hiAV], ',', 
                  color='black', alpha=alpha)
    plt.plot(AVvec, ratiovec, color='red', linewidth=4)
    plt.xlabel(r'$\langle A_V \rangle$')
    plt.ylabel(r'$A_{V,emission}  /  \langle A_V \rangle$')
    plt.axis([0, 3.5, 0, 7])
    
    exten = '.ratio'
    savefile = output_smooth_meanAV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # restore font size

    print 'Restoring original font defaults...'
    plt.rcdefaults()

    return


# read in image, and match resolution & astrometry of extinction map to it
def compare_img_to_AV(AV, ra_bins, dec_bins, imgfile, AV_resolution_in_arcsec=6.645, 
                      resolution_in_arcsec='', scaleimgfactor=1.0,
                      crop='True', outputAVfile='', 
                      usemeanAV=True):

    # Note. Resolutions are assumed to be FWHM

    f = pyfits.open(imgfile)
    hdr, img = f[0].header, f[0].data
    wcs = pywcs.WCS(hdr)
    #wcs.wcs.print_contents()    

    # scale image by arbitrary factor to bring onto AV
    img = img * scaleimgfactor

    # make grid of RA and Dec at each pixel
    i_dec, i_ra = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    if (wcs.wcs.naxis == 2):
        i_coords = np.array([[i_dec[i,j],i_ra[i,j]] for (i,j),val in np.ndenumerate(i_ra)])
    if (wcs.wcs.naxis == 3):
        i_coords = np.array([[i_dec[i,j],i_ra[i,j],0] for (i,j),val in np.ndenumerate(i_ra)])
    print 'img.shape: ', img.shape
    print 'i_dec.shape: ', i_dec.shape
    print 'i_coords.shape: ', i_coords.shape

    # solve for RA, dec at those coords
    img_coords = wcs.wcs_pix2sky(i_coords, 1)
    if (wcs.wcs.naxis > 2):
        img_coords = img_coords[:,0:2]
    img_coords = np.reshape(img_coords,(i_ra.shape[0],i_ra.shape[1],2))
    ra_img  = img_coords[:,:,0]
    dec_img = img_coords[:,:,1]
    print 'ra_img.shape:', ra_img.shape

    # get coords of  A_V data

    racenvec  = (ra_bins[0:-2]  +  ra_bins[1:-1]) / 2.0
    deccenvec = (dec_bins[0:-2] + dec_bins[1:-1]) / 2.0
    dec_AV, ra_AV = np.meshgrid(deccenvec, racenvec)

    # smooth to match resolution of image

    # center of bulge
    m31ra  = 10.6847929
    m31dec = 41.2690650    
    # use center of image as tangent point
    m31ra_mid = (ra_bins[-1] + ra_bins[0]) / 2.0
    m31dec_mid = (dec_bins[-1] + dec_bins[0]) / 2.0
    m31ra = m31ra_mid
    m31dec = m31dec_mid
    dra = (ra_bins[1] - ra_bins[0]) * np.cos(np.math.pi * m31dec / 180.0)
    ddec = dec_bins[1] - dec_bins[0]
    pixscale = 3600.0 * (dra + ddec) / 2.0

    print 'Size of AV image: ', AV.shape

    # prepare information needed to handle edges
    plt.figure(1)
    plt.clf()
    im = plt.imshow(AV, vmin=-0.25, vmax=4, cmap='hot')
    plt.colorbar(im)
    plt.suptitle('Original AV')

    mask = np.where(AV > 0, 1., 0.)
    AV = AV * mask

    plt.figure(2)
    plt.clf()
    plt.imshow(mask, vmin=0, vmax=1, cmap='hot')
    plt.suptitle('Mask for Original AV')

    plt.figure(3)
    plt.clf()
    im = plt.imshow(AV, vmin=-0.25, vmax=4, cmap='hot')
    plt.colorbar(im)
    plt.suptitle('Masked Original AV')

    if (resolution_in_arcsec != ''): 
        # convert FWHM to equivalent gaussian sigma: FWHM = 2*sqrt(2 ln 2) * sigma
        smootharcsec = np.sqrt(resolution_in_arcsec**2 - AV_resolution_in_arcsec**2) / (2.0*np.sqrt(2.0*np.log(2.0)))
        smoothpix = (smootharcsec / pixscale) 
        print 'Smoothing to ',resolution_in_arcsec,' using ',smoothpix,' pixel wide filter'

        AVsmooth = filt.gaussian_filter(AV, smoothpix, mode='reflect')  

        plt.figure(4)
        plt.clf()
        im = plt.imshow(AVsmooth, vmin=-0.25, vmax=4, cmap='hot')
        plt.colorbar(im)
        plt.suptitle('Smoothed Masked Original AV')

        # correct for masked regions

        masksmooth = filt.gaussian_filter(mask, smoothpix, mode='reflect')  

        plt.figure(5)
        plt.clf()
        plt.imshow(masksmooth, vmin=0, vmax=1, cmap='hot')
        plt.suptitle('Smoothed Mask')

        AVsmooth = (AVsmooth / masksmooth)
        i_bad = np.where(mask < 1)
        AVsmooth[i_bad] = 0.0

        plt.figure(6)
        plt.clf()
        im = plt.imshow(AVsmooth, vmin=-0.25, vmax=4, cmap='hot')
        plt.colorbar(im)
        plt.suptitle('Corrected Smooth AV')

    else:
        AVsmooth = AV

    # interpolate ra on ra_bins and dec on dec_bins to get coordinates in AV image
    # grab nearest pixel in A_V image (possible off-by-one?) and copy into copy of img

    print 'Interpolating to match image'
    i_ra_AV = np.arange(0.,len(ra_bins)-2)
    i_dec_AV = np.arange(0.,len(dec_bins)-2)
    x_AV = np.reshape(np.interp(ra_img.flatten(), ra_bins[1:-1], i_ra_AV,
                                left=-1, right=-1), 
                      ra_img.shape)
    y_AV = np.reshape(np.interp(dec_img.flatten(), dec_bins[1:-1], i_dec_AV,
                                left=-1, right=-1), 
                      dec_img.shape)
    # turn into int referencing appropriate bin
    #x_AV = np.round(x_AV).astype(int)
    #y_AV = np.round(y_AV).astype(int)
    x_AV = np.round(x_AV+1).astype(int)  # why +1? seems to work best...(tested all +,0,- 8/13-- no difference!)
    y_AV = np.round(y_AV+1).astype(int)

    # substitute AV values into appropriate pixels, provided they're in range
    # (probably a pythonic list comprehension way to do this...)
    AV_img = -1.0 + 0.0*img
    minx = 1e6
    miny = 1e6
    maxx = 0
    maxy = 0
    for (x,y),v in np.ndenumerate(AV_img):
        if ((x_AV[x, y] > 0) & (y_AV[x, y] > 0)):
            AV_img[x, y] = AVsmooth[x_AV[x,y], y_AV[x,y]]
            if (minx > x): 
                minx = x
            if (miny > y): 
                miny = y
            if (maxx < x): 
                maxx = x
            if (maxy < y): 
                maxy = y

    # if requested, output AV_image
    if (outputAVfile != ''):
        # fix up header if there's lingering velocity info...
        if ((hdr['NAXIS'] == 2) & ('CTYPE3' in hdr)):
            print 'Removing unneccessary 3rd axis from header'
            if ('CTYPE3' in hdr):
                del hdr['CTYPE3']
            if ('CRVAL3' in hdr):
                del hdr['CRVAL3']
            if ('CDELT3' in hdr):
                del hdr['CDELT3']
            if ('CRPIX3' in hdr):
                del hdr['CRPIX3']
            if ('CROTA3' in hdr):
                del hdr['CROTA3']
        print 'Writing output to ',outputAVfile
        pyfits.writeto(outputAVfile, AV_img, hdr)

    # if requested, crop to region where the AV map is valid
    if (crop):
        print 'Cropping to [',minx,':',maxx+1,',',miny,':',maxy+1,']'
        img = img[minx:maxx+1,miny:maxy+1]
        AV_img = AV_img[minx:maxx+1,miny:maxy+1]

    # Consider installing pywcsgrid2...

    # close file
    f.close()

    return wcs, img_coords, img, AV_img

#def compare_draine_dust(AVdatafile='../Results/SecondRunLoRes/merged.npz'):
def compare_draine_dust(AV, ra_bins, dec_bins):

    drainefile = '../draine_M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.fits'
    resolution = 10.0  # arcsec
    #resolution = ''  # arcsec
    outputfile = '../Results/FirstRun/draine_matched_AV.fits'
    scalefac = 0.739 / 1.e5  # Draine email May 29, 2013

    #wcs, im_coords, img, AV_img = compare_img_to_AV(drainefile, crop='True',
    wcs, im_coords, img, AV_img = compare_img_to_AV(AV, ra_bins, dec_bins, drainefile, crop='True',
                                                    scaleimgfactor = scalefac,
                                                    resolution_in_arcsec=resolution, 
                                                    #outputAVfile='', AVdatafile=AVdatafile,
                                                    outputAVfile='', 
                                                    usemeanAV=True)
    mask = np.where(AV_img > 0, 1.0, 0.0)
    lowAVmask = np.where(AV_img > 0.4, 1.0, 0.0)
    rat = AV_img / img
    medrat = np.median(rat[np.where(AV_img > 0.75)])
    print 'Median Ratio: ', medrat

    # Fit correlation to data 
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    # http://stackoverflow.com/questions/9990789/how-to-force-zero-interception-in-linear-regression
    
    i_fit = np.where((AV_img.flatten() > 0.5) & (AV_img.flatten() < 4))[0]
    x = AV_img.flatten()[i_fit]
    y = img.flatten()[i_fit]
    A = np.vstack([x, np.ones(len(x))]).T
    print 'Shape I, X, Y, A: ', i_fit.shape, x.shape, y.shape, A.shape
    m, c = np.linalg.lstsq(A, y)[0]
    print 'Fit by Dust Mass = ',m,' * A_V + ', c
    m0 = np.linalg.lstsq(x[:,np.newaxis], y)[0]
    print 'Fit by Dust Mass = ',m0,' * A_V'
    mean_rat = 1.0 / m0

    # Whole galaxy

    plt.figure(1, figsize=(14,7))
    plt.clf()
    
    plt.subplot(1,3,1)
    im = plt.imshow(img, interpolation='nearest',aspect='auto', 
               origin='lower', vmin=0, vmax=6)
    plt.colorbar(im)
    plt.title('$A_V$ (Emission)')

    plt.subplot(1,3,2)
    plt.imshow(AV_img, interpolation='nearest',aspect='auto', origin='lower',
               vmin=0,vmax=6)
    plt.colorbar(im)
    plt.title('$A_V$ (Extinction)')

    plt.subplot(1,3,3)
    im = plt.imshow(lowAVmask*((rat - mean_rat)/mean_rat), cmap='seismic',
                    interpolation='nearest', aspect='auto', 
                    origin='lower', vmin=-1,vmax=1)
    plt.colorbar(im)
    plt.title('Fractional $\Delta (A_{V,emit}/A_{V,ext})$')

    # brick 15 zoom
    plt.figure(2, figsize=(7, 10))
    plt.clf()

    x0, x1, y0, y1 = 150, 235, 260, 310
    plt.subplot(3,1,1)
    im = plt.imshow(mask[y0:y1,x0:x1]*img[y0:y1,x0:x1], 
                    interpolation='nearest',aspect='auto', 
                    origin='lower', vmin=0, vmax=6)
    plt.colorbar(im)
    plt.title('$A_V$ (Emission)')

    plt.subplot(3,1,2)
    im = plt.imshow(AV_img[y0:y1,x0:x1], interpolation='nearest',aspect='auto', origin='lower',
               vmin=0,vmax=6)
    plt.colorbar(im)
    plt.title('$A_V$ (Extinction)')

    plt.subplot(3,1,3)
    im = plt.imshow(lowAVmask[y0:y1,x0:x1]*((rat[y0:y1,x0:x1]-mean_rat)/mean_rat), 
                    cmap='seismic',
                    interpolation='nearest', aspect='auto', 
                    origin='lower', vmin=-1.5,vmax=1.5)
    plt.colorbar(im)
    plt.title('Fractional $\Delta (A_{V,emit}/A_{V,ext})$')

    # correlation as points
    #plt.figure(3)
    #plt.close()

    #plt.subplot(1,2,1)
    #plt.plot(AV_img, (img/1.e7), ',', alpha=0.1, color='b')
    #plt.xlabel('$A_V$')
    #plt.ylabel('Dust Mass')
    #plt.axis([0.0, 4.0, -0.005, 0.175])

    #plt.subplot(1,2,2)
    #plt.plot(AV_img, AV_img/(img/1.e7), ',', alpha=0.1, color='b')
    #plt.xlabel('$A_V$')
    #plt.ylabel('$A_V$ / Dust Mass')
    #plt.axis([0.0, 4.0, 0.0, 75])

    # correlation as density maps
    plt.figure(4)
    plt.clf()

    AVmin, AVmax = 0.0001, 8.0
    imgmin, imgmax = AVmin, AVmax
    ratmin, ratmax = 0.0, 4.0
    nbins = 100
    
    AVbins = np.linspace(AVmin, AVmax, num=nbins)
    imgbins = np.linspace(imgmin, imgmax, num=nbins)
    ratbins = np.linspace(ratmin, ratmax, num=nbins)

    h_AV_img, AVedges, imgedges = np.histogram2d(img.flatten(), AV_img.flatten(), 
                                                 range=[[AVmin, AVmax],
                                                        [AVmin, AVmax]], 
                                                 bins=nbins)
    extent_AV_img = [AVmin, AVmax, imgmin, imgmax]

    h_AV_AVoverimg, AVedges, imgedges = np.histogram2d(((rat - mean_rat)/mean_rat).flatten(), 
                                                        AV_img.flatten(), 
                                                 range=[[ratmin, ratmax],
                                                        [AVmin,AVmax]], 
                                                 bins=nbins)
    extent_AV_AVoverimg = [AVmin, AVmax, ratmin, ratmax]


    plt.subplot(1,2,1)
    plt.imshow(np.log(h_AV_img), extent=extent_AV_img, 
               origin='lower', aspect='auto')
    plt.plot(x, m*x + c, color='black')
    plt.plot(x, m0*x, color='black', linestyle='-')
    plt.xlabel('$A_V$ (Extinction)')
    plt.ylabel('$A_V$ (Emission)')

    plt.subplot(1,2,2)
    plt.imshow(np.log(h_AV_AVoverimg), extent=extent_AV_AVoverimg, 
               origin='lower', aspect='auto')
    plt.xlabel('$A_V$')
    plt.ylabel('$A_{V,extinction}$ / $A_{V,emission}$')

    return img, AV_img, i_fit

def make_model_frac_red(inclination = 70., hz_over_hr = 0.2):

    xvec = np.linspace(-10,10,100)
    yvec = np.linspace(-10,10,100)
    x, y = np.meshgrid(xvec, yvec)
    
    # conversion from degrees to radians
    radeg  = np.pi / 180.
    incl = inclination * radeg
    b_over_a = math.cos(incl)
    ecc = math.sqrt(1. - (b_over_a)**2.)
    #pa_deg = 43.5
    pa_deg = 0.0
    pa = pa_deg * radeg

    # get major axis length at each position
    r = np.sqrt((x * math.cos(pa) + y * math.sin(pa))**2 +
                (x * math.sin(pa) - y * math.cos(pa))**2 / (1.0 - ecc**2))
    
    x0 = x / math.cos(incl)

    # calculate reddened fraction
    f = (1 + np.sign(x) * hz_over_hr * math.tan(incl) * np.cos(theta)) / 2.0

    # plot it...
    plt.figure(1)
    plt.clf()
    im = plt.imshow(f,vmin=0, vmax=1, cmap='seismic',extent=[-10,10,-10,10])
    plt.colorbar(im)

    return f

def plot_all_low_AV_rgb_maglims():

    imgfileroot = '../Unreddened/low_AV_CMD_r.'

    plot_low_AV_rgb_maglims(ikeep=43)
    plt.savefig(imgfileroot + '0.28.png')
    
    plot_low_AV_rgb_maglims(ikeep=20)
    plt.savefig(imgfileroot + '0.5.png')
    
    plot_low_AV_rgb_maglims(ikeep=11)
    plt.savefig(imgfileroot + '0.7.png')

    plot_low_AV_rgb_maglims(ikeep=3)
    plt.savefig(imgfileroot + '1.1.png')

    return
    
def plot_low_AV_rgb_maglims(ikeep=20, reflgnstar=2.0, nrgbstars = 5000, nsubstep=3., 
                            mrange = [18.2, 25.],
                            crange = [0.3, 2.5],
                            deltapixorig = [0.015,0.25],
                            mnormalizerange = [19,21.5], 
                            maglimoff = [0.0, 0.25]):

    # plot CMD and magnitude limit diagram for desired value of reflgnstar (log10 stellar surf dens)

    # Define reddening parameters

    Amag_AV = 0.20443
    Acol_AV = 0.33669 - 0.20443
    t = np.arctan(-Amag_AV / Acol_AV)
    reference_color = 1.0

    # read in cleaned, low reddening data

    c, m, ra, dec, r, cstd, cm, nstar = iAV.read_clean()

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

    nnstarbins = len(nnstarhi)

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

    # Loop through bins of log10(nstar) to get mean nstar

    meannstar_array = np.zeros(nnstarbins)
    meanr_array = np.zeros(nnstarbins)

    for i in range(len(nnstarlo)):

        meannstar_array[i] = np.average(nstar[nnstarlo[i]:nnstarhi[i]])
        meanr_array[i] = np.average(r[nnstarlo[i]:nnstarhi[i]])

    # select bin with closest to desired nstar

        # http://stackoverflow.com/questions/9706041/finding-index-of-an-item-closest-to-the-value-in-a-list-thats-not-entirely-sort
    #ikeep = min(range(len(meannstar_array.tolist())), key=lambda i: abs(meannstar_array-reflgnstar))
    print 'Selecting element ',ikeep,' where meannstar=', meannstar_array[ikeep]
    

    # grab appropriate values from array

    meanr = meanr_array[ikeep]
    lgnstar = meannstar_array[ikeep]
    ckeep = c[nnstarlo[ikeep]:nnstarhi[ikeep]]
    mkeep = m[nnstarlo[ikeep]:nnstarhi[ikeep]]

    # set up magnitude limits

    mlim110 = p110(lgnstar)
    mlim160 = p160(lgnstar)
    cvec = np.linspace(0, 3.5, 100)
    maglim160 = mlim160 + 0*cvec
    maglim110 = mlim110 - cvec
    maglim = np.minimum(maglim160, maglim110)

    # plot stars

    plt.figure(1)
    plt.clf()
    plt.plot(ckeep, mkeep, ',', alpha=0.2, color='blue')
    plt.plot(cvec, maglim, color='black', linewidth=3)
    plt.xlabel('F110W - F160W')
    plt.ylabel('F160W')
    #plt.title(r"$\Sigma_{stars} = %5.2f$ arcsec$^{-2}$    $\langle r \rangle = %4.2f$ degrees" % 
    #          (lgnstar, meanr))
    plt.axis([0, 3.5, 25.5, 18])

    ax = plt.gca()
    ax.annotate(r"$\Sigma_{stars} = %4.2f$ arcsec$^{-2}$" % 
                (10.**lgnstar), xy=(2.25,24.5), xytext=None, horizontalalignment='left',
                fontsize=15.0)
    ax.annotate(r"$\langle r \rangle \approx %4.2f$ degrees" % 
                (meanr),   xy=(2.25,25), xytext=None, horizontalalignment='left',
                fontsize=15.0)
    
    # add reddening vector

    AV = 5.
    cref = 1.0
    mref = 18.8
    plt.arrow(cref, mref, AV*Acol_AV, AV*Amag_AV, width=0.05, color='r')
    plt.annotate(r"$A_V=5$", xy=(cref + 0.5*AV*Acol_AV, mref+0.5*AV*Amag_AV-0.1), xytext=None)
    cref = 0.94
    mref = 19.8
    plt.arrow(cref, mref, AV*Acol_AV, AV*Amag_AV, width=0.05, color='r')
    cref = 0.85
    mref = 20.8
    plt.arrow(cref, mref, AV*Acol_AV, AV*Amag_AV, width=0.05, color='r')
    cref = 0.75
    mref = 21.8
    plt.arrow(cref, mref, AV*Acol_AV, AV*Amag_AV, width=0.05, color='r')
    
    return

def plot_cumulative_log_normal():

    # investigate fraction of gas above give AV.
    # results: at A_median, 50% have higher AV (by design)
    #          10% have AV higher than A/Amedian > 1 + 2*sig (roughly)

    A_over_Amedian = np.linspace(0, 5., 100)
    
    sigvec = [0.3, 0.5, 0.7]
    nsig = len(sigvec)
    p = np.zeros((nsig,len(A_over_Amedian)))

    # initialize fit to log-normal tail
    Athresh = 1.0
    lg10pthresh = -2.
    npoly = 1
    linfitparam = np.zeros((nsig,2))

    for i, sig in enumerate(sigvec):
        
        p[i,:] = 0.5*spec.erfc(np.log(A_over_Amedian)/(sig*np.sqrt(2)))
        
        # fit linear relationship to high A_over_Amedian points
        i_hi = np.where((A_over_Amedian > Athresh) & (np.log10(p[i,:]) > lg10pthresh))
        x = A_over_Amedian[i_hi]
        y = np.log10(p[i,i_hi].flatten())
        linfitparam[i,:] = np.polyfit(x, y, npoly)
        print sig, linfitparam[i,:], linfitparam[i,:] * sig, linfitparam[i,:] * sig**2

    plt.figure(1)
    plt.clf()
    plt.plot([1, 1],[-6,1],linewidth=0.5, color='grey')
    plt.plot([0, 10],[-1,-1],linewidth=0.5, color='grey')
    for i, sig in enumerate(sigvec):
        poly = np.poly1d(linfitparam[i,:])
        plt.plot(A_over_Amedian, poly(A_over_Amedian), color='red', linewidth=0.5)

    p1 = plt.plot(A_over_Amedian, np.log10(p[0,:]), linewidth=2, color='black')
    p2 = plt.plot(A_over_Amedian, np.log10(p[1,:]), linewidth=3, color='black')
    p3 = plt.plot(A_over_Amedian, np.log10(p[2,:]), linewidth=5, color='black')
    plt.axis([0,4,-2,0.05])
    plt.xlabel(r"$A / A_{median}$")
    plt.ylabel(r"$log_{10} p(> A / A_{median} | \sigma_A)$")

    plt.legend([p1,p2,p3], ["$\sigma_A=0.3$", "$\sigma_A=0.5$", "$\sigma_A=0.7$"], frameon=False)

    return

def plot_mass_area(A_thresh = 1.0):

    # investigate fraction of cloud area above given AV threshold (as in Beaumont et al 2012)
    # as a function of changing median A_V

    sig = 0.4
    A_median = np.linspace(0, 5., 100)

    p = 0.5*spec.erfc(np.log(A_thresh / A_median)/(sig*np.sqrt(2)))
        
    plt.figure(1)
    plt.clf()
    plt.plot([A_thresh, A_thresh],[-6,1],linewidth=0.5, color='grey')
    plt.plot([0, 10],[-1,-1],linewidth=0.5, color='grey')
    plt.plot(A_median, np.log10(p), linewidth=3, color='black')
    plt.axis([0,4,-2,0.05])
    plt.xlabel(r"$A_{median}$")
    plt.ylabel(r'$log_{10} p(> A_{thresh}=%3.1f | \sigma_A=%3.1f)$' % (A_thresh,sig))

    return

def plot_pixel_demo(fileroot = 'demo_modelfit_data_7_63'):

    # plot example of fitting for an individual pixel

    dat = np.load(fileroot + '.npz')  # brick 15
    fg_cmd = dat['fg_cmd']
    fake_cmd = dat['fake_cmd']
    color_boundary = dat['color_boundary']
    qmag_boundary = dat['qmag_boundary']
    d_derived = dat['d_derived'].tolist()
    d = dat['d'].tolist()
    #print d_derived
    #print d
    #print d['A_V']
    bestfit = dat['bestfit']
    frac_red_mean = 0.4
    x = bestfit[0]
    alpha = np.log(0.5) / np.log(frac_red_mean)
    f = (np.exp(x) / (1.0 + np.exp(x)))**(1./alpha)
    sigma_squared = np.log((1. + np.sqrt(1. + 4. * (bestfit[2])**2)) / 2.)
    sigma = np.sqrt(sigma_squared)
    i_c = dat['i_c']
    i_q = dat['i_q']
    crange = [min(color_boundary), max(color_boundary)]
    qrange = [min(qmag_boundary), max(qmag_boundary)]
    AVvec = d['A_V']
    sigmavec = np.sqrt(np.log((1. + np.sqrt(1. + 4. * (d['w'])**2)) / 2.))
    sigmavec = d_derived['sigmavec']
    fvec = (np.exp(d['x']) / (1.0 + np.exp(d['x'])))**(1./alpha)
    meanAVvec = d_derived['meanAVvec']
    fdensevec = d_derived['fdenseval_1']
    fsfvec = d_derived['fdenseval_5']

    cinterp = interp.interp1d(np.arange(len(color_boundary)), color_boundary)
    qinterp = interp.interp1d(np.arange(len(qmag_boundary)), qmag_boundary)
    c = cinterp(i_c + 0.5)
    q = qinterp(i_q + 0.5)

    # plot image of unreddened model 

    plt.figure(1)
    plt.clf()
    plt.imshow(dat['fg_cmd'],extent=[crange[0],crange[1],qrange[1],qrange[0]], 
               origin='upper',aspect='auto', interpolation='nearest', cmap='gist_heat_r')
    plt.xlabel('F110W - F160W')
    plt.ylabel('Extinction Corrected F160W')
    plt.savefig(fileroot + '_unreddened.png')
    
    # plot image of unreddened model 

    plt.figure(2)
    plt.clf()
    plt.imshow(dat['fg_cmd'],extent=[crange[0],crange[1],qrange[1],qrange[0]], 
               origin='upper',aspect='auto', interpolation='nearest', cmap='gist_heat_r')
    plt.plot(c, q, '*', color='blue', linewidth=0, ms=12)
    plt.axis([crange[0],crange[1],qrange[1],qrange[0]])
    plt.xlabel('F110W - F160W')
    plt.ylabel('Extinction Corrected F160W')
    plt.savefig(fileroot + '_unreddened_w_data.png')
    
    # plot image of reddened model

    plt.figure(3)
    plt.clf()
    plt.imshow(fake_cmd,extent=[crange[0],crange[1],qrange[1],qrange[0]], 
               origin='upper',aspect='auto', interpolation='nearest', cmap='gist_heat_r')
    plt.xlabel('F110W - F160W')
    plt.ylabel('Extinction Corrected F160W')
    plt.savefig(fileroot + '_reddened.png')
    
    # plot image of reddened model + data

    plt.figure(4)
    plt.clf()
    p1 = plt.imshow(fake_cmd,extent=[crange[0],crange[1],qrange[1],qrange[0]], 
               origin='upper',aspect='auto', interpolation='nearest', cmap='gist_heat_r')
    plt.plot(c, q, '*', color='blue', linewidth=0, ms=12)
    plt.axis([crange[0],crange[1],qrange[1],qrange[0]])
    plt.xlabel('F110W - F160W')
    plt.ylabel('Extinction Corrected F160W')
    plt.annotate(r'$A_{V,median} = %4.2f$' % bestfit[1], xy=(2.4, 19.5), fontsize=17,horizontalalignment='right')
    plt.annotate(r'$\sigma = %4.2f$' % sigma, xy=(2.4, 20.0), fontsize=17,horizontalalignment='right')
    plt.annotate(r'$f_{reddened} = %4.2f$' % f, xy=(2.4, 20.5), fontsize=17,horizontalalignment='right')
    plt.savefig(fileroot + '_reddened_w_data.png')

    # plot banana diagrams
    
    plt.figure(5)
    plt.clf()
    plt.plot(AVvec, fvec, ',', alpha=0.2, c='b')
    plt.axis([0, 4, 0, 1])
    plt.xlabel('Median $A_V$')
    plt.ylabel('$f_{reddened}$')
    plt.savefig(fileroot + '_AV_fred.png')

    
    plt.figure(6)
    plt.clf()
    plt.plot(meanAVvec, sigmavec, ',', alpha=0.2, c='b')
    plt.axis([0, 4, 0, 1])
    plt.xlabel('Mean $A_V$')
    plt.ylabel('$\sigma$')
    plt.savefig(fileroot + '_AV_sigma.png')
    
    plt.figure(7)
    plt.clf()
    plt.plot(meanAVvec, fdensevec, ',', alpha=0.2, c='b')
    plt.axis([0, 4, 0, 1])
    plt.xlabel('Mean $A_V$')
    plt.ylabel(r"$f(A_V > 1)$")
    plt.savefig(fileroot + '_AV_fAV_gt_1.png')
    
    plt.figure(8)
    plt.clf()
    plt.plot(meanAVvec, fsfvec, ',', alpha=0.2, c='b')
    plt.axis([0, 4, 0, 1])
    plt.xlabel('Mean $A_V$')
    plt.ylabel(r"$f(A_V > 5)$")
    plt.savefig(fileroot + '_AV_fAV_gt_5.png')

    return

def plot_dense_gas(filename='merged.npz', resultsdir='../Results/', AKthresh=0.1):

    AVthresh = AKthresh / 0.11613
    print 'Using A_V threshold of ', AVthresh
    d_arcsec = 6.64515
    pc_per_arcsec = 3.762
    msun_per_pc_per_meanAV = 20.903   # Rieke et al 1985

    avdat = np.load(resultsdir + filename)

    AV = avdat['bestfit_values_clean'][:,:,1]
    stddev_over_AV = avdat['bestfit_values_clean'][:,:,2]
    ###  NOTE!!! Incorrect -- compensates for error in runs 1-3!!!!
    sig = np.log((1. + np.sqrt(1. + 4. * (stddev_over_AV)**2)) / 2.)
    AVmean = AV * np.exp(sig**2 / 2.0)
    fred = avdat['bestfit_values_clean'][:,:,0]
    fdense = 0.5*special.erfc(np.log(AVthresh/AV) / 
                              (np.sqrt(2)*sig))
    gasmass_per_pix = AVmean * msun_per_pc_per_meanAV * (d_arcsec * pc_per_arcsec)**2
    densegasmass_per_pix = gasmass_per_pix * fdense

    ra_bins = avdat['ra_bins']
    dec_bins = avdat['dec_bins']
    racenvec  = (ra_bins[0:-2]  +  ra_bins[1:-1]) / 2.0
    deccenvec = (dec_bins[0:-2] + dec_bins[1:-1]) / 2.0
    #ra_AV, dec_AV = np.meshgrid(racenvec, deccenvec)
    dec_AV, ra_AV = np.meshgrid(deccenvec, racenvec)

    #make_fits_image(resultsdir + 'merged_gasmass.fits',gasmass_per_pix, ra_bins, dec_bins)
    #make_fits_image(resultsdir + 'merged_densegas_AK0.8.fits',densegasmass_per_pix, ra_bins, dec_bins)
    #make_fits_image(resultsdir + 'merged_sigmalgnorm.fits',sig, ra_bins, dec_bins)

    #plt.figure(10)
    #plt.plot(AV,fdense,',')

    #plt.figure(11)
    #plt.plot(AV,gasmass_per_pix,',')
    #plt.axis([0,10,0,1e5])

    plt.figure(1)
    plt.clf()
    plt.imshow(AVmean[::-1,::-1].T,cmap='hot', vmin=0, vmax=4)

    plt.figure(2)
    plt.clf()
    plt.imshow(fred[::-1,::-1].T,cmap='hot', vmin=0, vmax=1)

    plt.figure(3)
    plt.clf()
    im = plt.imshow(sig[::-1,::-1].T,cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im)
    plt.savefig(resultsdir + 'merged_sigmalgnorm.png')

    plt.figure(4)
    plt.clf()
    print np.min(fdense), np.max(fdense)
    plt.imshow(fdense[::-1,::-1].T,cmap='hot_r', vmin=0, vmax=0.5)

    plt.figure(5)
    plt.clf()
    print np.min(densegasmass_per_pix), np.max(densegasmass_per_pix)
    plt.imshow(densegasmass_per_pix[::-1,::-1].T,cmap='hot_r', vmin=0, vmax=1e4)
    
    #plt.figure(6)
    #plt.clf()
    #plt.plot(AVmean, sig, ',', c='blue', alpha=0.3)
    #plt.axis([0, 7, 0, 1])
    #plt.xlabel('Mean $A_V$')
    #plt.ylabel('$\sigma$')

    plt.figure(7)
    plt.clf()
    iAVgood = np.where(AVmean > 0)
    AVbins = np.linspace(0.01,10.0,100)
    lgAVbins = np.log10(np.linspace(0,10.0,100))
    plt.hist(AVmean[iAVgood],bins=lgAVbins, log=True)

    return

def makeinterleaved_fits(fileroot='merged', resultsdir='../Results/',
                         cleanstr = '_clean'):

    # median extinction
    exten = '.AV'
    arrayname = 'bestfit_values' + cleanstr
    arraynum = 1
    a, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                              arrayname=arrayname, 
                              arraynum=arraynum)
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, a, x, y)
    AV = a

    # mean extinction
    exten = '.meanAV'
    if (fileroot == 'merged'):
        arrayname = 'dervied_values' + cleanstr  #  note mispelling!
    else:
        arrayname = 'derived_values' + cleanstr
    arraynum = 0
    a, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                              arrayname=arrayname, 
                              arraynum=arraynum)
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, a, x, y)
    meanAV = a

    # reddening fraction
    exten = '.fred'
    arrayname = 'bestfit_values' + cleanstr
    arraynum = 0
    a, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                              arrayname=arrayname, 
                              arraynum=arraynum)
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, a, x, y)
    fred = a

    # sigma
    exten = '.sigma'
    arrayname = 'bestfit_values' + cleanstr
    arraynum = 2
    a, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                              arrayname=arrayname, 
                              arraynum=arraynum)
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, a, x, y)
    sigma = a

    # uncertainties

    exten = '.AVerr'
    arrayname = 'percentile_values'
    arraynum = 3
    a1, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    arraynum = 5
    a2, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, (a2 - a1)/2.0, x, y)
    exten = '.AVfracerr'
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, ((a2 - a1)/2.0) / AV, x, y)


    exten = '.meanAVerr'
    arrayname = 'derived_percentile_values'
    arraynum = 0
    a1, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    arraynum = 2
    a2, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, (a2 - a1)/2.0, x, y)
    exten = '.meanAVfracerr'
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, ((a2 - a1)/2.0) / meanAV, x, y)


    exten = '.frederr'
    arrayname = 'percentile_values'
    arraynum = 0
    a1, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    arraynum = 2
    a2, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, (a2 - a1)/2.0, x, y)

    exten = '.sigmaerr'
    arrayname = 'percentile_values'
    arraynum = 6
    a1, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    arraynum = 8
    a2, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, (a2 - a1)/2.0, x, y)
    exten = '.sigmafracerr'
    fitsfile = resultsdir + fileroot + '_interleave' + exten + '.fits'
    print 'Writing fits file: ', fitsfile
    make_fits_image(fitsfile, ((a2 - a1)/2.0) / sigma, x, y)

    return

def interleave_maps(fileroot = 'ir-sf-b16-v8-st',
                    file_0='.05_0', 
                    file_1='', 
                    file_2='.0_05', 
                    file_3='.05_05', 
                    resultsdir='../Results/',
                    arrayname='bestfit_values_clean',
                    arraynum=0):

    print 'File_1: ', file_1
    print 'File_3: ', file_3

    d_0 = np.load(resultsdir + fileroot + file_0 + '.npz')
    d_1 = np.load(resultsdir + fileroot + file_1 + '.npz')
    d_2 = np.load(resultsdir + fileroot + file_2 + '.npz')
    d_3 = np.load(resultsdir + fileroot + file_3 + '.npz')

    #arrayname='dervied_values_clean'
    #arraynum=0
    print 'Grabbing entry from ', arrayname,'[:,:,',arraynum,']'
    bf_0 = d_0[arrayname][:,:,arraynum]
    bf_1 = d_1[arrayname][:,:,arraynum]
    bf_2 = d_2[arrayname][:,:,arraynum]
    bf_3 = d_3[arrayname][:,:,arraynum]

    array_dict = {'0': bf_0, '1': bf_1, '2': bf_2, '3': bf_3}

    ra_bins_0 = d_0['ra_bins']
    ra_bins_1 = d_1['ra_bins']
    ra_bins_2 = d_2['ra_bins']
    ra_bins_3 = d_3['ra_bins']

    dec_bins_0 = d_0['dec_bins']
    dec_bins_1 = d_1['dec_bins']
    dec_bins_2 = d_2['dec_bins']
    dec_bins_3 = d_3['dec_bins']

    rabins_dict = {'0': ra_bins_0, '1': ra_bins_1, '2': ra_bins_2, '3': ra_bins_3}
    decbins_dict = {'0': dec_bins_0, '1': dec_bins_1, '2': dec_bins_2, '3': dec_bins_3}

    print bf_0.shape, ra_bins_0.shape, dec_bins_0.shape
    print bf_1.shape, ra_bins_1.shape, dec_bins_1.shape
    print bf_2.shape, ra_bins_2.shape, dec_bins_2.shape
    print bf_3.shape, ra_bins_3.shape, dec_bins_3.shape

    AV, ra_edge, dec_edge = interleave(array_dict, rabins_dict, decbins_dict)

    return AV, ra_edge, dec_edge

def interleave(array_dict, xbins_dict, ybins_dict):
    """
    interleave evenly spaced grids with half pixel shifts, 
    producing proper edge vectors as well
    
    input:

    array_dict = {'0': array0, '1': array1, '2': array2, '3': array3}
    xbins_dict = {'0': xbins0, '1': xbins1, '2': xbins2, '3': xbins3}
    ybins_dict = {'0': ybins0, '1': ybins1, '2': ybins2, '3': ybins3}
    """

    # figure out order to interleave
    xminvec  = np.array([xbins_dict['0'][0], xbins_dict['1'][0], 
                         xbins_dict['2'][0], xbins_dict['3'][0]])
    yminvec  = np.array([ybins_dict['0'][0], ybins_dict['1'][0], 
                         ybins_dict['2'][0], ybins_dict['3'][0]])
    xrank = 0*xminvec + 1
    xrank[np.where(xminvec == min(xminvec))] = 0
    yrank = 0*yminvec + 1
    yrank[np.where(yminvec == min(yminvec))] = 0

    print 'Initial X Rank: ', xrank
    print 'Initial Y Rank: ', yrank

    n_ll = np.where((xrank == 0) & (yrank == 0))[0][0]
    n_lr = np.where((xrank == 1) & (yrank == 0))[0][0]
    n_ur = np.where((xrank == 1) & (yrank == 1))[0][0]
    n_ul = np.where((xrank == 0) & (yrank == 1))[0][0]

    # assign lower left, lower right, etc

    ll = array_dict[str(n_ll)]
    lr = array_dict[str(n_lr)]
    ur = array_dict[str(n_ur)]
    ul = array_dict[str(n_ul)]

    ll_xbins, ll_ybins = xbins_dict[str(n_ll)], ybins_dict[str(n_ll)]
    lr_xbins, lr_ybins = xbins_dict[str(n_lr)], ybins_dict[str(n_lr)]
    ul_xbins, ul_ybins = xbins_dict[str(n_ul)], ybins_dict[str(n_ul)]
    ur_xbins, ur_ybins = xbins_dict[str(n_ur)], ybins_dict[str(n_ur)]

    # interleave arrys
    nx1 = ll.shape[0]
    nx2 = lr.shape[0]
    ny1 = ll.shape[1]
    ny2 = ul.shape[1]
    output_grid = np.empty((nx1+nx2, ny1+ny2), dtype=ll.dtype)
    output_grid[0::2,0::2] = ll
    output_grid[1::2,0::2] = lr
    output_grid[0::2,1::2] = ul
    output_grid[1::2,1::2] = ur

    # interleave edge vectors
    nxbins1 = len(ll_xbins)
    nxbins2 = len(lr_xbins)
    nybins1 = len(ll_ybins)
    nybins2 = len(ul_ybins)
    xbins = np.empty(nxbins1+nxbins2, dtype=ll_xbins.dtype)
    ybins = np.empty(nybins1+nybins2, dtype=ll_ybins.dtype)
    xbins[0::2] = ll_xbins
    xbins[1::2] = lr_xbins
    ybins[0::2] = ll_ybins
    ybins[1::2] = ul_ybins

    # define new pixel boundaries for smaller spaced grid
    output_xbins = (xbins[0:-1] + xbins[1:]) / 2.0
    output_ybins = (ybins[0:-1] + ybins[1:]) / 2.0

    # return stuff

    return output_grid, output_xbins, output_ybins

def plot_final_maps(fitsimageroot='../Results/merged_interleave', 
                AVexten='.AV', fredexten='.fred',
                imgexten='.png'):

    region1 = [11.05, 41.32, 0.47, 0.45]   # brick 5 + SF ring
    region2 = [11.29, 41.8,  0.55, 0.5]    # brick 9 + brick 15 region
    region3 = [11.585, 42.1, 0.55, 0.5]
    nregions = 3
    regions = [region1, region2, region3]

    fitsimagerootAV = fitsimageroot + AVexten
    fitsimagerootfred = fitsimageroot + fredexten

    # plot mean AV images
    filename = fitsimagerootAV + '.fits'
    print 'Opening ', filename
    # FYI, next step slow because of "north" convention swapping image
    # orientation.
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=4, cmap='hot', 
                       interpolation='nearest', aspect='auto')

    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$A_V$')

    filename = fitsimagerootAV + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # loop through subregions

    for i in range(nregions):

        r = regions[i]
        gc.recenter(r[0], r[1], width=r[2], height=r[3])
        filename = fitsimagerootAV + '.region' + str(i+1) + imgexten
        print 'Saving ', filename
        gc.save(filename, adjust_bbox='True')

    # plot fred
    filename = fitsimagerootfred + '.fits'
    print 'Opening ', filename
    # FYI, next step slow because of "north" convention swapping image
    # orientation.
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=1, cmap='seismic', 
                       interpolation='nearest', aspect='auto')

    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$f_{red}$')

    filename = fitsimagerootfred + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # loop through subregions

    for i in range(nregions):

        r = regions[i]
        gc.recenter(r[0], r[1], width=r[2], height=r[3])
        filename = fitsimagerootfred + '.region' + str(i+1) + imgexten
        print 'Saving ', filename
        gc.save(filename, adjust_bbox='True')

    return

def plot_final_brick_example(fileroot = 'ir-sf-b16-v8-st',
                             resultsdir = '../Results/',
                             imgexten='.png'):

    mapfigsize  = (14.0,9.5)
    plotfigsize = (10.0,10.0)

    print 'Increasing font size...'
    
    font = {'weight': '500',
            'size': '24'}
    plt.rc('font', **font)

    # uses fits files that result from
    # makeinterleaved_fits(fileroot='ir-sf-b16-v8-st', cleanstr='')

    results_file = resultsdir + fileroot + '.npz'

    # median extinction
    arrayname = 'bestfit_values'
    arraynum = 1
    AV, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)

    # f_red
    arrayname = 'bestfit_values'
    arraynum = 0
    fred, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                                 arrayname=arrayname, 
                                 arraynum=arraynum)

    # sigma
    arrayname = 'bestfit_values'
    arraynum = 2
    sigma, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                                  arrayname=arrayname, 
                                  arraynum=arraynum)

    # AVerr
    arrayname = 'percentile_values'
    arraynum = 3
    a1, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    arraynum = 5
    a2, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    AVerr = (a2 - a1) / 2.0

    # frederr
    arrayname = 'percentile_values'
    arraynum = 0
    a1, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    arraynum = 2
    a2, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    frederr = (a2 - a1) / 2.0

    # sigmaerr
    arrayname = 'percentile_values'
    arraynum = 6
    a1, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    arraynum = 8
    a2, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                               arrayname=arrayname, 
                               arraynum=arraynum)
    sigmaerr = (a2 - a1) / 2.0

    # numstars
    arrayname = 'quality_values'
    arraynum = 1
    a, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                              arrayname=arrayname, 
                              arraynum=arraynum)
    nstars = a

    # lnlikelihood
    arrayname = 'quality_values'
    arraynum = 0
    a, x, y = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                              arrayname=arrayname, 
                              arraynum=arraynum)
    lnlikelihood = a


    dat = np.load(results_file)
    rabin = x
    decbin = y
    print 'rabin:  ', len(rabin), '[', min(rabin), ',', max(rabin), ']'
    print 'decbin: ', len(decbin), '[', min(decbin), ',', max(decbin), ']'
    rangevec = [max(decbin), min(decbin), max(rabin), min(rabin)]
    dra = (max(rabin) - min(rabin)) * np.sin((min(decbin) + max(decbin))/2.0)
    ddec = min(decbin) + max(decbin)
    
    # Best fit results

    fitsimageroot = resultsdir + fileroot + '_interleave'
    fitsimagerootAV = fitsimageroot + '.AV'
    fitsimagerootmeanAV = fitsimageroot + '.meanAV'
    fitsimagerootfred = fitsimageroot + '.fred'
    fitsimagerootsigma = fitsimageroot + '.sigma'
    fitsimagerootAVfracerr = fitsimageroot + '.AVfracerr'
    fitsimagerootmeanAVfracerr = fitsimageroot + '.meanAVfracerr'
    fitsimagerootfrederr = fitsimageroot + '.frederr'
    fitsimagerootsigmaerr = fitsimageroot + '.sigmaerr'
    fitsimagerootsigmafracerr = fitsimageroot + '.sigmafracerr'

    # AV map

    filename = fitsimagerootAV + '.fits'
    print 'Opening ', filename
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=mapfigsize,
                          north='True')
    gc.show_colorscale(vmin=0, vmax=4, cmap='hot', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')
    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('Median $A_V$')

    filename = fitsimagerootAV + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # meanAV map

    filename = fitsimagerootmeanAV + '.fits'
    print 'Opening ', filename
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=mapfigsize,
                          north='True')
    gc.show_colorscale(vmin=0, vmax=4, cmap='hot', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')
    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('Mean $A_V$')

    filename = fitsimagerootmeanAV + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # fred map

    filename = fitsimagerootfred + '.fits'
    print 'Opening ', filename
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=mapfigsize,
                          north='True')
    gc.show_colorscale(vmin=0, vmax=1, cmap='seismic', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')
    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$f_{red}$')

    filename = fitsimagerootfred + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # sigma map

    filename = fitsimagerootsigma + '.fits'
    print 'Opening ', filename
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=mapfigsize,
                          north='True')
    gc.show_colorscale(vmin=0.2, vmax=0.6, cmap='jet', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')
    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$\sigma$')

    filename = fitsimagerootsigma + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # Uncertainty results

    # AV fracerr map

    filename = fitsimagerootAVfracerr + '.fits'
    print 'Opening ', filename
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=mapfigsize,
                          north='True')
    gc.show_colorscale(vmin=0, vmax=1, cmap='gist_ncar', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')
    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$\Delta \widetilde{A_V} / \widetilde{A_V}$')

    filename = fitsimagerootAVfracerr + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # meanAV fracerr map

    filename = fitsimagerootmeanAVfracerr + '.fits'
    print 'Opening ', filename
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=mapfigsize,
                          north='True')
    gc.show_colorscale(vmin=0, vmax=1, cmap='gist_ncar', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')
    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text(r'$\Delta \langle A_V \rangle / \langle A_V  \rangle$')

    filename = fitsimagerootmeanAVfracerr + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # fred err map

    filename = fitsimagerootfrederr + '.fits'
    print 'Opening ', filename
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=mapfigsize,
                          north='True')
    gc.show_colorscale(vmin=0, vmax=0.5, cmap='gist_ncar', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')
    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$\Delta f_{red}$')

    filename = fitsimagerootfrederr + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # sigma fracerr map

    filename = fitsimagerootsigmafracerr + '.fits'
    print 'Opening ', filename
    gc = aplpy.FITSFigure(filename, convention='wells', 
                          figsize=mapfigsize,
                          north='True')
    gc.show_colorscale(vmin=0, vmax=1, cmap='gist_ncar', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')
    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$\Delta\sigma / \sigma$')

    filename = fitsimagerootsigmafracerr + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    ####################################################

    # Best fit value scatter plots

    scatteralpha=1.0
    print 'Plotting scatter plots...'

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV, fred, c=sigma, s=7, linewidth=0, alpha=scatteralpha, 
                     vmin=0.2, vmax=0.8)
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$f_{red}$')
    plt.axis([0, 5, 0, 1.0])
    plt.annotate(r"$\sigma$",xy=(0.9,0.9), xycoords='axes fraction', xytext=None,
                 horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_fred_sigma_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(fred,sigma,c=AV,s=7,linewidth=0,alpha=scatteralpha,
                     vmin=0, vmax=4)
    plt.colorbar(im)
    plt.xlabel('$f_{red}$')
    plt.ylabel('$\sigma$')
    plt.axis([0, 1.0, 0, 1])
    plt.annotate(r"$\widetilde{A_V}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.fred_vs_sigma_AV_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV,sigma,c=fred,s=7,linewidth=0,alpha=scatteralpha,
                     vmin=0, vmax=0.75)
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\sigma$')
    plt.axis([0, 5, 0, 1])
    plt.annotate(r"$f_{red}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_sigma_fred_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV,sigma,c=lnlikelihood,s=7,linewidth=0,alpha=scatteralpha,
                     vmin=-6.3, vmax=-4.8, cmap='jet_r')
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\sigma$')
    plt.axis([0, 5, 0, 1])
    plt.annotate(r"Log ${\mathcal{L}}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_sigma_lnlike_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV, fred, c=lnlikelihood, s=7, linewidth=0, alpha=scatteralpha, 
                     vmin=-6.3, vmax=-4.8, cmap='jet_r')
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$f_{red}$')
    plt.axis([0, 5, 0, 1.0])
    plt.annotate(r"Log ${\mathcal{L}}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_fred_lnlike_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(fred,sigma,c=lnlikelihood,s=7,linewidth=0,alpha=scatteralpha,
                     vmin=-6.3, vmax=-4.8, cmap='jet_r')
    plt.colorbar(im)
    plt.xlabel('$f_{red}$')
    plt.ylabel('$\sigma$')
    plt.axis([0, 1.0, 0, 1])
    plt.annotate(r"Log ${\mathcal{L}}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.fred_vs_sigma_lnlike_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    plt.draw()

    #################################
    # Uncertainty scatter plots, vs A_V on x-axis

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV, AVerr/AV, c=nstars, s=7, linewidth=0, 
                     alpha=scatteralpha, cmap='jet_r',
                     vmin=25, vmax=125)
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\Delta\widetilde{A_V} / \widetilde{A_V}$')
    plt.axis([0, 5, 0, 1.5])
    plt.annotate(r"$N_{stars}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_AVfracerr_nstars_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV, AVerr/AV, c=lnlikelihood, s=7, linewidth=0, 
                     alpha=scatteralpha,  cmap='jet_r',
                     vmin=-6.3, vmax=-4.8)
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\Delta\widetilde{A_V} / \widetilde{A_V}$')
    plt.axis([0, 5, 0, 1.5])
    plt.annotate(r"Log ${\mathcal{L}}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_AVfracerr_lnlike_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV, frederr, c=nstars, s=7, linewidth=0, 
                     alpha=scatteralpha,  cmap='jet_r',
                     vmin=25, vmax=125)
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\Delta f_{red}$')
    plt.axis([0, 5, 0, 0.5])
    plt.annotate(r"$N_{stars}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_frederr_nstars_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV, frederr, c=lnlikelihood, s=7, linewidth=0, 
                     alpha=scatteralpha,  cmap='jet_r',
                     vmin=-6.3, vmax=-4.8)
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\Delta f_{red}$')
    plt.axis([0, 5, 0, 0.5])
    plt.annotate(r"Log ${\mathcal{L}}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_frederr_lnlike_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV, sigmaerr/sigma, c=nstars, s=7, linewidth=0,
                     alpha=scatteralpha,  cmap='jet_r',
                     vmin=25, vmax=125)
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\Delta\sigma / \sigma$')
    plt.axis([0, 5, 0, 1.5])
    plt.annotate(r"$N_{stars}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_sigmafracerr_nstars_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    ######
    plt.figure(3, figsize=plotfigsize)
    plt.clf()

    im = plt.scatter(AV, sigmaerr/sigma, c=lnlikelihood, s=7, linewidth=0,
                     alpha=scatteralpha,  cmap='jet_r',
                     vmin=-6.3, vmax=-4.8)
    plt.colorbar(im)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$\Delta\sigma / \sigma$')
    plt.axis([0, 5, 0, 1.5])
    plt.annotate(r"Log ${\mathcal{L}}$",xy=(0.9,0.9), xycoords='axes fraction', 
                 xytext=None, horizontalalignment='right', verticalalignment='top')
    
    exten = '.AV_vs_sigmafracerr_lnlike_color'
    savefile = resultsdir + fileroot + '_interleave' + exten + imgexten
    print 'Saving scatter plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)


    print 'Restoring original font defaults...'
    plt.rcdefaults()

    return



    plt.figure(5)
    plt.close()
    fig5 = plt.figure(5, figsize=plotfigsize)
    plt.clf()
    plt.suptitle(brickname)

    plt.subplot(2,2,1)
    im = plt.scatter(fred, AVerr/fred, c=AV,
                     s=7, linewidth=0, alpha=scatteralpha, 
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta A_V / A_V$')
    plt.axis([0, 5, 0, 1.5])
    
    plt.subplot(2,2,2)
    im = plt.scatter(fred, frederr, c=AV,
                     s=7, linewidth=0, alpha=scatteralpha,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta f_{red}$')
    plt.axis([0, 5, 0, 0.35])
    
    plt.subplot(2,2,3)
    im = plt.scatter(fred, sigmaerr / (sigma), c=AV,
                     s=7, linewidth=0, alpha=scatteralpha,
                     vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\Delta \sigma / \sigma)$')
    plt.axis([0, 5, 0, 1])

    if (pngroot != ''):
        plt.savefig(pngroot + '.5.png', bbox_inches=0)
        
    return

def plot_optical_comparison_images(imgroot = 'andromeda_gendler-3color',
                                   imgdir = '../../Images/',
                                   outputdir = '../Results/',
                                   imgexten = '.png'):

    region1 = [11.05, 41.32, 0.47, 0.45]   # brick 5 + SF ring
    region2 = [11.29, 41.8,  0.55, 0.5]    # brick 9 + brick 15 region
    region3 = [11.585, 42.1, 0.55, 0.5]
    nregions = 3
    regions = [region1, region2, region3]

    # set figsize to match plot_final
    figsize = (10.5, 10.5)

    # assumes have already run:
    # extract_layers_from_rgb_fits(imgroot)

    fileroot = imgdir + imgroot

    fitsfilename = fileroot + '.fits'
    print 'Displaying ', fitsfilename
    gc = aplpy.FITSFigure(fitsfilename,north='True', 
                          figsize=figsize)    
    gc.show_grayscale(aspect='auto', invert='True')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')
    
    print 'Generating sub-region images...'
    for i in range(nregions):

        r = regions[i]
        gc.recenter(r[0], r[1], width=r[2], height=r[3])
        filename = outputdir + imgroot + '.region' + str(i+1) + imgexten
        print 'Saving ', filename
        gc.save(filename, adjust_bbox='True')

    return

def plot_draine_comparison_images(imgroot = 'draine_M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.AV',
                                  imgdir = '../',
                                  outputdir = '../Results/',
                                  imgexten = '.png',
                                  rescale = 2.2):

    region1 = [11.05, 41.32, 0.47, 0.45]   # brick 5 + SF ring
    region2 = [11.29, 41.8,  0.55, 0.5]    # brick 9 + brick 15 region
    region3 = [11.585, 42.1, 0.55, 0.5]
    nregions = 3
    regions = [region1, region2, region3]

    # set figsize to match plot_final, but then adjust for colorbar
    widthadjust = 0.3
    figsize = (10.5 - widthadjust,10.5)

    # assumes have already run:
    # extract_layers_from_rgb_fits(imgroot)

    fileroot = imgdir + imgroot

    fitsfilename = fileroot + '.fits'
    print 'Displaying ', fitsfilename
    gc = aplpy.FITSFigure(fitsfilename,north='True', 
                          figsize=figsize)    
    gc.show_colorscale(vmin=0, vmax=4.0 * rescale, cmap='hot', 
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$A_V$')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')
    
    print 'Generating sub-region images...'
    for i in range(nregions):

        r = regions[i]
        gc.recenter(r[0], r[1], width=r[2], height=r[3])
        filename = outputdir + imgroot + '.region' + str(i+1) + imgexten
        print 'Saving ', filename
        gc.save(filename, adjust_bbox='True')

    return

def plot_gas_comparison_images(imgroot = 'm31_HI_goodwcs.fixedepoch',
                                  imgdir = '../',
                                  outputdir = '../Results/',
                                  imgexten = '.png',
                                  rescale = 2.2):

    region1 = [11.05, 41.32, 0.47, 0.45]   # brick 5 + SF ring
    region2 = [11.29, 41.8,  0.55, 0.5]    # brick 9 + brick 15 region
    region3 = [11.585, 42.1, 0.55, 0.5]
    nregions = 3
    regions = [region1, region2, region3]

    # set figsize to match plot_final, but then adjust for colorbar
    widthadjust = 0.3
    figsize = (10.5 - widthadjust,10.5)

    # assumes have already run:
    # extract_layers_from_rgb_fits(imgroot)

    fileroot = imgdir + imgroot

    fitsfilename = fileroot + '.fits'
    print 'Displaying ', fitsfilename
    gc = aplpy.FITSFigure(fitsfilename,north='True', 
                          figsize=figsize)    
    gc.show_colorscale(cmap='RdBu_r', vmin=0, vmax=6500,
                       interpolation='nearest', aspect='auto')
    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')
    
    print 'Generating sub-region images...'
    for i in range(nregions):

        r = regions[i]
        gc.recenter(r[0], r[1], width=r[2], height=r[3])
        filename = outputdir + imgroot + '.region' + str(i+1) + imgexten
        print 'Saving ', filename
        gc.save(filename, adjust_bbox='True')

    return

def extract_layers_from_rgb_fits(fileroot):

    img = fileroot + '.fits'

    hdulist = fits.open(img)
    hdulist.info()

    hdr = hdulist[0].header
    hdr['NAXIS'] = 2
    hdr.remove('NAXIS3')

    data = hdulist[0].data
    print data.shape
    
    data1 = data[0,:,:]
    data2 = data[1,:,:]
    data3 = data[2,:,:]

    img1 = fileroot + '.R.fits'
    img2 = fileroot + '.G.fits'
    img3 = fileroot + '.B.fits'

    fits.writeto(img1, data1, hdr)
    fits.writeto(img2, data2, hdr)
    fits.writeto(img3, data3, hdr)

    hdulist.close()

    return

def convert_draine_to_AV():

    # make a version of the drain dust mass map in terms of A_V, based on DL07 models
    #
    # Bruce's email as of May 29, 2013:  A_V = 0.74*Sigma_Md/(10^5 Msol/kpc^2)
    
    fileroot = '../draine_M31_S350_110_SSS_110_Model_All_SurfBr_Mdust'
    img = fileroot + '.fits'

    hdulist = fits.open(img)
    hdulist.info()

    hdr = hdulist[0].header

    data = hdulist[0].data
    print data.shape

    data1 = data * 0.74 / 1.e5
    
    img1 = fileroot + '.AV.fits'

    fits.writeto(img1, data1, hdr)

    hdulist.close()

    return

def fix_epoch_in_gas_header_and_flatten():

    # make a version of the drain dust mass map in terms of A_V, based on DL07 models
    #
    # Bruce's email as of May 29, 2013:  A_V = 0.74*Sigma_Md/(10^5 Msol/kpc^2)
    
    fileroot = '../m31_HI_goodwcs'
    img = fileroot + '.fits'

    hdulist = fits.open(img)
    hdulist.info()

    hdr = hdulist[0].header
    hdr['EPOCH'] = 1950.0
    hdr['NAXIS'] = 2
    hdr.remove('NAXIS3')
    hdr.remove('NAXIS4')
    hdr.remove('CTYPE3')
    hdr.remove('CRVAL3')
    hdr.remove('CDELT3')
    hdr.remove('CRPIX3')
    hdr.remove('CROTA3')
    hdr.remove('CTYPE4')
    hdr.remove('CRVAL4')
    hdr.remove('CDELT4')
    hdr.remove('CRPIX4')
    hdr.remove('CROTA4')

    data = hdulist[0].data
    print data.shape

    img1 = fileroot + '.fixedepoch.fits'

    fits.writeto(img1, data[0,0,:,:], hdr)

    hdulist.close()

    return

def plot_fred_distributions(AVthresh=1.5, AVfracerrthresh=0.15, ferrthresh=0.2, 
                            imgexten='.png', plot_error_distribution=False):

    resultsdir = '../Results/'
    fileroot = 'merged_interleave'
    AVfile = resultsdir + fileroot + '.AV.fits'
    AVerrfile = resultsdir + fileroot + '.AVerr.fits'
    ffile = resultsdir + fileroot + '.fred.fits'
    ferrfile = resultsdir + fileroot + '.ferr.fits'
    output_fred_map_file = resultsdir + fileroot + '.goodfredmap' + imgexten
    output_fred_angle_file = resultsdir + fileroot + '.fredangle' + imgexten
    
    hdulist = fits.open(AVfile)
    AV = hdulist[0].data
    hdr = hdulist[0].header
    hdulist.close()

    hdulist = fits.open(AVerrfile)
    AVerr = hdulist[0].data
    hdulist.close()

    hdulist = fits.open(ffile)
    f = hdulist[0].data
    hdulist.close()

    hdulist = fits.open(ferrfile)
    ferr = hdulist[0].data
    hdulist.close()
    
    # calculate fractional error

    i_good = np.where(AV > 0)
    i_bad =  np.where(AV <= 0)

    AVfracerr = AVerr / AV
    AVfracerr[i_bad] = 0

    # keep only high extinction, high reliability points

    i_keep = np.where((AV > AVthresh) & (AVfracerr < AVfracerrthresh) & (ferr < ferrthresh))
    i_ra = i_keep[1]
    i_dec = i_keep[0]

    # set up WCS and get ra, dec of good pixels
    
    w = wcs.WCS(hdr)
    ra_dec_coords = w.wcs_pix2world([[i_ra[i], i_dec[i]] for i in range(len(i_ra))], 1)
    ra = ra_dec_coords[:,0]
    dec = ra_dec_coords[:,1]
    print ra_dec_coords.shape

    # get angle relative to center

    # center of bulge
    m31ra  = 10.6847929
    m31dec = 41.2690650    
    dra = (ra - m31ra) * np.cos(np.math.pi * m31dec / 180.0)
    ddec = (dec - m31dec)
    theta = 90. - np.arctan(ddec / dra) * 180. / np.math.pi
    r = iAV.get_major_axis(ra, dec)
    

    # make nicer output
    print 'Increasing font size...'
    font = {'weight': '500',
            'size': '18'}
    plt.rc('font', **font)

    #
    plt.figure(1)
    plt.close()
    plt.figure(1, figsize=(12,10))
    im = plt.scatter(ra, dec, c=f[i_keep], linewidth=0, s=4,
                     vmin=0.1, vmax=0.9)
    color_bar = plt.colorbar(im)
    color_bar.ax.set_aspect(50.)
    color_bar.set_label('$f_{red}$')
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.annotate('$\widetilde{A_V} > %4.2f$' % AVthresh,
                     xy=(0.95, 0.90), fontsize=20, horizontalalignment='right',
                     xycoords = 'axes fraction')
    plt.annotate('$\Delta f_{red} < %3.1f$' % ferrthresh,
                     xy=(0.95, 0.85), fontsize=20, horizontalalignment='right',
                     xycoords = 'axes fraction')
    plt.draw()
    print 'Saving map to ', output_fred_map_file
    plt.savefig(output_fred_map_file, bbox_inches=0)


    plt.figure(3)
    plt.close()
    fig = plt.figure(3, figsize=(14,10))
    ax = fig.add_axes([.1, 0.1, 0.8, 0.8])
    im = ax.scatter(theta, f[i_keep], c=r, linewidth=0, s=10, vmin=0.15, vmax=1.3, cmap='hsv')
    color_bar = plt.colorbar(im)
    color_bar.ax.set_aspect(50.)
    color_bar.set_label('Radius (Degrees)')
    color_bar.draw_all()
    ax.axis([23,100,0.1,0.9])
    ax.set_xlabel(r'$\theta$ (degrees)')
    ax.set_ylabel(r'$f_{red}$')
    # now add zoomed inset
    ax_inset = fig.add_axes([.365, 0.41, 0.37, 0.465])
    ax_inset.scatter(theta, f[i_keep], c=r, linewidth=0, s=5, vmin=0.15, vmax=1.3, cmap='hsv')
    ax_inset.plot([25.05, 49.9], [0.5, 0.5],color='black')
    ax_inset.axis([25.05, 49.9, 0.1, 0.899])
    print 'Saving map to ', output_fred_angle_file
    plt.savefig(output_fred_angle_file, bbox_inches=0)

    if (plot_error_distribution):
        i_ok = np.where(AV > 1)
        plt.figure(2)
        plt.clf()
        im = plt.scatter(AVfracerr[i_ok], ferr[i_ok], c=AV[i_ok], linewidth=0, s=4,alpha=0.1,
                         vmin=1, vmax=3)
        plt.colorbar(im)
        plt.plot([0,0.15],[ferrthresh,ferrthresh], color='red', linewidth=3)
        plt.plot([AVfracerrthresh,AVfracerrthresh],[0,0.3], color='red', linewidth=3)
        plt.axis([0, 0.15, 0, 0.3])
        plt.xlabel('$\Delta A_V / A_V$')
        plt.ylabel('$\Delta f_{red}$')
        
    # restore font size
    print 'Restoring original font defaults...'
    plt.rcdefaults()

    return

def plot_AV_sig_vs_MW(resultsfileroot='merged', resultsdir='../Results/', imgexten='.png'):

    resultsfile = resultsdir + resultsfileroot + '.npz'
    imgfile = resultsdir + resultsfileroot + '.AV_sig_vs_MW' + imgexten
    dat = np.load(resultsfile)
    
    AV = dat['bestfit_values_clean'][:,:,1].flatten()
    sig = dat['bestfit_values_clean'][:,:,2].flatten()
    i_good = np.where(AV > 0)

    # Data from Table 2 of Lombardi, Alves, & Lada 2010, but original
    # from Kainailanen et al 2009 -- MW molecular clouds
    #
    # see milkywayMC.dat

    AK_MW  = np.array([0.15, 0.38, 0.18, 0.08, 0.16, 0.33, 0.11, 0.12, 
                       0.14, 0.10, 0.12, 0.42, 0.10, 0.14, 0.12, 0.15, 
                       0.08, 0.12, 0.13, 0.13, 0.11, 0.16, 0.14])
    AK_AV = 0.112
    AV_MW = AK_MW / AK_AV

    sig_MW = np.array([0.42, 0.28, 0.49, 0.43, 0.48, 0.51, 0.35, 0.35, 
                       0.35, 0.44, 0.32, 0.29, 0.39, 0.41, 0.38, 0.50, 
                       0.45, 0.46, 0.50, 0.48, 0.49, 0.59, 0.51])

    SF_MW  = np.array([0, 0, 1, 1, 1, 1, 1, 1, 
                       1, 1, 1, 1, 1, 1, 1, 0, 
                       0, 0, 1, 1, 1, 1, 1])

    resolution_MW = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
                              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
                              0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6])

    i_SF = np.where(SF_MW == 1)
    i_noSF = np.where(SF_MW == 0)
    i_lores = np.where(resolution_MW == 0.6)
    i_hires = np.where(resolution_MW == 0.1)
    i_SF_lores = np.where((SF_MW == 1) & (resolution_MW == 0.6))
    i_SF_hires = np.where((SF_MW == 1) & (resolution_MW == 0.1))
    i_noSF_lores = np.where((SF_MW == 0) & (resolution_MW == 0.6))
    i_noSF_hires = np.where((SF_MW == 0) & (resolution_MW == 0.1))

    AVrange = [0.,5.]
    sigrange = [0., 1.]
    nbins = [100, 100]
    hist, sigedge, AVedge = np.histogram2d(sig[i_good], AV[i_good], normed=True, 
                                           range=[sigrange, AVrange], 
                                           bins=nbins)
    extentvec = [AVedge[0], AVedge[-1], sigedge[0], sigedge[-1]]

    # make nicer output
    print 'Increasing font size...'
    font = {'weight': '500',
            'size': '18'}
    plt.rc('font', **font)

    # plot distributions
    plt.figure(1)
    plt.close()
    plt.figure(1, figsize=(10,9))
    im = plt.imshow(np.log10(hist), vmin=-3, vmax=0.5, aspect='auto', 
                    extent=extentvec, origin='lower', cmap='gray_r', interpolation='nearest')
    #plt.colorbar(im)
    plt.xlabel('$A_V$')
    plt.ylabel('$\sigma$')
    plt.plot(AV_MW[i_SF_lores], sig_MW[i_SF_lores], '*', color='red', 
             ms=18, markeredgewidth=1)
    plt.plot(AV_MW[i_noSF_lores], sig_MW[i_noSF_lores], 'o', color='red', 
             ms=10, markeredgewidth=1)
    plt.plot(AV_MW[i_SF_hires], sig_MW[i_SF_hires], '*', color='#0040FF', 
             ms=18, markeredgewidth=1)
    plt.plot(AV_MW[i_noSF_hires], sig_MW[i_noSF_hires], 'o', color='cyan', 
             ms=10, markeredgewidth=1)
    print 'Saving plot to ' + imgfile
    plt.savefig(imgfile, bbox_inches=0)

    # plot distributions
    plt.figure(2)
    plt.close()
    plt.figure(2, figsize=(10,9))
    plt.plot(AV, sig, ',', alpha=0.1, color='black')
    plt.axis(extentvec)
    plt.xlabel('$A_V$')
    plt.ylabel('$\sigma$')
    plt.plot(AV_MW[i_SF_lores], sig_MW[i_SF_lores], '*', color='red', ms=15)
    plt.plot(AV_MW[i_noSF_lores], sig_MW[i_noSF_lores], 'o', color='red', ms=10)
    plt.plot(AV_MW[i_SF_hires], sig_MW[i_SF_hires], '*', color='blue', ms=15)
    plt.plot(AV_MW[i_noSF_hires], sig_MW[i_noSF_hires], 'o', color='blue', ms=10)

    # restore font size
    print 'Restoring original font defaults...'
    plt.rcdefaults()

def plot_final_totgas_compare(fileroot='merged', resultsdir='../Results/',
                              cleanstr = '_clean', imgexten='.png',
                              write_fits = 'False', write_ratio_fits = 'True',
                              gasimgscale = 1.8e21,
                              biasfix = 0.26, ratiofix=1.0,
                              biasfixmean = 0.275, ratiofixmean=1.0,
                              smooth_img=0):

    gasfile = '../GasMaps/working/tg_old_wsrt_at_45.fits'  # in atoms / cm^2
    gasresolution = 45.0   # FWHM
    AVresolution = 6.645      # FWHM
    output_smooth_AV_root = resultsdir + fileroot + '_interleaved_gas_smoothed.AV'
    output_smooth_meanAV_root = resultsdir + fileroot + '_interleaved_gas_smoothed.meanAV'
    output_smooth_AV_file = output_smooth_AV_root + '.fits'
    output_smooth_meanAV_file = output_smooth_meanAV_root + '.fits'
    output_smooth_AV_ratio_file = output_smooth_AV_root + '.ratio.fits'
    output_smooth_meanAV_ratio_file = output_smooth_meanAV_root + '.ratio.fits'
    output_smooth_AV_ratiocorr_file = output_smooth_AV_root + '.ratiocorr.fits'
    output_smooth_meanAV_ratiocorr_file = output_smooth_meanAV_root + '.ratiocorr.fits'
        
    # median extinction
    arrayname = 'bestfit_values' + cleanstr
    arraynum = 1
    a, ra_bins, dec_bins = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                                           arrayname=arrayname, 
                                           arraynum=arraynum)
    AV = a
    print 'Size of extinction map: ', AV.shape

    # mean extinction
    if (fileroot == 'merged'):
        arrayname = 'dervied_values' + cleanstr  #  note mispelling!
    else:
        arrayname = 'derived_values' + cleanstr
    arraynum = 0
    a, ra_bins, dec_bins = interleave_maps(fileroot=fileroot, resultsdir=resultsdir,
                                           arrayname=arrayname, 
                                           arraynum=arraynum)
    meanAV = a

    # run the smoothing algorithm on both A_V maps

    if (write_fits == 'True'):

        wcs, im_coords, gasimg, AV_img = compare_img_to_AV(AV, ra_bins, dec_bins, gasfile, 
                                                           crop='True',
                                                           scaleimgfactor = 1.0,
                                                           resolution_in_arcsec=gasresolution, 
                                                           AV_resolution_in_arcsec=AVresolution, 
                                                           outputAVfile=output_smooth_AV_file)
        wcs, im_coords, gasimg, meanAV_img = compare_img_to_AV(meanAV, ra_bins, dec_bins, gasfile, 
                                                               crop='True',
                                                               scaleimgfactor = 1.0,
                                                               resolution_in_arcsec=gasresolution, 
                                                               AV_resolution_in_arcsec=AVresolution, 
                                                               outputAVfile=output_smooth_meanAV_file)

    # read from fits files (guarantees consistency -- compare_img_to_AV returns cropped version, but FITS=full)

    f = pyfits.open(output_smooth_AV_file)
    AV_img = f[0].data
    f = pyfits.open(output_smooth_meanAV_file)
    meanAV_img = f[0].data

    f = pyfits.open(gasfile)
    hdr, gasimg = f[0].header, f[0].data
    wcs = pywcs.WCS(hdr)
    
    print 'Rescaling gas image by %g for clarity' % gasimgscale
    gasimg = gasimg / gasimgscale

    # smooth images to reduce noise, if requested
    if (smooth_img > 0):

        print 'Boxcar smoothing images to reduce noise, using width ', int(smooth_img)
        gasimg = scipy.stsci.convolve.boxcar(gasimg, (int(smooth_img), int(smooth_img)))
        AV_img = scipy.stsci.convolve.boxcar(AV_img, (int(smooth_img), int(smooth_img)))
        meanAV_img = scipy.stsci.convolve.boxcar(meanAV_img, (int(smooth_img), int(smooth_img)))

    # get ra dec of gas image

    i_dec, i_ra = np.meshgrid(np.arange(gasimg.shape[1]), np.arange(gasimg.shape[0]))
    i_coords = np.array([[i_dec[i,j],i_ra[i,j]] for (i,j),val in np.ndenumerate(i_ra)])
    if (wcs.wcs.naxis == 3):
        i_coords = np.array([[i_dec[i,j],i_ra[i,j],0] for (i,j),val in np.ndenumerate(i_ra)])
    print 'gasimg.shape: ', gasimg.shape
    print 'i_dec.shape: ', i_dec.shape
    print 'i_coords.shape: ', i_coords.shape
    # solve for RA, dec at those coords
    img_coords = wcs.wcs_pix2sky(i_coords, 1)
    if (wcs.wcs.naxis > 2):
        img_coords = img_coords[:,0:2]
    img_coords = np.reshape(img_coords,(i_ra.shape[0],i_ra.shape[1],2))
    ra  = img_coords[:,:,0]
    dec = img_coords[:,:,1]

    lgnstar = np.log10(iAV.get_nstar_at_ra_dec(ra, dec, renormalize_to_surfdens=True))

    # select regions for analysis

    mask = np.where(AV_img > 0, 1.0, 0.0)
    i_masked = np.where(mask == 0)
    i_unmasked = np.where(mask > 0)
    mask_loAV = np.where(AV_img > 0.25, 1.0, 0.0)
    i_loAV = np.where((mask_loAV == 0) & (mask > 0))
    i_hiAV = np.where(mask_loAV > 0)

    AVratio = gasimg / AV_img 
    meanAVratio = gasimg / meanAV_img
    AVratiocorr = (gasimg / ratiofix) / (AV_img + biasfix)
    meanAVratiocorr = (gasimg / ratiofixmean) / (meanAV_img + biasfixmean)
    # clean up masked regions -- do replacement to deal with NaN's in divisions.
    AVratio[i_masked] = 0.0
    meanAVratio[i_masked] = 0.0
    AVratiocorr[i_masked] = 0.0
    meanAVratiocorr[i_masked] = 0.0

    # write ratio fits files

    if (write_ratio_fits == 'True'):
        print ' Writing ratio maps'
        pyfits.writeto(output_smooth_AV_ratio_file, AVratio, header=hdr)
        pyfits.writeto(output_smooth_meanAV_ratio_file, meanAVratio, header=hdr)
        pyfits.writeto(output_smooth_AV_ratiocorr_file, AVratiocorr, header=hdr)
        pyfits.writeto(output_smooth_meanAV_ratiocorr_file, meanAVratiocorr, header=hdr)

    # set minimum AV to use in plots

    AVlim = 0.325

    i_good = np.where(AV_img > AVlim)

    ################################################
    # Make plots

    # set image region

    region1 = [11.05, 41.32, 0.47, 0.45]   # brick 5 + SF ring
    region2 = [11.29, 41.8,  0.55, 0.5]    # brick 9 + brick 15 region
    region3 = [11.585, 42.1, 0.55, 0.5]
    allregion = [11.33, 41.72, 0.92, 1.28]   # brick 5 + SF ring

    # Redefine fits files, in case write_fits='False'

    output_smooth_AV_file = output_smooth_AV_root + '.fits'
    output_smooth_meanAV_file = output_smooth_meanAV_root + '.fits'

    # image of smoothed AV

    gc = aplpy.FITSFigure(output_smooth_AV_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=8, cmap='RdBu_r', 
                       interpolation='nearest', aspect='auto')

    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$A_V$')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])
    filename = output_smooth_AV_root + '.region1' + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of meanAV smoothed

    gc = aplpy.FITSFigure(output_smooth_meanAV_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=8, cmap='RdBu_r', 
                       interpolation='nearest', aspect='auto')

    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text('$A_V$')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])
    filename = output_smooth_meanAV_root + '.region1' + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of smoothed AV ratio

    ratiomax = 3.

    gc = aplpy.FITSFigure(output_smooth_AV_ratio_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=ratiomax, cmap='spectral', 
                       interpolation='nearest', aspect='auto')
    # add AV level contours
    print 'Adding contours from ',output_smooth_AV_file
    gc.show_contour(output_smooth_AV_file, levels=[1.0], convention='wells',
                    colors='black', linewidths=4)


    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text(r'$A_{V,gas} / \widetilde{A_V}$')

    r = allregion
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region1'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region2
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region2'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region3
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region3'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of smoothed meanAV ratio

    gc = aplpy.FITSFigure(output_smooth_meanAV_ratio_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=ratiomax, cmap='spectral', 
                       interpolation='nearest', aspect='auto')
    # add AV level contours
    gc.show_contour(output_smooth_meanAV_file, levels=[1.0], convention='wells',
                    colors='black', linewidths=4)


    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    gc.colorbar.set_axis_label_text(r'$A_{V,gas} / \langle A_V \rangle$')

    r = allregion
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region1'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region2
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region2'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region3
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiomap.region3'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of smoothed AV corrected ratio

    gc = aplpy.FITSFigure(output_smooth_AV_ratiocorr_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=2, cmap='seismic', 
                       interpolation='nearest', aspect='auto')
    # add AV level contours
    gc.show_contour(output_smooth_AV_file, levels=[1.0], convention='wells',
                    colors='black', linewidths=4)


    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    if (ratiofix != 1): 
        gc.colorbar.set_axis_label_text(r'$(A_{V,gas} / %4.2f) / (\widetilde{A_V} + %4.2f)$' % (ratiofix, biasfix))
    else:
        gc.colorbar.set_axis_label_text(r'$A_{V,gas} / (\widetilde{A_V} + %4.2f)$' % biasfix)


    r = allregion
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region1'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region2
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region2'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region3
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region3'
    filename = output_smooth_AV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    # image of smoothed meanAV ratio

    gc = aplpy.FITSFigure(output_smooth_meanAV_ratiocorr_file, convention='wells', 
                          figsize=(10.5,10.5),
                          north='True')
    gc.show_colorscale(vmin=0, vmax=2, cmap='seismic', 
                       interpolation='nearest', aspect='auto')
    # add AV level contours
    gc.show_contour(output_smooth_meanAV_file, levels=[1.0], convention='wells',
                    colors='black', linewidths=4)

    gc.set_tick_labels_format(xformat='ddd.d', yformat='ddd.d')

    gc.add_grid()
    gc.grid.set_alpha(0.1)
    gc.grid.set_xspacing('tick')
    gc.grid.set_yspacing('tick')

    gc.add_colorbar()
    gc.colorbar.set_width(0.15)
    gc.colorbar.set_location('right')
    if (ratiofix != 1):
        gc.colorbar.set_axis_label_text(r'$(A_{V,gas} / %4.2f) / (\langle A_V \rangle + %4.2f)$' % (ratiofixmean, biasfixmean))
    else:
        gc.colorbar.set_axis_label_text(r'$A_{V,gas} / (\langle A_V \rangle + %4.2f)$' % biasfixmean)

    r = allregion
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region1
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region1'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region2
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region2'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    r = region3
    gc.recenter(r[0], r[1], width=r[2], height=r[3])

    exten = '.ratiocorrmap.region3'
    filename = output_smooth_meanAV_root + exten + imgexten
    print 'Saving ', filename
    gc.save(filename, adjust_bbox='True')

    ################################################
    # adopt larger fonts and set alpha value

    print 'Increasing font size...'
    
    font = {'weight': '500',
            'size': '18'}
    plt.rc('font', **font)
    
    alpha = 1.0
    scatteralpha = 0.3
    scattersize = 3
    greyval = '#B3B3B3'
    plotfigsize = (10.0,10.0)

    # Correlation plot AV  (ghost out low AV points)

    plt.figure(11, figsize=plotfigsize)
    plt.close()
    plt.figure(11, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(AV_img[i_loAV], gasimg[i_loAV], ',', color=greyval, alpha=alpha)
    #im = plt.plot(AV_img[i_hiAV], gasimg[i_hiAV], ',', color='black', alpha=alpha)
    im = plt.scatter(AV_img[i_hiAV], gasimg[i_hiAV], c=lgnstar[i_hiAV], vmin=-1.2, vmax=np.log10(3.0),
                     linewidth=0, s=scattersize, cmap='gist_ncar', alpha=scatteralpha)
    cb = plt.colorbar(im)
    cb.set_alpha(1)
    cb.ax.set_aspect(50.)
    cb.set_label('Log$_{10} \Sigma_{stars}$  (arcsec$^{-2})$')
    cb.draw_all()
    plt.xlabel(r'$\widetilde{A_V}$')
    plt.ylabel(r'$A_{V,gas}$')
    plt.axis([0, 3.5, -0.25, 3.5])
    
    exten = '.correlation'
    savefile = output_smooth_AV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # Correlation plot meanAV (ghost out low AV points)

    plt.figure(12, figsize=plotfigsize)
    plt.close()
    plt.figure(12, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(meanAV_img[i_loAV], gasimg[i_loAV], ',', color=greyval, alpha=alpha)
    #im = plt.plot(meanAV_img[i_hiAV], gasimg[i_hiAV], ',', color='black', alpha=alpha)
    im = plt.scatter(meanAV_img[i_hiAV], gasimg[i_hiAV], c=lgnstar[i_hiAV], vmin=-1.2, vmax=np.log10(3.0),
                     linewidth=0, s=scattersize, cmap='gist_ncar', alpha=scatteralpha)
    cb = plt.colorbar(im)
    cb.set_alpha(1)
    cb.ax.set_aspect(50.)
    cb.set_label('Log$_{10} \Sigma_{stars}$  (arcsec$^{-2})$')
    cb.draw_all()
    plt.xlabel(r'$\langle A_V \rangle$')
    plt.ylabel(r'$A_{V,gas}$')
    plt.axis([0, 3.5, -0.25, 3.5])
    
    exten = '.correlation'
    savefile = output_smooth_meanAV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # corrected Correlation plot AV  (ghost out low AV points)

    plt.figure(21, figsize=plotfigsize)
    plt.close()
    plt.figure(21, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(AV_img[i_loAV] + biasfix, gasimg[i_loAV] / ratiofix, ',', color=greyval, alpha=alpha)
    #im = plt.plot(AV_img[i_hiAV] + biasfix, gasimg[i_hiAV] / ratiofix, ',', color='black', alpha=alpha)
    im = plt.scatter(AV_img[i_hiAV] + biasfix, gasimg[i_hiAV] / ratiofix, c=lgnstar[i_hiAV], vmin=-1.2, vmax=np.log10(3.0),
                     linewidth=0, s=scattersize, cmap='gist_ncar', alpha=scatteralpha)
    cb = plt.colorbar(im)
    cb.set_alpha(1)
    cb.ax.set_aspect(50.)
    cb.set_label('Log$_{10} \Sigma_{stars}$  (arcsec$^{-2})$')
    cb.draw_all()
    plt.plot([-1,10], [-1,10], color='red', linewidth=4)
    plt.xlabel(r'$\widetilde{A_V} + %4.2f$' % biasfix)
    if (ratiofix != 1):
        plt.ylabel(r'$A_{V,gas} / %4.2f$' % ratiofix)
    else:
        plt.ylabel(r'$A_{V,gas}$')
    plt.axis([-0.25, 3.5, -0.25, 3.5])
    
    exten = '.correlationcorr'
    savefile = output_smooth_AV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # corrected Correlation plot AV  (ghost out low AV points)

    plt.figure(22, figsize=plotfigsize)
    plt.close()
    plt.figure(22, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(meanAV_img[i_loAV] + biasfixmean, gasimg[i_loAV] / ratiofixmean, ',', color=greyval, alpha=alpha)
    #im = plt.plot(meanAV_img[i_hiAV] + biasfixmean, gasimg[i_hiAV] / ratiofixmean, ',', color='black', alpha=alpha)
    im = plt.scatter(meanAV_img[i_hiAV] + biasfixmean, gasimg[i_hiAV] / ratiofixmean, c=lgnstar[i_hiAV], vmin=-1.2, vmax=np.log10(3.0),
                     linewidth=0, s=scattersize, cmap='gist_ncar', alpha=scatteralpha)
    cb = plt.colorbar(im)
    cb.set_alpha(1)
    cb.ax.set_aspect(50.)
    cb.set_label('Log$_{10} \Sigma_{stars}$  (arcsec$^{-2})$')
    cb.draw_all()
    plt.plot([-1,10], [-1,10], color='red', linewidth=4)
    plt.xlabel(r'$\langle A_V \rangle + %4.2f$' % biasfixmean)
    if (ratiofixmean != 1):
        plt.ylabel(r'$A_{V,gas} / %4.2f$' % ratiofixmean)
    else:
       plt.ylabel(r'$A_{V,gas}$')
    plt.axis([-0.25, 3.5, -0.25, 3.5])
    
    exten = '.correlationcorr'
    savefile = output_smooth_meanAV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # Ratio plot AV  (ghost out low AV points)

    AVvec = np.linspace(0.001,10,100)
    ratiovec = ratiofix * (1.0 + biasfix/AVvec)

    plt.figure(13, figsize=plotfigsize)
    plt.close()
    plt.figure(13, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(AV_img[i_loAV], gasimg[i_loAV] / AV_img[i_loAV], ',', color=greyval, alpha=alpha)
    #im = plt.plot(AV_img[i_hiAV], gasimg[i_hiAV] / AV_img[i_hiAV], ',', color='black', alpha=alpha)
    im = plt.scatter(AV_img[i_hiAV], gasimg[i_hiAV] / AV_img[i_hiAV], c=lgnstar[i_hiAV], vmin=-1.2, vmax=np.log10(3.0),
                     linewidth=0, s=scattersize, cmap='gist_ncar', alpha=scatteralpha)
    cb = plt.colorbar(im)
    cb.set_alpha(1)
    cb.ax.set_aspect(50.)
    cb.set_label('Log$_{10} \Sigma_{stars}$  (arcsec$^{-2})$')
    cb.draw_all()
    plt.plot(AVvec, ratiovec, color='red', linewidth=4)
    plt.xlabel('$\widetilde{A_V}$')
    plt.ylabel('$A_{V,gas}  /  \widetilde{A_V}$')
    plt.axis([0, 3.5, 0, 7])
    
    exten = '.ratio'
    savefile = output_smooth_AV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # Ratio plot meanAV  (ghost out low AV points)

    ratiovec = ratiofixmean * (1.0 + biasfixmean/AVvec)

    plt.figure(14, figsize=plotfigsize)
    plt.close()
    plt.figure(14, figsize=plotfigsize)
    plt.clf()
    im = plt.plot(meanAV_img[i_loAV], gasimg[i_loAV] / meanAV_img[i_loAV], ',', 
                  color=greyval, alpha=alpha)
    #im = plt.plot(meanAV_img[i_hiAV], gasimg[i_hiAV] / meanAV_img[i_hiAV], ',', 
    #              color='black', alpha=alpha)
    im = plt.scatter(meanAV_img[i_hiAV], gasimg[i_hiAV] / meanAV_img[i_hiAV], c=lgnstar[i_hiAV], vmin=-1.2, vmax=np.log10(3.0),
                     linewidth=0, s=scattersize, cmap='gist_ncar', alpha=scatteralpha)
    cb = plt.colorbar(im)
    cb.set_alpha(1)
    cb.ax.set_aspect(50.)
    cb.set_label('Log$_{10} \Sigma_{stars}$  (arcsec$^{-2})$')
    cb.draw_all()
    plt.plot(AVvec, ratiovec, color='red', linewidth=4)
    plt.xlabel(r'$\langle A_V \rangle$')
    plt.ylabel(r'$A_{V,gas}  /  \langle A_V \rangle$')
    plt.axis([0, 3.5, 0, 7])
    
    exten = '.ratio'
    savefile = output_smooth_meanAV_root + exten + imgexten
    print 'Saving correlation plot to ', savefile
    plt.savefig(savefile, bbox_inches=0)

    # restore font size

    print 'Restoring original font defaults...'
    plt.rcdefaults()

    return


