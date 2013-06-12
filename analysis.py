import pylab
import math
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import read_brick_data as rbd
import makefakecmd as mfc
import isolatelowAV as iAV
from scipy.ndimage import filters as filt
from random import sample
import ezfig  # morgan's plotting code
import pyfits as pyfits
import pywcs as pywcs
import makefakecmd as mfc

def merge_results(savefilelist=['ir-sf-b14-v8-st.npz', 'newfit_test.npz'], resultsdir='../Results/',
                  mergefileroot='merged'):

    savefilelist = ['ir-sf-b02-v8-st.npz', 'ir-sf-b04-v8-st.npz', 'ir-sf-b05-v8-st.npz', 
                    'ir-sf-b06-v8-st.npz', 'ir-sf-b08-v8-st.npz', 'ir-sf-b09-v8-st.npz', 
                    'ir-sf-b12-v8-st.npz', 'ir-sf-b14-v8-st.npz', 'ir-sf-b15-v8-st.npz', 
                    'ir-sf-b16-v8-st.npz', 'ir-sf-b17-v8-st.npz', 'ir-sf-b18-v8-st.npz', 
                    'ir-sf-b19-v8-st.npz', 'ir-sf-b21-v8-st.npz', 'ir-sf-b22-v8-st.npz', 
                    'ir-sf-b23-v8-st.npz']
    mergefile = resultsdir + mergefileroot + '.npz'
    pngfileroot = resultsdir + mergefileroot

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
    nz_quality = 2
    bestfit_values = np.zeros([nx, ny, nz_bestfit])
    percentile_values = np.zeros([nx, ny, nz_sigma])
    quality_values = np.zeros([nx, ny, nz_quality])

    # loop through list of files

    for i, savefile in enumerate(savefilelist):

        print 'Merging ', resultsdir + savefilelist[i]

        dat = np.load(resultsdir + savefilelist[i])
        bf = dat['bestfit_values']
        p  = dat['percentile_values']
        q  = dat['quality_values']
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
          bad_pix_mask = bad_pix_mask,
          bestfit_values_clean = bestfit_values_clean,
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
    sig_A_V = dat['bestfit_values'][:,:,2]
    sig = dat['bestfit_values'][:,:,2] * dat['bestfit_values'][:,:,1]
    stdvec = (dat['percentile_values'][:,:,[2,5,8]] - dat['percentile_values'][:,:,[0,3,6]]) / 2.0
    medvec = dat['percentile_values'][:,:,[1,4,7]]
    f_std = stdvec[:,:,0]
    f_med = medvec[:,:,0]
    A_V_std = stdvec[:,:,1]
    A_V_med = medvec[:,:,1]
    sig_A_V_std = stdvec[:,:,2]
    sig_A_V_med = medvec[:,:,2]
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
        sig_A_V_std = sig_A_V_std[indices[0],indices[1]]
        sig_A_V_med = sig_A_V_med[indices[0],indices[1]]

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
    plt.plot(sig_A_V, sig_A_V_std, ',', **kwargs)
    plt.axis([-0.05, 2, -0.05, 1])
    plt.xlabel('$\sigma / A_V$')
    plt.ylabel('$\Delta(\sigma / A_V)$')

    plt.subplot(3,3,8)
    plt.plot(sig_A_V, sig_A_V_med - sig_A_V, ',', **kwargs)
    plt.axis([-0.05, 2, -1, 1])
    plt.xlabel('$\sigma / A_V$')
    plt.ylabel(' Median $\sigma / A_V$ - Bestfit $\sigma / A_V$')

    plt.subplot(3,3,9)
    plt.plot(sig_A_V_std, abs(sig_A_V_med - sig_A_V), ',', **kwargs)
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
    #x0 = np.interp(ra0, rabins, i_ra) + 1.0
    #y0 = np.interp(dec0, decbins, i_dec) + 1.0
    y0 = np.interp(ra0, rabins, i_ra) + 1.0
    x0 = np.interp(dec0, decbins, i_dec) + 1.0
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
    #hdr.update('CDELT1', dra, 'd(RA*cos(Dec))/dx')
    #hdr.update('CDELT2', ddec, 'd(Dec)/dy')
    ############
    #hdr.update('RA_TAN', m31ra, 'Tangent Point')
    #hdr.update('DEC_TAN', m31dec, 'Tangent Point')
    
    pyfits.writeto(fitsfilename, array.T, header=hdr)

    return

def make_radius_image(fitsfilename, rabins, decbins):

    # position of pixel centers
    racenvec  = (rabins[0:-2]  +  rabins[1:-1]) / 2.0
    deccenvec = (decbins[0:-2] + decbins[1:-1]) / 2.0

    ra, dec = np.meshgrid(racenvec, deccenvec)
    r = iAV.get_major_axis(ra, dec)

    make_fits_image(fitsfilename, r, rabins, decbins)
    make_fits_image('ra_image.fits', ra, rabins, decbins)

    return r, ra, dec
    
            
# read in image, and match resolution & astrometry of extinction map to it
def compare_img_to_AV(imgfile, resolution_in_arcsec='', scaleimgfactor=1.0,
                      crop='True', outputAVfile='', AVdatafile='../Results/merged.npz',
                      usemeanAV=True):

    f = pyfits.open(imgfile)
    hdr, img = f[0].header, f[0].data
    wcs = pywcs.WCS(hdr)

    # scale image by arbitrary factor to bring onto AV
    img = img * scaleimgfactor

    # make grid of RA and Dec at each pixel
    i_ra, i_dec = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    i_coords = np.array([[i_ra[i,j],i_dec[i,j]] for (i,j),val in np.ndenumerate(i_ra)])

    #i_coords = np.array([[i_ra[20,30], i_dec[20,30]]])

    # solve for RA, dec at those coords
    img_coords = wcs.wcs_pix2sky(i_coords, 1)
    img_coords = np.reshape(img_coords,(i_ra.shape[0],i_ra.shape[1],2))
    ra_img  = img_coords[:,:,0]
    dec_img = img_coords[:,:,1]

    # read A_V data

    print 'Reading merged AV file.'
    avdat = np.load(AVdatafile)
    AV = avdat['bestfit_values_clean'][:,:,1]
    stddev_over_AV = avdat['bestfit_values_clean'][:,:,2]
    if usemeanAV:
        print 'Using Mean A_V, not median'
        ###  NOTE!!! Incorrect -- compensates for error in runs 1-3!!!!
        sig = np.log((1. + np.sqrt(1. + 4. * (stddev_over_AV)**2)) / 2.)
        AV = AV * np.exp(sig**2 / 2.0)
    ra_bins = avdat['ra_bins']
    dec_bins = avdat['dec_bins']
    racenvec  = (ra_bins[0:-2]  +  ra_bins[1:-1]) / 2.0
    deccenvec = (dec_bins[0:-2] + dec_bins[1:-1]) / 2.0
    ra_AV, dec_AV = np.meshgrid(racenvec, deccenvec)

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

    if (resolution_in_arcsec != ''): 
        smootharcsec = np.sqrt(resolution_in_arcsec**2 - pixscale**2)
        smoothpix = smootharcsec / pixscale
        print 'Smoothing to ',resolution_in_arcsec,' using ',smoothpix,' pixel wide filter'

        AVsmooth = filt.gaussian_filter(AV, smoothpix, mode='reflect')
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
    x_AV = np.round(x_AV+1).astype(int)  # why +1? seems to work best...
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

def compare_draine_dust(AVdatafile='../Results/SecondRunLoRes/merged.npz'):

    drainefile = '../draine_M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.fits'
    resolution = 10.0  # arcsec
    #resolution = ''  # arcsec
    outputfile = '../Results/FirstRun/draine_matched_AV.fits'
    scalefac = 0.74 / 1.e5  # Draine email May 29, 2013

    wcs, im_coords, img, AV_img = compare_img_to_AV(drainefile, crop='True',
                                                    scaleimgfactor = scalefac,
                                                    resolution_in_arcsec=resolution, 
                                                    outputAVfile='', AVdatafile=AVdatafile,
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
    
