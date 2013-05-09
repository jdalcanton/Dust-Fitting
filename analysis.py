import pylab
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import read_brick_data as rbd
import makefakecmd as mfc
from scipy.ndimage import filters as filt
from random import sample
import ezfig  # morgan's plotting code

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
            

     

