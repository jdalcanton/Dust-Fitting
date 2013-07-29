import math
import numpy as np
import read_brick_data as rbd
import makefakecmd as mfcmd
import os.path as op
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage
import pylab
import PIL
import time
import ezfig  # morgan's plotting code
import random as rnd
#import triangle

def get_major_axis(ra, dec, m31ra=10.6847929, m31dec = 41.2690650, pa=38.5, incl=74.):

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
    r = np.sqrt((decoff * math.cos(m31pa) + raoff * math.sin(m31pa))**2 +
                (decoff * math.sin(m31pa) - raoff * math.cos(m31pa))**2 / (1.0 - ecc**2))
    
    return r

def nstar_multigauss(param, ra, dec, ncomp=10):

    # param = [ra0, dec0, pa, incl, n0, sig0] -- last 4 repeated for each gaussian
    # p_init = [10.6847929, 41.2690650, 38.5, 74.0, 300., 0.5]
    # dat = np.load('../Unreddened/FourthRunHiRes/allbricks.npz')
    # ra = dat['ragridval']
    # dec = dat['decgridval']
    
    # add up gaussians at each ra, dec
    npar = 4
    nstar = np.sum(np.array([((param[4+i*npar] / np.sqrt(2. * np.pi * param[5+i*npar])) 
                              * np.exp(-0.5 * (get_major_axis(ra, dec, 
                                                              m31ra=param[0], m31dec=param[1], 
                                                              pa=param[2+i*npar], 
                                                              incl=param[3+i*npar]) 
                                               / param[5+i*npar])**2)) 
                             for i in range(0, int(ncomp))]), 
                   axis = 0)

    return nstar

def nstar_multidisk(param, ra, dec, ncomp=10):

    # param = [ra0, dec0, pa, incl, n0, sig0] -- last 4 repeated for each gaussian
    # p_init = [10.6847929, 41.2690650, 38.5, 74.0, 300., 0.5]
    # dat = np.load('../Unreddened/FourthRunHiRes/allbricks.npz')
    # ra = dat['ragridval']
    # dec = dat['decgridval']
    
    # add up gaussians at each ra, dec
    npar = 4
    nstar = np.sum(np.array([(param[4+i*npar]
                              * np.exp(-(get_major_axis(ra, dec, 
                                                        m31ra=param[0], 
                                                        m31dec=param[1], 
                                                        pa=param[2+i*npar], 
                                                        incl=param[3+i*npar]) 
                                               / param[5+i*npar])))
                             for i in range(0, int(ncomp))]), 
                   axis = 0)

    return nstar

############# CODE FOR RUNNING EMCEE AND PLOTTING RESULTS ###################

def likelihoodfunc(param, ra, dec, nstar, ncomp=10):
    """
    Return likelihood compared to map of number of stars to pixel,
    assuming Poisson statistics.
    (not true in low density regimes, but close enough)
    see http://www.astro.ubc.ca/people/jvw/ASTROSTATS/Answers/Chap2/max%20like%20and%20poisson.pdf
    see http://www-cdf.fnal.gov/physics/statistics/recommendations/modeling.html
    and http://www.physics.utah.edu/~detar/phys6720/handouts/curve_fit/curve_fit/node2.html
    """

    # calculate expected number per pixel map corresponding to parameters

    #print ncomp
    mu = nstar_multigauss(param, ra, dec, ncomp=ncomp)
    #mu = nstar_multidisk(param, ra, dec, ncomp=ncomp)

    # calculate weights to decrease weight given to inner regions

    sig_weight = 10.0
    weight = np.exp(-0.5*(np.log10(mu) / sig_weight)**2)
    
    # calculate log likelihood based on Poisson distribution

    lnp =  (weight*(nstar * np.log(mu) - mu)).sum()

    return lnp

def ln_prob(param, ra, dec, nstar, ncomp=10):

    lnp = ln_priors(param, ncomp=ncomp)

    if not np.isfinite(lnp):
        return -np.inf

    return lnp + likelihoodfunc(param, ra, dec, nstar, ncomp=ncomp)
    
def ln_priors(param, return_prior_parameters=False, ncomp=10):
    
    # param = [ra0, dec0, pa, incl, n0, sig0] -- last 4 repeated for each gaussian
    
    npar = 4
    i = np.arange(0,ncomp)
    i_pa   = 2 + i*npar
    i_incl = 3 + i*npar
    i_n0   = 4 + i*npar
    i_sig0 = 5 + i*npar

    # set up ranges

    radeg  = np.pi / 180.     # conversion from degrees to radians
    m31ra  = 10.6847929
    m31dec = 41.2690650
    ddec = (15.00 / 3600.) * radeg
    dra = ddec / math.cos(m31dec * radeg)
    ra  = [m31ra  - dra,   m31ra + dra]
    dec = [m31dec - ddec, m31dec + ddec]

    pa = [-90, 90]
    pa = [30, 55]
    incl = [40, 85]
    n0 = [0, 1000]
    sig0 = [0.02, 2.0] 

    if return_prior_parameters:

        return {'ra': ra, 'dec': dec,
                'pa': pa, 'incl': incl,
                'n0': n0, 'sig0': sig0}
    else: 

        # return -Inf if the parameters are out of range, or n0 aren't in ascending order

        ra_check   = (ra[0] < param[0])  & (param[0] < ra[1])
        dec_check  = (dec[0] < param[1]) & (param[1] < dec[1])

        pa_check   = bool(((  pa[0] < param[i_pa])   & (  param[i_pa] < pa[1]  )).all())
        incl_check = bool(((incl[0] < param[i_incl]) & (param[i_incl] < incl[1])).all())
        n0_check   = bool(((  n0[0] < param[i_n0])   & (  param[i_n0] < n0[1]  )).all())
        sig0_check = bool(((sig0[0] < param[i_sig0]) & (param[i_sig0] < sig0[1])).all())

        #n0_order = bool(np.array([param[i_n0[i]] < param[i_n0[i+1]] 
        #                          for i in range(0, len(i_n0)-1)]).all())
        #print 'RA:   ', ra_check
        #print 'Dec:  ', dec_check
        #print 'PA:   ', pa_check
        #print 'Incl: ', incl_check
        #print 'N:    ', n0_check
        #print 'Sig:  ', sig0_check
        #print 'Norder:  ', n0_order

        # if all parameters are in range, return a constant prior

        #if (pa_check & incl_check & n0_check & sig0_check & ra_check & dec_check & n0_order):
        if (pa_check & incl_check & n0_check & sig0_check & ra_check & dec_check):
            return 0.0001

        # else, return -Inf

        return -np.Inf

def fit_disk(ncomp = 10, d_dec=15., use_prev_results=True):

    # load data file

    dat = np.load('../Unreddened/allbricks.npz')
    ra = dat['ragridval']
    dec = dat['decgridval']
    nsb = dat['nstargridval']

    # convert surface density to number per pixel for poisson fit

    nstar = nsb * d_dec**2
    maxn = np.max(nstar)
    maxr = np.max(get_major_axis(ra, dec))
    print 'Maximum Nstar: ', maxn, '   Maximum radius: ', maxr
    
    # cut out central regions

    rcut = 0.085  # in degrees
    r = get_major_axis(ra, dec, pa=50., incl=60.0)
    i_keep = np.where(r > rcut)
    ra  =  ra[i_keep]
    dec = dec[i_keep]
    nsb = nsb[i_keep]
    nstar = nstar[i_keep]

    # set up default point size in maps based on number of grid points (s=1 for d_arcsec=10 case)
    print len(nstar), ' grid points'
    s_size = 1.0 * (48185. / len(nstar))

    # display initial data

    plt.figure(1)
    plt.clf()
    im = plt.scatter(ra, dec, c=np.log10(nstar), s=s_size, 
                     linewidth=0, cmap='gist_ncar', alpha=0.5,
                     vmin = 0.7, vmax=np.log10(maxn))
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('Log$_{10} N_{stars}$  (pixel$^{-1}$)')
    
    # initilize parameters

    ra0  = 10.6847929
    dec0 = 41.2690650
    n0 = 100.0
    pa0 = 38.5
    incl0 = 76.
    sig0 =  1.1                  # 0.75 * maxr

    radeg  = np.pi / 180.     # conversion from degrees to radians
    ddec = (2.0 / 3600.) * radeg
    dra = ddec / math.cos(dec0 * radeg)

    nstep   = np.arange(1, ncomp+1, dtype='float') / float(ncomp+1)
    pavec   = pa0 + 0*nstep
    inclvec = incl0 + 0*nstep
    #nvec    = maxn * nstep
    #sigvec  = sig0 + 0*nstep
    nvec    = n0  + 0*nstep
    sigvec  = maxr * nstep

    param_init = [ra0, dec0]
    for i in np.arange(0, ncomp):
        param_init.append(pavec[i])
        param_init.append(inclvec[i])
        param_init.append(nvec[i])
        param_init.append(sigvec[i])
    param_init = np.array(param_init)

    # Use decent previous fits as initial guess, then add on component

    if (use_prev_results):

        if (ncomp > 1):
            nref = 1
            p1 = [3.93645194e+01,   7.30858338e+01,   7.11295877e+02,   4.90161893e-01]
            param_init[-nref*4:] = p1
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p1, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p1, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev
            
        if (ncomp > 2):
            nref = 2
            p2 = [4.73166865e+01,   5.99870583e+01, 4.81336935e+02,   1.39948604e-01,   
                  3.98414192e+01,   7.30185283e+01, 5.03869301e+02,   5.55158752e-01]
            p2 = [4.14808984e+01,   6.73164528e+01,   6.50561463e+02,   2.21283498e-01,   
                  4.19622955e+01,   7.07158849e+01,   2.85669750e+02,   6.85786748e-01]
            param_init[-nref*4:] = p2
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p2, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p2, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev
                
                
        if (ncomp > 3):
            nref = 3
            p3 = [5.05459045e+01,   7.22253097e+01, 1.69318973e+02,   4.90886494e-01,   
                  4.57758589e+01,   5.87937584e+01, 4.65367395e+02,   1.36488059e-01,   
                  3.73440133e+01,   7.53184149e+01, 3.65562158e+02,   6.02162788e-01]
            p3 = [3.41267976e+01,   7.75642217e+01,   1.82451671e+02,   6.75914073e-01,   
                  4.84803167e+01,   6.28895179e+01,   5.48883102e+02,   1.44046825e-01,   
                  4.47832770e+01,   7.18988721e+01,   3.21356219e+02,   5.40801785e-01]
            param_init[-nref*4:] = p3
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p3, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p3, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev

        if (ncomp > 4):
            nref = 4
            p4 = [4.25718259e+01,   6.85019072e+01,  1.37502015e+02,   5.22236801e-01,   
                  5.01858117e+01,   7.29953410e+01,  1.42362750e+02,   4.91876915e-01,   
                  4.56888073e+01,   5.80334262e+01,  4.44301388e+02,   1.37071615e-01,   
                  3.69972070e+01,   7.78414475e+01,  2.62300404e+02,   6.31635433e-01]
            p4 = [3.82787028e+01,   7.64413805e+01,   1.39672344e+02,   6.50552884e-02,   
                  3.59755493e+01,   7.74631162e+01,   2.02073577e+02,   6.50340923e-01,   
                  4.72949787e+01,   5.82485668e+01,   4.61504742e+02,   1.33771259e-01,   
                  4.46486702e+01,   7.17130298e+01,   3.27427644e+02,   5.14422367e-01]
            param_init[-nref*4:] = p4
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p4, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p4, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev
            
        if (ncomp > 5):
            nref = 5
            p5 = [3.66735629e+01,   8.12548924e+01,   9.45814304e+01,   5.77153003e-01,   
                  3.92733680e+01,   6.66999022e+01,   8.31095063e+01,   4.86859612e-01,   
                  5.17872689e+01,   7.31786701e+01,   1.28726070e+02,   4.45730868e-01,   
                  4.80393788e+01,   5.60482349e+01,   4.23824688e+02,   1.28115895e-01,   
                  3.89856718e+01,   7.44761054e+01,   2.60538878e+02,   6.13915620e-01]
            p5 = [4.08296975e+01,   8.42122020e+01,   1.40518124e+02,   6.04942648e-02,   
                  5.10408325e+01,   5.57462295e+01,   4.46432252e+02,   1.24575915e-01,   
                  4.14984976e+01,   8.34743383e+01,   1.27493218e+02,   2.23111004e-01,   
                  4.49562646e+01,   7.11882154e+01,   3.28936955e+02,   5.08416423e-01,   
                  3.60896513e+01,   7.82635200e+01,   1.96937378e+02,   6.76373443e-01]
            p5 = [5.45621018e+01,   7.33165211e+01,   1.32833299e+02,   7.31329102e-01,   
                  2.41546127e+01,   7.60988458e+01,   1.35220363e+02,   6.88612435e-02,   
                  3.50019957e+01,   7.74636875e+01,   1.71080350e+02,   7.14089280e-01,   
                  5.09236719e+01,   5.83239977e+01,   4.18996689e+02,   1.38103422e-01,   
                  3.81504493e+01,   7.28212443e+01,   2.35785022e+02,   4.31320994e-01]
            param_init[-nref*4:] = p5
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p5, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p5, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev

        if (ncomp > 6):
            nref = 6
            p6 = [2.72140779e+01,   4.05102546e+01,   2.54002075e+01,   1.99634614e+00,   
                  3.74808868e+01,   7.08523242e+01,   1.36093594e+02,   5.31173551e-01,   
                  3.98347190e+01,   7.21438716e+01,   9.67724580e+01,   5.16746863e-01,   
                  5.84280282e+01,   7.85772793e+01,   8.96928741e+01,   5.71914351e-01,   
                  4.63917417e+01,   5.79697778e+01,   4.40509331e+02,   1.31587621e-01,   
                  3.85979405e+01,   7.68760339e+01,   2.22796953e+02,   5.63111841e-01]
            p6 = [3.80186206e+01,   7.91001790e+01,   1.13462417e+02,   6.54336183e-02,   
                  4.84068268e+01,   5.27374660e+01,   4.18219432e+02,   1.16002168e-01,   
                  4.55326344e+01,   8.30037721e+01,   1.22214507e+02,   1.81992183e-01,   
                  4.72426967e+01,   7.26182257e+01,   2.80970312e+02,   4.61273919e-01,   
                  3.76517216e+01,   7.80572468e+01,   2.12273007e+02,   6.04260266e-01,
                  2.92665995e+01,   7.11052497e+01,   9.09046408e+01,   7.57757945e-01]
            param_init[-nref*4:] = p6
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p6, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p6, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev
            
        if (ncomp > 7):
            nref = 7
            p7 = [3.15513400e+01,   7.69719638e+01,   8.16744226e+01,   7.52091838e-01,   
                  2.80627783e+01,   4.09016828e+01,   2.30994195e+01,   1.93820380e+00,   
                  3.38867833e+01,   7.02930841e+01,   1.29382378e+02,   4.66601617e-01,   
                  4.13698426e+01,   8.11809017e+01,   1.05535809e+02,   5.47424460e-01,   
                  6.05328247e+01,   8.15442809e+01,   9.09113235e+01,   6.67060591e-01,   
                  4.25126697e+01,   5.52542448e+01,   4.19292419e+02,   1.22288097e-01,   
                  4.16456917e+01,   7.29177861e+01,   1.98882498e+02,   5.28357123e-01]
            p7 = [4.01051506e+01,   7.56331861e+01,   1.14372397e+02,   5.76347869e-02,   
                  4.68960736e+01,   5.15396574e+01,   3.73541282e+02,   1.18884151e-01,   
                  4.56320114e+01,   7.78249453e+01,   9.33827553e+01,   1.83122365e-01,   
                  4.95719716e+01,   7.12437508e+01,   2.44938525e+02,   4.77092348e-01,   
                  3.61021779e+01,   7.82500997e+01,   1.64515961e+02,   5.70382769e-01,   
                  2.79789871e+01,   7.10990145e+01,   7.74096730e+01,   6.03570149e-01,
                  4.00727132e+01,   8.01192844e+01,   9.29694717e+01,   8.28751782e-01]
            param_init[-nref*4:] = p7
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p7, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p7, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev

        if (ncomp > 8):
            nref = 8
            p8 = [3.39363142e+01,   7.85287244e+01,   1.38702742e+02,   6.42649638e-02,   
                  4.82058024e+01,   5.24170440e+01,   3.53594912e+02,   1.22808701e-01,   
                  4.29800083e+01,   7.47310762e+01,   7.86142269e+01,   1.79231034e-01,   
                  5.17214144e+01,   7.31418155e+01,   1.70415609e+02,   4.78292692e-01,   
                  3.68409150e+01,   7.55664077e+01,   1.61262877e+02,   5.09279512e-01,   
                  2.89720891e+01,   6.52821285e+01,   6.04515385e+01,   6.04069317e-01,   
                  3.89364739e+01,   8.08187625e+01,   9.34527253e+01,   6.73541100e-01,   
                  3.87785578e+01,   7.42778803e+01,   9.09723586e+01,   6.77996833e-01]
            param_init[-nref*4:] = p8
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p8, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p8, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev
            
        if (ncomp > 10):
            nref = 10
            p10 = [4.31559542e+01,   7.84830689e+01,   8.35825950e+01,   8.22992628e-02,   
                   3.55328014e+01,   5.29314492e+01,   2.23871970e+02,   1.19949831e-01,   
                   3.81764940e+01,   4.17475494e+01,   6.35564868e+01,   1.28309506e-01,   
                   4.66175343e+01,   8.40657780e+01,   1.70007448e+02,   1.46519833e-01,   
                   4.48557961e+01,   5.87070479e+01,   3.16043629e+01,   1.51580247e-01,   
                   5.38683882e+01,   7.39207501e+01,   1.28906602e+02,   4.20422912e-01,   
                   3.66775996e+01,   6.85328778e+01,   1.29260424e+02,   5.50944163e-01,   
                   3.91283240e+01,   7.27560073e+01,   5.36048026e+01,   5.56276127e-01,
                   3.69392592e+01,   7.95083069e+01,   1.64681872e+02,   6.18585410e-01,   
                   4.65443761e+01,   7.59174114e+01,   5.85047277e+01,   7.71851643e-01]  
            param_init[-nref*4:] = p10
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p10, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p10, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev

        if (ncomp > 12):
            nref = 12
            p12 = [5.18815078e+01,   8.12854891e+01,   1.15172061e+02,   7.99289784e-02,   
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
            param_init[-nref*4:] = p12
            lnl_init = likelihoodfunc(param_init[0:2].tolist()+p12, ra, dec, nstar, ncomp=nref)
            nbest = nstar_multigauss(param_init[0:2].tolist()+p12, ra, dec, ncomp=nref)
            stddev = np.std((nbest - nstar)/nstar)
            print 'Ncomp: ', nref, '  Initial Likelihood ', lnl_init,'  Stddev: ',stddev

    print 'Initial Guess of ', ncomp,' components with ', len(param_init), ' parameters:'
    lnl_init = likelihoodfunc(param_init, ra, dec, nstar, ncomp=ncomp)
    print 'Initial likelihood: ', lnl_init
    print param_init

    # Run MCMC

    sampler, d, bestfit, sigma, acor, burn_sampler = \
        run_emcee(ra, dec, nstar, param=param_init, ncomp=ncomp)
    
    # report results

    print 'Final Solution: '
    print bestfit
    lnl = likelihoodfunc(bestfit, ra, dec, nstar, ncomp=ncomp)
    nbest = nstar_multigauss(bestfit, ra, dec, ncomp=ncomp)
    print 'Final Likelihood: ', lnl,'  Stddev: ', np.std((nbest - nstar) / nstar)
    print 'Improvement: ', lnl - lnl_init

    plt.figure(2)
    plt.clf()
    im = plt.scatter(ra, dec, c=np.log10(nbest), s=s_size, 
                     linewidth=0, cmap='gist_ncar', alpha=0.5,
                     vmin = 0.7, vmax=np.log10(maxn))
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('Model Log$_{10} N_{stars}$  (pixel$^{-1}$)  ['+str(ncomp)+' Components]')

    plt.figure(3)
    plt.clf()
    im = plt.scatter(ra, dec, c=((nbest - nstar) / nstar), s=s_size, 
                     linewidth=0, cmap='seismic', alpha=0.5, 
                     vmin = -0.5, vmax = 0.5)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([12., 10.5, 41.1, 42.4])
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title('$\Delta N_{stars} / N_{stars}$')

    plt.figure(4)
    plt.clf()
    r = get_major_axis(ra, dec)
    im = plt.scatter(r, ((nbest - nstar) / nstar), c=(np.log10(nstar)), s=s_size, 
                     linewidth=0, cmap='gist_ncar', alpha=0.5, 
                     vmin = 0.7, vmax = 3)
    plt.plot(r, r*0)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0, 1.5, -0.75, 0.75])
    plt.xlabel('Major Axis Length (degrees)')
    plt.ylabel('$\Delta N_{stars} / N_{stars}$')
    plt.title('Log$_{10} N_{stars}$  Stddev: ' + str(np.std((nbest - nstar) / nstar)))

    plot_gaussians(bestfit, param_orig=param_init, fignum=5)

    # used for triangle plot of output
    varnames = [' ']
    for i in np.arange(0, ncomp):
        varnames.append('PA' + str(i))
        varnames.append('Incl' + str(i))
        varnames.append('N' + str(i))
        varnames.append('r' + str(i))
    varnames=varnames[1:]
    

    #sampler.chain (3 d array, number of walkers, number of steps, number of parameters)
    # sampler.flatchain (2d array of number of walker * number of steps, number of parameters)
    #plt.plot(sampler.chain[:,:,0].T, alpha=0.3)  # plot first parameter as a function of step num

    return sampler, d, bestfit, sigma, acor, burn_sampler, varnames

def run_emcee(ra, dec, nstars, param=[0.5,1.5,0.2], likelihoodfunction='',
              nwalkers=300, nsteps=50, nburn=500, nthreads=0, pool=None,
              ncomp=10):
    # NOTE: nthreads not actually enabled!  Keep nthreads=0!

    import emcee

    # setup emcee

    assert(ln_priors(param, ncomp=ncomp)), "First Guess outside the priors"

    ndim = len(param)

    ra0 = param[0]
    dec0 = param[1]
    #p0 = [param*(1. + np.random.normal(0, 0.05, ndim)) 
    #      for i in xrange(nwalkers)]
    p0 = [np.concatenate([np.array([ra0, dec0]), 
                          param[2:]*(1. + np.random.normal(0, 0.15, ndim-2))])
          for i in xrange(nwalkers)]
    print len(p0)
    print 'Perturbed Initial Guess: '
    print p0[0]
    print ln_priors(p0[0], ncomp=ncomp)
    print p0[1]
    print ln_priors(p0[1], ncomp=ncomp)
    
    if (likelihoodfunction == ''):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, 
                                        args=[ra, dec, nstars, ncomp])
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihoodfunction, 
                                        args=[ra, dec, nstars, ncomp])
    
    
    # burn in....
    print 'Burning in ', nburn, ' steps with ', nwalkers, ' walkers.'
    pos, prob, state = sampler.run_mcmc(p0, nburn)
    burn_sample = np.array(sampler.chain)

    # Correlation function values -- keep at least this many nsteps x 2
    try:
        acor = sampler.acor
    except:
        acor = -666.

    # run final...
    print 'Running final ', nsteps
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)

    names = ['ra', 'dec']
    for i in np.arange(0, ncomp):
        names.append('pa' + str(i))
        names.append('incl' + str(i))
        names.append('n' + str(i))
        names.append('sig' + str(i))

    d = { k: sampler.flatchain[:,e] for e, k in enumerate(names) }
    d['lnp'] = sampler.lnprobability.flatten()
    idx = d['lnp'].argmax()

    bestfit = np.array([d[names[k]][idx] for k in range(2 + 4*ncomp)])
    percval = [16, 50, 84]
    sigma = np.array([np.percentile(d[names[k]], percval) 
                      for k in range(2 + 4*ncomp)])

    return sampler, d, bestfit, sigma, acor, burn_sample


def plot_gaussians(param_vec, param_init=[], fignum=1):

    # assumes ra, dec plus series of PA, incl, normalization, sigma

    npar = 4
    ncomp = (len(param_vec) - 2) / npar
    i = np.arange(0,ncomp)
    i_pa   = 2 + i*npar
    i_incl = 3 + i*npar
    i_n0   = 4 + i*npar
    i_sig0 = 5 + i*npar

    pa = param_vec[i_pa]
    incl = param_vec[i_incl]
    n0 = param_vec[i_n0]
    sig0 = param_vec[i_sig0]
    maxn0 = np.max(n0)
    
    maxsize = 500.
    sizevec = maxsize * (n0 / maxn0)

    plt.figure(fignum)
    plt.clf()
    if (len(param_init) == len(param_vec)): 
        pai = param_init[i_pa]
        incli = param_init[i_incl]
        n0i = param_init[i_n0]
        sig0i = param_init[i_sig0]
        maxn0i = np.max(n0)
        sizeveci = maxsize * (n0i / maxn0i)
        print incli
        im = plt.scatter(sig0i, pai, c=incli, s=sizeveci, linewidth=0, 
                         marker='s', cmap='cool', alpha=0.5,
                         vmin = 40., vmax=90.)
    im = plt.scatter(sig0, pa, c=incl, s=sizevec, linewidth=1, 
                     cmap='cool', alpha=1.0,
                     vmin = 40., vmax=90.)
    color_bar = plt.colorbar(im)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.axis([0, 1.0, 30, 60])
    plt.xlabel('Radial Standard Deviation (degrees)')
    plt.ylabel('Position Angle (degrees)')
    plt.title('Color: Inclination   Size: Normalization')

def plot_all_gaussians():

    p1 = [3.93645194e+01,   7.30858338e+01,   7.11295877e+02,   4.90161893e-01]
    p2 = [4.14808984e+01,   6.73164528e+01,   6.50561463e+02,   2.21283498e-01,   
          4.19622955e+01,   7.07158849e+01,   2.85669750e+02,   6.85786748e-01]
    p3 = [3.41267976e+01,   7.75642217e+01,   1.82451671e+02,   6.75914073e-01,   
          4.84803167e+01,   6.28895179e+01,   5.48883102e+02,   1.44046825e-01,   
          4.47832770e+01,   7.18988721e+01,   3.21356219e+02,   5.40801785e-01]
    p4 = [3.82787028e+01,   7.64413805e+01,   1.39672344e+02,   6.50552884e-02,   
          3.59755493e+01,   7.74631162e+01,   2.02073577e+02,   6.50340923e-01,   
          4.72949787e+01,   5.82485668e+01,   4.61504742e+02,   1.33771259e-01,   
          4.46486702e+01,   7.17130298e+01,   3.27427644e+02,   5.14422367e-01]
    p5 = [5.45621018e+01,   7.33165211e+01,   1.32833299e+02,   7.31329102e-01,   
          2.41546127e+01,   7.60988458e+01,   1.35220363e+02,   6.88612435e-02,   
          3.50019957e+01,   7.74636875e+01,   1.71080350e+02,   7.14089280e-01,   
          5.09236719e+01,   5.83239977e+01,   4.18996689e+02,   1.38103422e-01,   
          3.81504493e+01,   7.28212443e+01,   2.35785022e+02,   4.31320994e-01]
    p6 = [3.80186206e+01,   7.91001790e+01,   1.13462417e+02,   6.54336183e-02,   
          4.84068268e+01,   5.27374660e+01,   4.18219432e+02,   1.16002168e-01,   
          4.55326344e+01,   8.30037721e+01,   1.22214507e+02,   1.81992183e-01,   
          4.72426967e+01,   7.26182257e+01,   2.80970312e+02,   4.61273919e-01,   
          3.76517216e+01,   7.80572468e+01,   2.12273007e+02,   6.04260266e-01,
          2.92665995e+01,   7.11052497e+01,   9.09046408e+01,   7.57757945e-01]
    p7 = [4.01051506e+01,   7.56331861e+01,   1.14372397e+02,   5.76347869e-02,   
          4.68960736e+01,   5.15396574e+01,   3.73541282e+02,   1.18884151e-01,   
          4.56320114e+01,   7.78249453e+01,   9.33827553e+01,   1.83122365e-01,   
          4.95719716e+01,   7.12437508e+01,   2.44938525e+02,   4.77092348e-01,   
          3.61021779e+01,   7.82500997e+01,   1.64515961e+02,   5.70382769e-01,   
          2.79789871e+01,   7.10990145e+01,   7.74096730e+01,   6.03570149e-01,
          4.00727132e+01,   8.01192844e+01,   9.29694717e+01,   8.28751782e-01]
    p8 = [3.39363142e+01,   7.85287244e+01,   1.38702742e+02,   6.42649638e-02,   
          4.82058024e+01,   5.24170440e+01,   3.53594912e+02,   1.22808701e-01,   
          4.29800083e+01,   7.47310762e+01,   7.86142269e+01,   1.79231034e-01,   
          5.17214144e+01,   7.31418155e+01,   1.70415609e+02,   4.78292692e-01,   
          3.68409150e+01,   7.55664077e+01,   1.61262877e+02,   5.09279512e-01,   
          2.89720891e+01,   6.52821285e+01,   6.04515385e+01,   6.04069317e-01,   
          3.89364739e+01,   8.08187625e+01,   9.34527253e+01,   6.73541100e-01,   
          3.87785578e+01,   7.42778803e+01,   9.09723586e+01,   6.77996833e-01]
    p10 = [4.31559542e+01,   7.84830689e+01,   8.35825950e+01,   8.22992628e-02,   
           3.55328014e+01,   5.29314492e+01,   2.23871970e+02,   1.19949831e-01,   
           3.81764940e+01,   4.17475494e+01,   6.35564868e+01,   1.28309506e-01,   
           4.66175343e+01,   8.40657780e+01,   1.70007448e+02,   1.46519833e-01,   
           4.48557961e+01,   5.87070479e+01,   3.16043629e+01,   1.51580247e-01,   
           5.38683882e+01,   7.39207501e+01,   1.28906602e+02,   4.20422912e-01,   
           3.66775996e+01,   6.85328778e+01,   1.29260424e+02,   5.50944163e-01,   
           3.91283240e+01,   7.27560073e+01,   5.36048026e+01,   5.56276127e-01,
           3.69392592e+01,   7.95083069e+01,   1.64681872e+02,   6.18585410e-01,   
           4.65443761e+01,   7.59174114e+01,   5.85047277e+01,   7.71851643e-01]  
    p12 = [5.18815078e+01,   8.12854891e+01,   1.15172061e+02,   7.99289784e-02,   
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
    
    plot_gaussians(np.array([1, 2] + p1), fignum=1)
    plot_gaussians(np.array([1, 2] + p2), fignum=2)
    plot_gaussians(np.array([1, 2] + p5), fignum=5)
    plot_gaussians(np.array([1, 2] + p10), fignum=10)
    plot_gaussians(np.array([1, 2] + p12), fignum=12)
    

