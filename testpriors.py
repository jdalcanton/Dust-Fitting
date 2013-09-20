import numpy as np
from scipy.stats import norm
import scipy.special as special
import matplotlib.pyplot as plt

def testAVdependentprior(fprior=0.2):

    fprior_use = fprior

    nx = 100

    fvec = np.linspace(0.025, 0.975, nx)
    alpha = np.log(fprior_use) / np.log(0.5)
    xvec = np.log(fvec**(1.0/alpha) / (1.0 - fvec**(1.0/alpha)))
    xpriorval = np.log(fprior**(1.0/alpha) / (1.0 - fprior**(1.0/alpha)))

    AVvec = np.array([0.025, 0.1, 0.25, 0.5, 1, 1.5, 2., 4., 6., 10.])
    AVmeshvec = np.linspace(0.001, 2, nx)
    sigmeshvec = np.linspace(0.001, 1., nx)

    #fredvec = fprior_use + 0.0*AVvec
    #fredmeshvec = fprior_use + 0.0*AVmeshvec

    AVgrid, xgrid = np.meshgrid(AVmeshvec, xvec)
    lnpgrid = np.zeros(AVgrid.shape)
    lnpsiggrid = np.zeros(AVgrid.shape)

    # set up ln_priors function

    ln_priors = lnpriorobj(fprior_use)
    param = ln_priors([0.,0.5,0.4], return_prior_parameters=True)
    print param

    for j in range(len(AVmeshvec)):
        for i in range(len(xvec)):

            lnpgrid[i, j] = ln_priors([xvec[i],AVmeshvec[j],0.4])
            lnpsiggrid[i, j] = ln_priors([xpriorval,1.5,sigmeshvec[i]])

    rangevec = [np.min(AVmeshvec), np.max(AVmeshvec), np.min(fvec), np.max(fvec)]
    rangesigvec = [np.min(AVmeshvec), np.max(AVmeshvec), np.min(sigmeshvec), np.max(sigmeshvec)]
    
    lnp = np.zeros((nx, len(AVvec)))
    for j in range(len(AVvec)):
        for i in range(nx):

            lnp[i, j] = ln_priors([xvec[i],AVvec[j],0.4])

    plt.figure(1)
    #plt.clf()
    legvec = []
    for j in range(len(AVvec)):
        p = plt.plot(xvec, lnp[:, j], linewidth=2)
        legvec = np.append(legvec, p)
    plt.legend(legvec,AVvec, loc='lower left')
    plt.xlabel('x')
    plt.ylim([-0.75, 1])

    plt.figure(2)
    #plt.clf()
    legvec = []
    for j in range(len(AVvec)):
        p = plt.plot(fvec, lnp[:, j], linewidth=3)
        legvec = np.append(legvec, p)
    plt.legend(legvec,AVvec, loc='lower right')
    plt.xlabel('f')
    plt.axis([0, 1, 0, 1])

    plt.figure(3)
    lnpmax = np.max(lnpgrid)
    lnprange = 1.0
    im = plt.imshow(lnpgrid, vmin = lnpmax - lnprange, vmax=lnpmax, aspect='auto', origin='lower',
                    extent=rangevec)
    plt.xlabel('$A_V$')
    plt.ylabel('$f_{red}$')
    plt.colorbar(im)

    plt.figure(4)
    lnpmax = np.max(lnpsiggrid)
    lnprange = 1.0
    im = plt.imshow(lnpsiggrid, vmin = lnpmax - lnprange, vmax=lnpmax, aspect='auto', origin='lower',
                    extent=rangesigvec)
    plt.xlabel('$A_V$')
    plt.ylabel('$\sigma$')
    plt.colorbar(im)

def testprior(fprior, sigma, xstart):

    nx = 100
    xvec = np.linspace(-1.0+0.0001, 5, 100)
    fvec = (np.exp(xvec) / (1.+np.exp(xvec)))**(np.log(fprior)/np.log(0.5))

    lnp = np.zeros(nx)
    for i in range(nx):
        lnp[i] = ln_priors([xvec[i],1.,0.4], sigma, xstart)

    plt.figure(1)
    #plt.clf()
    plt.plot(xvec, lnp - np.max(lnp))
    plt.axis([-1.5, 4, -0.75, 0.1])
    plt.xlabel('X')
    plt.ylabel('ln Prob')

    plt.figure(2)
    #plt.clf()
    plt.plot(fvec, lnp - np.max(lnp))
    plt.axis([0, 1, -0.75, 0.1])
    plt.xlabel('X')
    plt.ylabel('ln Prob')

def skew_normal(x, mean=0.0, stddev=1.0, alpha=0.0, align_mode=True, printstuff=False):
    """
    Return a skew_normal distribution with the desired mean and stddev. alpha controls the 
    degree of skewness.  Based on: http://en.wikipedia.org/wiki/Skew_normal_distribution
    """
    
    xsig = stddev / np.sqrt(1. - (2./np.pi) * (alpha**2 / (1. + alpha**2)))
    xmean = mean - xsig * (alpha / np.sqrt(1. + alpha**2)) * np.sqrt(2./np.pi)

    if align_mode:   # treat mean as desired mode
        mode_shift_poly = [3.04035639e-29,   5.58273969e-21,  -1.67582716e-26,  -3.62284547e-18,
                           4.30636532e-24,   1.04139549e-15,  -6.77910067e-22,  -1.74658626e-13,
                           7.08698868e-20,   1.89492508e-11,  -4.98639476e-18,  -1.39368656e-09,
                           2.35161849e-16,   7.08204586e-08,  -7.76046641e-15,  -2.48823548e-06,
                           2.20786871e-13,   5.95653863e-05,  -6.65071653e-12,  -9.41834368e-04,
                           1.60896830e-10,   9.31976144e-03,  -2.11035367e-09,  -5.12998001e-02,
                           1.06153311e-08,  -2.71757803e-02,  -8.61473382e-09]  # from find_peak_skew_normal
        mode_shift = np.poly1d(mode_shift_poly)
        xmean = xmean - stddev * mode_shift(alpha)

    if printstuff:
        print 'Mean: ', mean, ' Stddev:  ',stddev, ' Alpha: ',alpha
        print 'XMean:', xmean, ' XStddev: ',xsig

    y = alpha*((x - xmean)/xsig)
    cdf = norm.cdf(y)
    #cdf = 0.0 * y
    #cdf = np.where((-3 <= y) & (y < -1), (1.0/16.0)*(9.0*y + 3.0*y**2 + y**3/3.0 + 9.0), cdf)
    #cdf = np.where((-1 <= y) & (y < 1), (1.0/8.0)*(3.0*y - y**3/3. + 4.0), cdf)
    #cdf = np.where((1 <= y) & (y < 3), (1.0/16.0)*(9.0*y - 3.*y**2 + y**3/3.0 + 7.0), cdf)
    #cdf = np.where(3 <= y, 1.0, cdf)
    
    return 2.0 * norm.pdf(x,loc=xmean,scale=xsig) * cdf

def find_peak_skew_normal(mean=0., stddev=1.):
    
    avec = linspace(-10.,10.,1000)
    x = linspace(-5.,5.,1000)
    maxvec = array([nanmax(skew_normal(x,mean=mean,stddev=stddev,alpha=a,align_mode=False)) for a in avec])
    pkvec = array([x[where(skew_normal(x,mean=mean,stddev=stddev,alpha=a,align_mode=False) == 
                     maxvec[i])][0] for i, a in enumerate(avec)])

    plt.plot(avec,pkvec/stddev)
    plt.xlabel('alpha')
    plt.ylabel('Mode / Stddev')

    npoly=26
    param2 = polyfit(avec, pkvec, npoly)
    print 'Polynomial fit to residuals: ',param2
    p2 = poly1d(param2)
    plt.plot(avec, p2(avec))
    plt.plot(avec, pkvec - p2(avec))


def ln_priors_lognormal_f(p, sigma, xstart, return_prior_parameters=False):
    
    # set up ranges

    p0 = [xstart, 5.0]          # x = ln(f/(1-f)) where fracred -- red fraction
    p1 = [0.0001, 10.0]         # median A_V
    p2 = [0.01, 1.5]            # sigma
                              
    # set up gaussian for x
    p0mean = 0.0              
    p0stddev = 1.5

    # set up lognormal for x
    p0mode = -p0[0]             
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
            #print p0
            #print p[0]
            return -np.inf

        # if all parameters are in range, return the ln of the Gaussian 
        # (for a Gaussian prior on x) and the ln of the log normal prior
        # on sigma (p[2])
        lnp = 0.0000001
        # gaussian
        #lnp += -0.5 * (p[0] - p0mean) ** 2 / p0stddev**2
        # log normal
        lnp += - np.log(p[0]-p0offset) - 0.5 * (np.log(p[0]-p0offset) - p0mu) ** 2 / p0sigma**2
        lnp += -(- np.log(0-p0offset) - 0.5 * (np.log(0-p0offset) - p0mu) ** 2 / p0sigma**2)
        #
        lnp += - np.log(p[2]) - 0.5 * (np.log(p[2]) - p2mu) ** 2 / p2sigma**2

        return lnp

def ln_priors_skew_normal(p, sigma, xstart, return_prior_parameters=False, f_mean=0.5):
    
    # set up ranges

    p0 = [xstart, 5.0]          # x = ln(f/(1-f)) where fracred -- red fraction
    p1 = [0.0001, 10.0]         # median A_V
    p2 = [0.01, 1.5]            # sigma
                              
    # set up skew-normal for x
    p0mode = 0.0             
    bsig0 = -0.5
    asig0 = 1.0
    sig0max = 100.
    p0sigma = min(10.0**(bsig0 + asig0*p[1]), sig0max)
    p0alpha = (0.5 / f_mean) - 1.0

    # Note: This approximation doesn't appear to be good enough. BUT, looks ok when using "skew_normal". Bug?
    p0mode_shift_poly = [3.04035639e-29,   5.58273969e-21,  -1.67582716e-26,  -3.62284547e-18,
                         4.30636532e-24,   1.04139549e-15,  -6.77910067e-22,  -1.74658626e-13,
                         7.08698868e-20,   1.89492508e-11,  -4.98639476e-18,  -1.39368656e-09,
                         2.35161849e-16,   7.08204586e-08,  -7.76046641e-15,  -2.48823548e-06,
                         2.20786871e-13,   5.95653863e-05,  -6.65071653e-12,  -9.41834368e-04,
                         1.60896830e-10,   9.31976144e-03,  -2.11035367e-09,  -5.12998001e-02,
                         1.06153311e-08,  -2.71757803e-02,  -8.61473382e-09]  # from find_peak_skew_normal
    p0mode_shift = np.poly1d(p0mode_shift_poly)

    p0width = p0sigma / np.sqrt(1.0 - (2.0/np.pi) * (p0alpha**2 / (1.0 + p0alpha**2)))
    p0mean = p0mode - p0sigma * p0mode_shift(p0alpha)
    #print p0mode_shift(p0alpha), p0mode - p0mean
    p0location = p0mean - p0width * (p0alpha / np.sqrt(1.0 + p0alpha**2)) * np.sqrt(2.0/np.pi)

    y = (p[0] - p0location) / p0width
    y0 = (p0mode - p0location) / p0width

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
            #print p0
            #print p[0]
            return -np.inf

        # if all parameters are in range, return the ln of the Gaussian 
        # (for a Gaussian prior on x) and the ln of the log normal prior
        # on sigma (p[2])
        lnp = 0.0000001
        # gaussian
        #lnp += -0.5 * (p[0] - p0mean) ** 2 / p0stddev**2
        # log normal
        lnp += - 0.5 * y**2
        lnp += + 0.5 * y0**2
        lnp += + np.log(1.0 + special.erf(y*p0alpha))
        lnp += - np.log(1.0 + special.erf(y0*p0alpha))
        #
        lnp += - np.log(p[2]) - 0.5 * (np.log(p[2]) - p2mu) ** 2 / p2sigma**2

        return lnp

def ln_priors_function(p, return_prior_parameters=False, f_mean=0.2):

    # set up easy ranges (AV, sig)

    p1 = [0.0001, 10.0]         # median A_V
    p2 = [0.01, 1.5]            # sigma
                              
    if ((p1[0] > p[1]) | (p[1] > p1[1]) | 
        (p2[0] > p[2]) | (p[2] > p2[1])):

        return -np.inf

    # set up ranges for x (i.e., regularlized f_red)

    alpha = np.log(0.5) / np.log(f_mean)
    frange = np.array([0.02, 0.98])
    p0 = np.log(frange**alpha / (1.0 - frange**alpha))
                              
    if ((p0[0] > p[0]) | (p[0] > p0[1])):

        return -np.inf

    # correct geometrical f_mean for A_V-dependent filling factors
    gamma = 2.0
    AV0 = 0.2

    f_fill_min = 0.1
    f_fill = f_fill_min + (1.0 - f_fill_min) * ((p[1]/AV0)**gamma / 
                                                (1.0 + (p[1]/AV0)**gamma))
    #f_mean_corr = np.maximum(f_mean * f_fill, f_mean_min)
    f_mean_corr = f_mean * f_fill
    
    # shift mean of x from 0 at f=f_mean, to x_corr equivalent to f_mean_corr
    x_corr = np.log(f_mean_corr**alpha / (1.0 - f_mean_corr**alpha))

    # set range over which we want typical f to vary
    frangescale = 0.25
    df = 0.15
    fsigrange = [f_mean*frangescale, f_mean + min(df, 0.99 - f_mean)]
    x_min = np.log(fsigrange[0]**alpha 
                   / (1.0 - fsigrange[0]**alpha))
    x_max = np.log(fsigrange[1]**alpha 
                   / (1.0 - fsigrange[1]**alpha))
    
    # set up split-normal for x

    p0scale1 = 0.1
    p0scale2 = 0.5
    # set split gaussian widths (for < x_corr and >x_corr)
    p0sig1 = p0scale1 * min(abs(x_corr - x_min), x_min*1.05)
    p0sig2 = p0scale2 * (x_max - x_corr)

    p0sigvec = [p0sig1, p0sig2]
    # automatically select proper sigma, based on p[0]<0 or p[0]>0
    p0sig = p0sigvec[((np.sign(p[0] - x_corr) + 1) / 2).astype(int)]


    # set up log normal for sigma, keeping same mode
    p2mode = 0.30              # w = broad gaussian 
    p2sigma = 0.5
    p2mu = np.log(p2mode) + p2sigma**2   # mu = ln(median)

    if return_prior_parameters:

        return {'p0': p0, 'p1': p1, 'p2': p2, 'frange': frange,
                'gamma': gamma, 'AV0': AV0, 'AVparam': p[1],
                'f_mean': f_mean, 'f_fill': f_fill, 'f_mean_corr': f_mean_corr, 'x_corr': x_corr, 
                'f_fill_min': f_fill_min, 'frangescale': frangescale, 'df': df, 
                'fsigrange': fsigrange, 'x_min': x_min, 'x_max': x_max,
                'alpha': alpha,
                'p0scale1': p0scale1, 'p0scale2': p0scale2, 
                'p0sigvec': p0sigvec, 'p0sig': p0sig,
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


class lnpriorobj(object):

    def __init__(self, f_mean):
        self.f_mean = f_mean

    def __call__(self, param, **kwargs):

        return ln_priors_function(param, f_mean = self.f_mean, **kwargs)

    def map_call(self, args):

        return self(*args)


def erf_approx(x):  # not actually as fast as special.erf
    
    p  =  0.47047
    a1 =  0.3480242
    a2 = -0.0958798
    a3 =  0.7478556

    t = 1.0 / (1.0 + p * np.abs(x))
    e = 1.0 - (a1*t + a2*t**2 + a3*t**3)*np.exp(-x**2)

    return (x / np.abs(x))*e
