import makefakecmd

def initialize_cmd():
    
    fn = '../Data/ir-sf-b21-v8-st.fits'

    m110range = [16.0,26.0]
    m160range = [18.4,26.0]
    mrange = m160range
    crange = [0.3,3.0]

    # range for defining "narrow"
    mfitrange = [18.7,22.5]
    cfitrange = [0.3,2.0]
    clo, mlo, ilo, cnar, mnar, inr,cm,cstd = isolate_low_AV_color_mag(
        filename=fn, frac=0.05, mrange=mfitrange,d_arcsec=10.)

    fracred = 0.5
    medianAV = 1.5
    muAV = 0.2
    Amag_AV = 0.20443
    Acol_AV = 0.33669 - 0.20443
    AVparam = [fracred, medianAV, muAV, Amag_AV, Acol_AV]

    t = arctan(-Amag_AV / Acol_AV)
    c0 = 1.0
    qnar = mnar + (cnar-c0)*sin(t)/cos(t)

    plt.figure(1)
    plt.clf()
    cmdnarorig, extent, cedges, medges = display_CM_diagram(cnar,mnar,crange=crange,mrange=mrange,nbins=[50,50])

    plt.figure(2)
    plt.clf()
    cmdnar, extent, cedges, medges = display_CM_diagram(cnar,qnar,crange=crange,mrange=mrange,nbins=[50,50])

    plt.figure(3)
    plt.clf()
    masknar,meancol,sigcol = clean_fg_cmd(cmdnar,2.5,niter=4,showplot=0)
    plt.imshow(cmdnar*masknar,extent=[cedges[0],cedges[-1],medges[-1],medges[0]], origin='upper',aspect='auto', interpolation='nearest')
    plt.xlabel('F110W - F160W')
    plt.ylabel('Extinction Corrected F160W')
    #plt.plot(cedges,medges[rint(meancol).astype(int)],color='yellow')
    #plt.plot(cedges,medges[rint(meancol-3.0*sigcol).astype(int)],color='yellow')
    #plt.plot(cedges,medges[rint(meancol+3.0*sigcol).astype(int)],color='yellow')

    m1, m2, ra, dec = read_mag_position_gst(fn)
    c = array(m1 - m2)
    m = array(m2)
    q = m + (c-c0)*sin(t)/cos(t)

    plt.figure(4)
    plt.clf()
    clim = cedges[rint(meancol - 2.0*sigcol).astype(int)]
    maskdata1 = make_data_mask(cmdnarorig,cedges,medges,m110range,m160range,clim)

    plt.imshow(maskdata1,extent=[cedges[0],cedges[-1],medges[-1],medges[0]], origin='upper',aspect='auto', interpolation='nearest')
    plt.xlabel('F110W - F160W')
    plt.ylabel('Extinction Corrected F160W')
    plt.gca().autoscale(False)
    plt.plot(c,m,',',color='yellow',alpha=0.2)
    plt.imshow(maskdata1,extent=[cedges[0],cedges[-1],medges[-1],medges[0]], origin='upper',aspect='auto', interpolation='nearest',alpha=0.1)

    plt.figure(5)
    plt.clf()
    maskdata2 = make_data_mask(cmdnar,    cedges,medges,m110range,m160range,clim,useq=[c0,t])
    plt.imshow(maskdata2,extent=[cedges[0],cedges[-1],medges[-1],medges[0]], origin='upper',aspect='auto', interpolation='nearest')
    plt.xlabel('F110W - F160W')
    plt.ylabel('Extinction Corrected F160W')
    plt.gca().autoscale(False)
    plt.plot(c,q,',',color='yellow',alpha=0.05)

    plt.figure(6)
    plt.clf()

    t0=time.time()
    fakecmd = makefakecmd(cmdnar*masknar, cedges, medges, AVparam, c0, SHOWPLOT=True)
    print time.time() - t0

