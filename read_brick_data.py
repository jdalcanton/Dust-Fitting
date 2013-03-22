import pyfits
from numpy import *
import matplotlib.pyplot as plt
import pylab
import PIL
import time
from scipy import ndimage

# read photometry file and return color and magnitude

def read_col_mag(filename = '../../Data/12056_M31-B15-F09-IR_F110W_F160W.st.fits'):
    """
    Reads fits file and returns color and magnitude for valid photometry 

    color, mag2 = read_col_mag(fitsfilename)
    """
    fitstab = pyfits.open(filename)
    mag1 = fitstab[1].data.field('MAG1_IR')
    mag2 = fitstab[1].data.field('MAG2_IR')
    fitstab.close()

    i = where((mag1 < 30) & (mag2 < 30) & (mag1 > 10) & (mag2 > 10))

    return mag1[i] - mag2[i], mag2[i]

def read_mag_position(filename = '../../Data/12056_M31-B15-F09-IR_F110W_F160W.st.fits'):
    """
    Reads fits file and returns magnitudes & position for valid photometry 

    mag1, mag2, ra, dec = read_mag_position(fitsfilename)
    """

    fitstab = pyfits.open(filename)
    # mag1 = fitstab[1].data.field('MAG1_IR')
    # mag2 = fitstab[1].data.field('MAG2_IR')
    # mag1 = fitstab[1].data.field('MAG1')
    # mag2 = fitstab[1].data.field('MAG2')
    mag1 = fitstab[1].data.field('ir_MAG1')
    mag2 = fitstab[1].data.field('ir_MAG2')
    ra   = fitstab[1].data.field('RA')
    dec   = fitstab[1].data.field('DEC')
    fitstab.close()

    # cul 
    i = where((mag1 < 30) & (mag2 < 30) & (mag1 > 10) & (mag2 > 10))

    return mag1[i], mag2[i], ra[i], dec[i]

def read_mag_position_gst(filename = '../../Data/sixfilt-b18-v8-st-large.fits'):
    """
    Reads fits file and returns magnitudes & position for valid photometry 
    Applies cuts to mimic GST

    mag1, mag2, ra, dec = read_mag_position_gst(fitsfilename)
    """

    fitstab = pyfits.open(filename)
    # mag1 = fitstab[1].data.field('MAG1_IR')
    # mag2 = fitstab[1].data.field('MAG2_IR')
    # mag1 = fitstab[1].data.field('MAG1')
    # mag2 = fitstab[1].data.field('MAG2')
    mag1 = fitstab[1].data.field('ir_MAG1')
    mag2 = fitstab[1].data.field('ir_MAG2')
    ra   = fitstab[1].data.field('RA')
    dec   = fitstab[1].data.field('DEC')
    round1 = fitstab[1].data.field('ir_round1')
    round2 = fitstab[1].data.field('ir_round2')
    sharp1 = fitstab[1].data.field('ir_sharp1')
    sharp2 = fitstab[1].data.field('ir_sharp2')
    snr1 = fitstab[1].data.field('ir_snr1')
    snr2 = fitstab[1].data.field('ir_snr2')
    crowd1 = fitstab[1].data.field('ir_crowd1')
    crowd2 = fitstab[1].data.field('ir_crowd2')
    inbrick = fitstab[1].data.field('inbrick')
    fitstab.close()

    # cul 
    redsnrthresh = 5.0
    bluesnrthresh = 5.0
    sharpthresh1 = 0.3
    sharpthresh2 = -0.05 # eliminates a lot of stars
    # sharpthresh3 = -0.1 + round1 * 0.02 Eliminated (sharp1 > sharpthresh3)
    # sharpthresh4 = -0.1 + round2 * 0.02 Eliminated (sharp2 > sharpthresh4)
    # roundthresh1 = -0.25  Eliminates almost no stars (round1 > roundthresh1)
    #  roundthresh2 = 7  ELIMINATED! BIG COMPLETENESS HIT (round < roundthresh2)
    i = where((mag1 < 30) & (mag2 < 30) & (mag1 > 10) & (mag2 > 10) & 
              (snr1 > bluesnrthresh) & (snr2 > redsnrthresh) & 
              (abs(sharp1) < sharpthresh1) & (abs(sharp2) < sharpthresh1) & 
              (sharp1 > sharpthresh2) & (sharp2 > sharpthresh2))
    print 'Cutting from ', len(mag1), ' to ', len(i)

    return mag1[i], mag2[i], ra[i], dec[i]


def read_mag_position_gst_allparam(filename = '../../Data/sixfilt-b18-v8-st-large.fits'):
    """
    Reads fits file and returns magnitudes & position for valid photometry 
    Applies cuts to mimic GST

    mag1, mag2, ra, dec, crowd1, crowd2, round1, round2, sharp1, sharp2  = read_mag_position_gst_allparam(fitsfilename)
    """

    fitstab = pyfits.open(filename)
    # mag1 = fitstab[1].data.field('MAG1_IR')
    # mag2 = fitstab[1].data.field('MAG2_IR')
    # mag1 = fitstab[1].data.field('MAG1')
    # mag2 = fitstab[1].data.field('MAG2')
    mag1 = fitstab[1].data.field('ir_MAG1')
    mag2 = fitstab[1].data.field('ir_MAG2')
    ra   = fitstab[1].data.field('RA')
    dec   = fitstab[1].data.field('DEC')
    round1 = fitstab[1].data.field('ir_round1')
    round2 = fitstab[1].data.field('ir_round2')
    sharp1 = fitstab[1].data.field('ir_sharp1')
    sharp2 = fitstab[1].data.field('ir_sharp2')
    snr1 = fitstab[1].data.field('ir_snr1')
    snr2 = fitstab[1].data.field('ir_snr2')
    crowd1 = fitstab[1].data.field('ir_crowd1')
    crowd2 = fitstab[1].data.field('ir_crowd2')
    inbrick = fitstab[1].data.field('inbrick')
    fitstab.close()

    # cul 
    snrthresh = 4.0
    sharpthresh = 0.3
    sharpthresh2 = -0.075
    roundthresh1 = -0.25
    roundthresh2 = 6
    i = where((mag1 < 30) & (mag2 < 30) & 
              (mag1 > 10) & (mag2 > 10) & 
              (snr1 > snrthresh) & (snr2 > snrthresh) & 
              (abs(sharp1)<sharpthresh) & (abs(sharp2)<sharpthresh))
    print 'Cutting from ', len(mag1), ' to ', len(i)

    return mag1[i], mag2[i], ra[i], dec[i], crowd1[i], crowd2[i], round1[i], round2[i], sharp1[i], sharp2[i]

