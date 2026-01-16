import numpy as np  # math pack!
from librosa import lpc
from scipy import interpolate
from scipy.signal import convolve, savgol_filter

eps = 10 ** -16


def standardization(x):
    # x: 1-d signal
    return (x - np.mean(x)) / (eps + np.std(x))


def detrend(x, n=1):
    """
    detrend by polynomail of degree n

    x: 1-d signal
    Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to
    points (x, y). Returns a vector of coefficients p that minimises the
    squared error in the order deg, deg-1, … 0
    """
    idx = np.arange(0, len(x))
    model = np.polyfit(idx, x, n)
    trend = np.polyval(model, idx)
    return x - trend


def pre_whiten(x, method="AR"):
    """
    pre-whitening the signal
    :param x: 1-d signal for pre-whitening
    :param method: select from "cepstrum" (cepstrum pre-whitening),
                    and "AR" (AutoRegressive pre-whitening)
    :return: pre-whitened signal

    Ref.:
        [1] Borghesani P. et al, Application of cepstrum pre-whitening for the diagnosis of bearing
    	    faults under variable speed conditions, MSSP, 2013.
    """
    eps = 10 ** -16

    if method in ["AR"]:
        """
        AutoRegressive (AR) Modeling:
        -Best whitening quality in most cases.
        -Accurately models underlying signal structure and removes predictable components.
         Especially effective for stationary signals and when the correct model order is chosen.

        Remark:
            The order p determines how many past values are used to predict the current value. 
            A model order that’s too low may underfit (not remove enough structure), 
            and an order that’s too high may overfit (introduce noise). 
        """
        # Sets the number of lag terms (model order) for the AR process.
        lags = 100
        # -- Linear Prediction Coefficients via Burg’s method
        coef = lpc(x, order=lags)
        x = convolve(coef, x)  # equivalence to fftfilt in matlab
        # Removes edge effects introduced by the convolution.
        return x[lags:-100]

    elif method in ["cepstrum"]:
        """
        Simple and fast, yet not that good, empirically.
        """
        X = np.fft.fft(x)
        X_white = X / (np.abs(X) + eps)
        return np.real(np.fft.ifft(X_white))
        # return  np.real(np.fft.ifft(  np.fft.fft(x) / (eps+np.abs(np.fft.fft(x)))  ))
    else:
        raise NotImplementedError


def tacho_to_rpm(tacho, fs, PPR=1, threshold=None, slope=2):
    """
    Convert tachometer impulse series to revolution-per-minute series

    Inputs:
        tacho: tachometer impulse series
        fs: sampling frequency
        PPR: pulses per revolution, resolution of the tachometer
        slope: 2 or -2, 2 means using positive slope and -2 means using negative slope to indicate the change of impulses
    Outputs:
       rpm: revolution-per-minute time series (unit: rpm), meaning the amount of revolutions in a minute

    Ref.: https://github.com/efsierraa/PyCycloVarREB/blob/master/functions/REB_functions.py
    """

    # time resolution is the inverse of frequency resolution
    dt = 1 / fs
    # get time axis stamps (unit: second)
    t = np.arange(0, len(tacho)) * dt
    # Produce +1 where signal is above trigger level
    # and -1 where signal is below trigger level
    trigger_level = threshold if threshold is not None else np.mean(tacho)
    xs = np.sign(tacho - trigger_level)
    # Differentiate this to find where xs changes
    # between -1 and +1 and vice versa
    xDiff = np.diff(xs)  # a sequence of 2,0,or -2; 2 means a positive change, -2 means a negative change

    # We need to synchronize xDiff with variable t from the
    # code above, since DIFF shifts one step
    tDiff = t[1:]

    # Now find the time instances of positive slope positions
    # (-2 if negative slope is used)
    tTacho = tDiff[xDiff == slope]  # xDiff.T return the indexes boolean
    # Count the time between the tacho signals and compute
    # the RPM at these instances
    # rev_per_impulse = 1/PPR # how many revolutions in one impulse
    # impulse_per_sec = 1/np.diff(tTacho) # how many impulses in one second
    # sec_per_min = 60 # how many seconds in one minute
    # rpmt_ = sec_per_min * rev_per_impulse * impulse_per_sec # how many revolutions in one minute

    rpm = 60 / PPR / np.diff(tTacho)  # Temporary rpm values

    if len(rpm) < 3:
        raise ValueError('The length of tachometer impulses is too short for estimation and interpolation')
    else:
        # Use three tacho pulses at the time and assign mean
        # value to the center tacho pulse
        rpm = 0.5 * (rpm[0:-1] + rpm[1:])
        tTacho = tTacho[1:-1]  # diff again shifts one sample
        # Smoothing
        wfiltsv = int(2 ** np.fix(np.log2(.05 * fs))) - 1
        if len(rpm) > wfiltsv:
            rpm = savgol_filter(rpm, wfiltsv, 2)  # smoothing filter
        # instantiate an interpolator
        # Fits a spline y = spl(x) of degree k to the provided x, y data. k=1, piece wise linear
        # Spline function passes through all provided points. Equivalent to UnivariateSpline with s = 0.
        rpmt_interpolator = interpolate.InterpolatedUnivariateSpline(x=tTacho, y=rpm, w=None, bbox=[None, None], k=1)
        # Evaluate the fitted spline at the given points
        rpm = rpmt_interpolator(t)

    return rpm, t


def angular_resampling(t, rpm, sig_t, keepLen=False, reLen=None):
    """
    Computed order tracking / Angular resampling

    Inputs:
        t: time stamps (unit: second)
        rpm: revolution-per-minute time series (unit: rpm), meaning the amount of revolutions in a minute
        sig_t: signal in time domain
        keepLen: sig_cyc have the same length of sig_t

    outputs:
        sig_cyc: signal in cycle domain, which is angular-resampled from sig_t
        fs_cyc: sampling frequency in cycle domain. If none, use the resolution fs_cyc = int(1 / (dt * min(rpm_in_hz)))
                fs_cyc>=2*order_highest:
                      fs_cyc should be at least twice larger than the highest order in order spectrum (Nyquist sampling frequency).
                      fs_cyc>=reSample_num*order_resolution:
                      fs_cyc should ensure enough order_resolution to differentiate orders in order spectrum.
                      reSample_num = int(fs_cyc*max(cumulative_phase))
                      order_resolution = int(1/max(cumulative_phase))

    Ref.:
    [1] https://doi.org/10.3390/s24020454
    [2] W. Cheng, R. X. Gao, J. Wang, T. Wang, W. Wen, and J. Li,
        “Envelope deformation in computed order tracking and error in order
        analysis,” Mechanical Systems and Signal Processing, vol. 48, no. 1–2,
        pp. 92–102, Oct. 2014, doi: 10.1016/j.ymssp.2014.03.004.
    """
    # The inputs should correspond to the same sampling frequency in time domain.
    # Thus, they should have the same lengths. Otherwise, raise the error and the message.
    assert len(t) == len(rpm) == len(
        sig_t), "The inputs should correspond to the same sampling frequency in time domain!"

    rpm_in_hz = rpm / 60  # convert rpm unit to revolution-per-second or equivalently Hz

    # Time resolution
    dt = t[1] - t[0]  # 1/fs

    # Calculate cumulative phase of the shaft， integral rpm over time
    cumulative_phase = np.cumsum(rpm_in_hz * dt)
    cumulative_phase -= cumulative_phase[0]  # zero phase at the starting point

    # Determine angular sampling frequency and points
    if reLen is None and not keepLen:
        if min(rpm_in_hz) == 0 or min(rpm_in_hz) < 0:
            raise ValueError('The rpm of the reference should not be 0 or negative!')
        fs_cyc = int(1 / (dt * min(rpm_in_hz)))
        reSampleNum = int(fs_cyc * max(cumulative_phase))
    else:
        reSampleNum = len(sig_t) if keepLen else reLen
        fs_cyc = int(reSampleNum / max(cumulative_phase))  # points per phase

    # Generate constant phase intervals
    constant_phase_intervals = np.linspace(start=0, stop=max(cumulative_phase), num=reSampleNum)

    # -- Angular resampling process: fit samples using interpolation, and re-select samples from fitted functions
    # Interpolate to find new time points with constant phase intervals
    interp_func = interpolate.interp1d(cumulative_phase, t, kind='linear')  # fit cumulative_phase and t
    # interp_func = interpolate.UnivariateSpline(cumulative_phase, t, k=3, s=0)  # fit cumulative_phase and t
    times_of_constant_phase_intervals = interp_func(
        constant_phase_intervals)  # re-select t based on constant_phase_intervals
    # Use UnivariateSpline for spline interpolation
    # Fits a spline y = spl(x) of degree k to the provided x, y data. k=3 by default, a cubic spline.
    # If s=0, spline will interpolate through all data points. This is equivalent to InterpolatedUnivariateSpline.
    spline_interpolator = interpolate.UnivariateSpline(x=t, y=sig_t, k=3, s=0)  # fit t and sig_t
    # Evaluate the fitted spline at the given points
    sig_cyc = spline_interpolator(
        times_of_constant_phase_intervals)  # re-select sig_t based on times_of_constant_phase_intervals

    return sig_cyc, fs_cyc


def sig_segmentation(data, label, seg_len, start=0, stop=None):
    '''
    This function is mainly used to segment the raw 1-d signal into samples and labels
    using the sliding window to split the data
    '''
    data_seg = []
    lab_seg = []
    start_temp, stop_temp, stop = start, seg_len, stop if stop is not None else len(data)
    while stop_temp <= stop:
        sig = data[start_temp:stop_temp]
        sig = sig.reshape(-1, 1)
        data_seg.append(sig)  # z-score normalization
        lab_seg.append(label)
        start_temp += seg_len
        stop_temp += seg_len
    return data_seg, lab_seg




