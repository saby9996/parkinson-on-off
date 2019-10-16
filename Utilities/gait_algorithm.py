from scipy import signal
from scipy import integrate
from pywt import cwt
import numpy as np
import pandas as pd

#Signal Processing
from scipy.signal import butter, lfilter
import math
#Self Declared Modules and Packages
from Utilities import helper_functions

def gait_params(x_axis, y_axis, z_axis, fs=32, Wearable_Height=12, g=8):
    fc = fs / 2 - 1
    Wearable_Height = Wearable_Height / 100
    # Variable Change
    am, ap, av = x_axis, y_axis, z_axis
    # Normalization
    am = am / max(abs(am))
    ap = ap / max(abs(ap))
    av = av / max(abs(av))
    apMean = np.mean(ap)  # Mean of ap as best estimate of sin(theta_p)
    amMean = np.mean(am)  # Mean of am best estimate of sin(theta_m)
    # Equations
    aP = ap * math.cos(math.asin(apMean)) - av * apMean  # Equation1
    avv = ap * apMean + av * math.cos(math.asin(apMean))  # Equation 2
    aM = am * math.cos(math.asin(amMean)) - avv * amMean  # Equation 3
    aV = am * amMean + avv * math.cos(math.asin(amMean)) - g  # Equation 4
    AccDataDown = aV - np.mean(aV)
    # Filtering
    # The 4th order Butterworth filter with a cut off frequency of 15-20 Hz.
    AccDataDown = signal.detrend(AccDataDown)
    Wn = fc / (fs / 2)
    AccDataDownFilt = helper_functions.butter_worth_lowpass(4, Wn, AccDataDown)
    # Integration and peak detection
    Integratedav = integrate.cumtrapz(AccDataDownFilt)
    x = np.array(range(0, len(Integratedav))) / fs
    a = (Integratedav[-1] - Integratedav[0]) / (x[-1] - x[0])
    b = Integratedav[0] - a * x[0]
    con = np.conjugate(np.transpose(x))
    Integratedav1 = Integratedav - a * con - b
    CWTIntegratedav, _ = cwt(Integratedav1, 10, 'gaus1')
    CWTIntegratedav = np.squeeze(CWTIntegratedav)
    pi = float(3.14285714)
    S1 = np.negative(CWTIntegratedav) / (10 ** (3 / 2)) / (2 * pi) ** (1 / 4)
    IC, Peaks1 = signal.find_peaks(np.negative(S1))

    x = np.array(range(0, len(S1))) / fs
    a = (S1[-1] - S1[0]) / (x[-1] - x[0])
    b = S1[0] - a * x[0]

    S11 = S1 - a * x - b
    S2, _ = cwt(S11, 10, 'gaus1')
    S2 = np.squeeze(S2)
    S2 = np.negative(S2)
    FC, Peaks2 = signal.find_peaks(S2)

    min_length = min(len(FC), len(IC))

    # Create Arrays
    StrideTime = np.zeros(min_length - 1)
    StanceTime = np.zeros(min_length - 1)

    for i in range(0, (min_length - 1)):
        StrideTime[i] = IC[i + 1] - IC[i]
        StanceTime[i] = FC[i + 1] - FC[i]
    SwingTime = StrideTime - StanceTime

    mean_StrideTime = np.mean(StrideTime[1:len(StrideTime) - 1]) / fs
    time_instants = np.array(np.arange(0 / fs, len(AccDataDownFilt) / fs, 1 / fs))
    vel = integrate.cumtrapz(AccDataDownFilt, time_instants)
    displacement = integrate.cumtrapz(vel, time_instants[:-1])
    Wn = 0.1 / (fs / 2)
    displacement_filt = helper_functions.butter_worth_highpass(4, Wn, displacement)

    h = np.zeros(len(IC) - 1)
    StrideLength = np.zeros(len(IC) - 1)
    for xx in range(0, len(IC) - 1):
        step = abs(displacement_filt[IC[xx]:IC[xx + 1]])
        h[xx] = max(step) - min(step)
        StrideLength[xx] = abs(2 * math.sqrt(2 * 25 * Wearable_Height * h[xx] - math.pow(h[xx], 2)))
    mean_StrideLength = np.mean(StrideLength)
    StepVelocity = mean_StrideLength / mean_StrideTime

    size1 = min(len(StrideTime), len(StrideLength))
    StrideTimes = StrideTime[1:size1] / fs
    StrideLengths = StrideLength[1:size1]
    StrideVelocity = np.mean(StrideLengths / StrideTimes)

    meanStrideTime = np.mean(StrideTimes)
    mean_StepLength = mean_StrideLength / 2
    mean_StepTime = meanStrideTime / 2

    return StrideVelocity, StepVelocity, mean_StrideLength, mean_StepLength, meanStrideTime, mean_StepTime