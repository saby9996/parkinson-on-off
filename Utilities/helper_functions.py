# Developed By Code|<Ill at 10/14/2019
# Developed VM IP 203.241.246.158

import pandas as pd
import numpy as np
import math
import base64
import io
from scipy import signal
from scipy import stats


#Resultant of Three Axws
def resultant(axis_x, axis_y, axis_z):
    '''
    :param axis_x: Axis of the Sensor
    :param axis_y: Axis of the Sensor
    :param axis_z: Axis of the Sensor
    :return: Single Array
    '''
    axis_x=np.array(axis_x)
    axis_y=np.array(axis_y)
    axis_z=np.array(axis_z)
    resultant_vec=[]
    adder=((axis_x*axis_x)+(axis_y*axis_y)+(axis_z*axis_z))
    for i in adder:
        resultant_vec.append(math.sqrt(i))

    return resultant_vec

#Parse DataFrame From Uploaded Variable
def parse_contents(contents, filename):
    #contents=contents[0]
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    print(df.shape)

    return df

#Butterworth Lowpass Filter
def butter_worth_lowpass(order, Wn, AccDataDown):
    B,A=signal.butter(order, Wn)
    AccDataDownFilt=signal.lfilter(B,A,AccDataDown)
    return AccDataDownFilt

def butter_worth_highpass(order, Wn, AccDataDown):
    B,A=signal.butter(order, Wn,btype='highpass')
    AccDataDownFilt=signal.lfilter(B,A,AccDataDown)
    return AccDataDownFilt


# Features
def mean(x, y, z):
    """Calculates mean"""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_z = np.mean(z)
    return mean_x, mean_y, mean_z


def std_dev(x, y, z):
    """Calculates standard deviation"""
    std_x = np.std(x)
    std_y = np.std(y)
    std_z = np.std(z)
    return std_x, std_y, std_z


def mad(x, y, z):
    """Calculates median absolute deviation"""
    mad_x = np.median(np.abs(x - np.median(x)))
    mad_y = np.median(np.abs(y - np.median(y)))
    mad_z = np.median(np.abs(z - np.median(z)))
    return mad_x, mad_y, mad_z


def minimum(x, y, z):
    """Calculates minimum"""
    return min(x), min(y), min(z)


def maximum(x, y, z):
    """Calculates maximum"""
    return max(x), max(y), max(z)


def energy_measure(x, y, z):
    """Calculates energy measures"""
    em_x = np.mean(np.square(x))
    em_y = np.mean(np.square(y))
    em_z = np.mean(np.square(z))
    return em_x, em_y, em_z


def inter_quartile_range(x, y, z):
    """Calculates inter-quartile range"""
    iqr_x = np.subtract(*np.percentile(x, [75, 25]))
    iqr_y = np.subtract(*np.percentile(y, [75, 25]))
    iqr_z = np.subtract(*np.percentile(z, [75, 25]))
    return iqr_x, iqr_y, iqr_z


def sma(x, y, z):
    """Calculates signal magnitude area"""
    abs_x = np.absolute(x)
    abs_y = np.absolute(y)
    abs_z = np.absolute(z)
    return np.mean(abs_x + abs_y + abs_z)


def skewness(x, y, z):
    """Calculates skewness"""
    skew_x = stats.skew(x)
    skew_y = stats.skew(y)
    skew_z = stats.skew(z)
    return skew_x, skew_y, skew_z


def kurt(x, y, z):
    """Calculates kurtosis"""
    kurt_x = stats.kurtosis(x, fisher=False)
    kurt_y = stats.kurtosis(y, fisher=False)
    kurt_z = stats.kurtosis(z, fisher=False)
    return kurt_x, kurt_y, kurt_z