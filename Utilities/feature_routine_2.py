import numpy as np
import scipy as sp
import pandas as pd

from Utilities import helper_functions
from Utilities import gait_algorithm

def mega_process(main_data):
    #Static Variable Generation

    mean_Left_AccX_Array = []
    mean_Left_AccY_Array = []
    mean_Left_AccZ_Array = []

    min_Left_AccX_Array = []
    min_Left_AccY_Array = []
    min_Left_AccZ_Array = []

    max_Left_AccX_Array = []
    max_Left_AccY_Array = []
    max_Left_AccZ_Array = []

    EME_Left_AccX_Array = []
    EME_Left_AccY_Array = []
    EME_Left_AccZ_Array = []

    IQR_Left_AccX_Array = []
    IQR_Left_AccY_Array = []
    IQR_Left_AccZ_Array = []

    SKEW_Left_AccX_Array = []
    SKEW_Left_AccY_Array = []
    SKEW_Left_AccZ_Array = []

    KURT_Left_AccX_Array = []
    KURT_Left_AccY_Array = []
    KURT_Left_AccZ_Array = []

    Stride_Vel_Left_Array = []
    Step_Vel_Left_Array = []
    Stride_Len_Left_Array = []
    Step_Len_Left_Array = []
    Stride_Time_Left_Array = []
    Step_Time_Left_Array = []

    mean_Right_AccX_Array = []
    mean_Right_AccY_Array = []
    mean_Right_AccZ_Array = []

    min_Right_AccX_Array = []
    min_Right_AccY_Array = []
    min_Right_AccZ_Array = []

    max_Right_AccX_Array = []
    max_Right_AccY_Array = []
    max_Right_AccZ_Array = []

    EME_Right_AccX_Array = []
    EME_Right_AccY_Array = []
    EME_Right_AccZ_Array = []

    IQR_Right_AccX_Array = []
    IQR_Right_AccY_Array = []
    IQR_Right_AccZ_Array = []

    SKEW_Right_AccX_Array = []
    SKEW_Right_AccY_Array = []
    SKEW_Right_AccZ_Array = []

    KURT_Right_AccX_Array = []
    KURT_Right_AccY_Array = []
    KURT_Right_AccZ_Array = []

    Stride_Vel_Right_Array = []
    Step_Vel_Right_Array = []
    Stride_Len_Right_Array = []
    Step_Len_Right_Array = []
    Stride_Time_Right_Array = []
    Step_Time_Right_Array = []

    grouper_array = []

    def second_gen(data):
        second_data = []
        length_frame = len(data)
        x = 1
        for y in range(0, length_frame):
            if ((data['frame'][y]) % 32 != 0):
                second_data.append(x)
            else:
                second_data.append(x)
                x += 1
        return second_data

    def ten_sec(data):
        ten_data = []
        unique_seconds = data['Second_Data'].unique()
        x = 1
        for i in unique_seconds:
            if (i % 10 != 0):
                ten_data.append(x)
            else:
                ten_data.append(x)
                x += 1
        ten_diction = dict(zip(unique_seconds, ten_data))
        ten_sec_array = data['Second_Data'].map(ten_diction)
        return ten_sec_array

    def feature_generation(data, grouper_id):

        mean_Left_Acc_X, mean_Left_Acc_Y, mean_Left_Acc_Z = helper_functions.mean(data['Left_Accl_X'], data['Left_Accl_Y'], data['Left_Accl_Z'])
        min_Left_Acc_X, min_Left_Acc_Y, min_Left_Acc_Z = helper_functions.minimum(data['Left_Accl_X'], data['Left_Accl_Y'], data['Left_Accl_Z'])
        max_Left_Acc_X, max_Left_Acc_Y, max_Left_Acc_Z = helper_functions.maximum(data['Left_Accl_X'], data['Left_Accl_Y'], data['Left_Accl_Z'])
        EME_Left_Acc_X, EME_Left_Acc_Y, EME_Left_Acc_Z = helper_functions.energy_measure(data['Left_Accl_X'], data['Left_Accl_Y'], data['Left_Accl_Z'])
        IQR_Left_Acc_X, IQR_Left_Acc_Y, IQR_Left_Acc_Z = helper_functions.inter_quartile_range(data['Left_Accl_X'], data['Left_Accl_Y'], data['Left_Accl_Z'])
        skew_Left_Acc_X, skew_Left_Acc_Y, skew_Left_Acc_Z = helper_functions.skewness(data['Left_Accl_X'], data['Left_Accl_Y'], data['Left_Accl_Z'])
        kurt_Left_Acc_X, kurt_Left_Acc_Y, kurt_Left_Acc_Z = helper_functions.kurt(data['Left_Accl_X'], data['Left_Accl_Y'], data['Left_Accl_Z'])
        Stride_Vel_Left, Step_Vel_Left, Stride_Len_Left, Step_Len_Left, Stride_Time_Left, Step_Time_Left = gait_algorithm.gait_params(data['Left_Accl_X'], data['Left_Accl_Y'], data['Left_Accl_Z'])

        mean_Right_Acc_X, mean_Right_Acc_Y, mean_Right_Acc_Z = helper_functions.mean(data['Right_Accl_X'], data['Right_Accl_Y'], data['Right_Accl_Z'])
        min_Right_Acc_X, min_Right_Acc_Y, min_Right_Acc_Z = helper_functions.minimum(data['Right_Accl_X'], data['Right_Accl_Y'], data['Right_Accl_Z'])
        max_Right_Acc_X, max_Right_Acc_Y, max_Right_Acc_Z = helper_functions.maximum(data['Right_Accl_X'], data['Right_Accl_Y'], data['Right_Accl_Z'])
        EME_Right_Acc_X, EME_Right_Acc_Y, EME_Right_Acc_Z = helper_functions.energy_measure(data['Right_Accl_X'], data['Right_Accl_Y'], data['Right_Accl_Z'])
        IQR_Right_Acc_X, IQR_Right_Acc_Y, IQR_Right_Acc_Z = helper_functions.inter_quartile_range(data['Right_Accl_X'], data['Right_Accl_Y'], data['Right_Accl_Z'])
        skew_Right_Acc_X, skew_Right_Acc_Y, skew_Right_Acc_Z = helper_functions.skewness(data['Right_Accl_X'], data['Right_Accl_Y'], data['Right_Accl_Z'])
        kurt_Right_Acc_X, kurt_Right_Acc_Y, kurt_Right_Acc_Z = helper_functions.kurt(data['Right_Accl_X'], data['Right_Accl_Y'], data['Right_Accl_Z'])
        Stride_Vel_Right, Step_Vel_Right, Stride_Len_Right, Step_Len_Right, Stride_Time_Right, Step_Time_Right = gait_algorithm.gait_params(data['Right_Accl_X'], data['Right_Accl_Y'], data['Right_Accl_Z'])

        grouper_array.append(grouper_id)

        mean_Left_AccX_Array.append(mean_Left_Acc_X)
        mean_Left_AccY_Array.append(mean_Left_Acc_Y)
        mean_Left_AccZ_Array.append(mean_Left_Acc_Z)

        min_Left_AccX_Array.append(min_Left_Acc_X)
        min_Left_AccY_Array.append(min_Left_Acc_Y)
        min_Left_AccZ_Array.append(min_Left_Acc_Z)

        max_Left_AccX_Array.append(max_Left_Acc_X)
        max_Left_AccY_Array.append(max_Left_Acc_Y)
        max_Left_AccZ_Array.append(max_Left_Acc_Z)

        EME_Left_AccX_Array.append(EME_Left_Acc_X)
        EME_Left_AccY_Array.append(EME_Left_Acc_Y)
        EME_Left_AccZ_Array.append(EME_Left_Acc_Z)

        IQR_Left_AccX_Array.append(IQR_Left_Acc_X)
        IQR_Left_AccY_Array.append(IQR_Left_Acc_Y)
        IQR_Left_AccZ_Array.append(IQR_Left_Acc_Z)

        SKEW_Left_AccX_Array.append(skew_Left_Acc_X)
        SKEW_Left_AccY_Array.append(skew_Left_Acc_Y)
        SKEW_Left_AccZ_Array.append(skew_Left_Acc_Z)

        KURT_Left_AccX_Array.append(kurt_Left_Acc_X)
        KURT_Left_AccY_Array.append(kurt_Left_Acc_Y)
        KURT_Left_AccZ_Array.append(kurt_Left_Acc_Z)

        Stride_Vel_Left_Array.append(Stride_Vel_Left)
        Step_Vel_Left_Array.append(Step_Vel_Left)
        Stride_Len_Left_Array.append(Stride_Len_Left)
        Step_Len_Left_Array.append(Step_Len_Left)
        Stride_Time_Left_Array.append(Stride_Time_Left)
        Step_Time_Left_Array.append(Step_Time_Left)

        mean_Right_AccX_Array.append(mean_Right_Acc_X)
        mean_Right_AccY_Array.append(mean_Right_Acc_Y)
        mean_Right_AccZ_Array.append(mean_Right_Acc_Z)

        min_Right_AccX_Array.append(min_Right_Acc_X)
        min_Right_AccY_Array.append(min_Right_Acc_Y)
        min_Right_AccZ_Array.append(min_Right_Acc_Z)

        max_Right_AccX_Array.append(max_Right_Acc_X)
        max_Right_AccY_Array.append(max_Right_Acc_Y)
        max_Right_AccZ_Array.append(max_Right_Acc_Z)

        EME_Right_AccX_Array.append(EME_Right_Acc_X)
        EME_Right_AccY_Array.append(EME_Right_Acc_Y)
        EME_Right_AccZ_Array.append(EME_Right_Acc_Z)

        IQR_Right_AccX_Array.append(IQR_Right_Acc_X)
        IQR_Right_AccY_Array.append(IQR_Right_Acc_Y)
        IQR_Right_AccZ_Array.append(IQR_Right_Acc_Z)

        SKEW_Right_AccX_Array.append(skew_Right_Acc_X)
        SKEW_Right_AccY_Array.append(skew_Right_Acc_Y)
        SKEW_Right_AccZ_Array.append(skew_Right_Acc_Z)

        KURT_Right_AccX_Array.append(kurt_Right_Acc_X)
        KURT_Right_AccY_Array.append(kurt_Right_Acc_Y)
        KURT_Right_AccZ_Array.append(kurt_Right_Acc_Z)

        Stride_Vel_Right_Array.append(Stride_Vel_Right)
        Step_Vel_Right_Array.append(Step_Vel_Right)
        Stride_Len_Right_Array.append(Stride_Len_Right)
        Step_Len_Right_Array.append(Step_Len_Right)
        Stride_Time_Right_Array.append(Stride_Time_Right)
        Step_Time_Right_Array.append(Step_Time_Right)




    #Convert the axis values to Absolute
    main_data['Left_Accl_X'] = main_data['Left_Accl_X'] / abs(main_data['Left_Accl_X'].max())
    main_data['Left_Accl_Y'] = main_data['Left_Accl_Y'] / abs(main_data['Left_Accl_Y'].max())
    main_data['Left_Accl_Z'] = main_data['Left_Accl_Z'] / abs(main_data['Left_Accl_Z'].max())
    main_data['Right_Accl_X'] = main_data['Right_Accl_X'] / abs(main_data['Right_Accl_X'].max())
    main_data['Right_Accl_Y'] = main_data['Right_Accl_Y'] / abs(main_data['Right_Accl_Y'].max())
    main_data['Right_Accl_Z'] = main_data['Right_Accl_Z'] / abs(main_data['Right_Accl_Z'].max())

    #filtering 4th Order Butterworth Filter
    main_data['Left_Accl_X'] = helper_functions.butter_worth_lowpass(4, 0.9375, main_data['Left_Accl_X'])
    main_data['Left_Accl_Y'] = helper_functions.butter_worth_lowpass(4, 0.9375, main_data['Left_Accl_Y'])
    main_data['Left_Accl_Z'] = helper_functions.butter_worth_lowpass(4, 0.9375, main_data['Left_Accl_Z'])
    main_data['Right_Accl_X'] = helper_functions.butter_worth_lowpass(4, 0.9375, main_data['Right_Accl_X'])
    main_data['Right_Accl_Y'] = helper_functions.butter_worth_lowpass(4, 0.9375, main_data['Right_Accl_Y'])
    main_data['Right_Accl_Z'] = helper_functions.butter_worth_lowpass(4, 0.9375, main_data['Right_Accl_Z'])

    #Second Generation
    main_data['Second_Data'] = second_gen(main_data)

    #10 Seconds grouping
    main_data['Ten_Sec'] = ten_sec(main_data)

    # Creating Feature Arrays
    for k in main_data['Ten_Sec'].unique():
        temp_ten = pd.DataFrame()
        temp_ten = main_data[main_data['Ten_Sec'] == k]
        feature_generation(temp_ten, k)

    feature_frame = pd.DataFrame()
    feature_frame['grouper'] = grouper_array
    feature_frame['mean_Left_AccX_Array'] = mean_Left_AccX_Array
    feature_frame['mean_Left_AccY_Array'] = mean_Left_AccY_Array
    feature_frame['mean_Left_AccZ_Array'] = mean_Left_AccZ_Array
    feature_frame['min_Left_AccX_Array'] = min_Left_AccX_Array
    feature_frame['min_Left_AccY_Array'] = min_Left_AccY_Array
    feature_frame['min_Left_AccZ_Array'] = min_Left_AccZ_Array
    feature_frame['max_Left_AccX_Array'] = max_Left_AccX_Array
    feature_frame['max_Left_AccY_Array'] = max_Left_AccY_Array
    feature_frame['max_Left_AccZ_Array'] = max_Left_AccZ_Array
    feature_frame['EME_Left_AccX_Array'] = EME_Left_AccX_Array
    feature_frame['EME_Left_AccY_Array'] = EME_Left_AccY_Array
    feature_frame['EME_Left_AccZ_Array'] = EME_Left_AccZ_Array
    feature_frame['IQR_Left_AccX_Array'] = IQR_Left_AccX_Array
    feature_frame['IQR_Left_AccY_Array'] = IQR_Left_AccY_Array
    feature_frame['IQR_Left_AccZ_Array'] = IQR_Left_AccZ_Array
    feature_frame['SKEW_Left_AccX_Array'] = SKEW_Left_AccX_Array
    feature_frame['SKEW_Left_AccY_Array'] = SKEW_Left_AccY_Array
    feature_frame['SKEW_Left_AccZ_Array'] = SKEW_Left_AccZ_Array
    feature_frame['KURT_Left_AccX_Array'] = KURT_Left_AccX_Array
    feature_frame['KURT_Left_AccY_Array'] = KURT_Left_AccY_Array
    feature_frame['KURT_Left_AccZ_Array'] = KURT_Left_AccZ_Array
    feature_frame['Stride_Vel_Left_Array'] = Stride_Vel_Left_Array
    feature_frame['Step_Vel_Left_Array'] = Step_Vel_Left_Array
    feature_frame['Stride_Len_Left_Array'] = Stride_Len_Left_Array
    feature_frame['Step_Len_Left_Array'] = Step_Len_Left_Array
    feature_frame['Stride_Time_Left_Array'] = Stride_Time_Left_Array
    feature_frame['Step_Time_Left_Array'] = Step_Time_Left_Array

    feature_frame['mean_Right_AccX_Array'] = mean_Right_AccX_Array
    feature_frame['mean_Right_AccY_Array'] = mean_Right_AccY_Array
    feature_frame['mean_Right_AccZ_Array'] = mean_Right_AccZ_Array
    feature_frame['min_Right_AccX_Array'] = min_Right_AccX_Array
    feature_frame['min_Right_AccY_Array'] = min_Right_AccY_Array
    feature_frame['min_Right_AccZ_Array'] = min_Right_AccZ_Array
    feature_frame['max_Right_AccX_Array'] = max_Right_AccX_Array
    feature_frame['max_Right_AccY_Array'] = max_Right_AccY_Array
    feature_frame['max_Right_AccZ_Array'] = max_Right_AccZ_Array
    feature_frame['EME_Right_AccX_Array'] = EME_Right_AccX_Array
    feature_frame['EME_Right_AccY_Array'] = EME_Right_AccY_Array
    feature_frame['EME_Right_AccZ_Array'] = EME_Right_AccZ_Array
    feature_frame['IQR_Right_AccX_Array'] = IQR_Right_AccX_Array
    feature_frame['IQR_Right_AccY_Array'] = IQR_Right_AccY_Array
    feature_frame['IQR_Right_AccZ_Array'] = IQR_Right_AccZ_Array
    feature_frame['SKEW_Right_AccX_Array'] = SKEW_Right_AccX_Array
    feature_frame['SKEW_Right_AccY_Array'] = SKEW_Right_AccY_Array
    feature_frame['SKEW_Right_AccZ_Array'] = SKEW_Right_AccZ_Array
    feature_frame['KURT_Right_AccX_Array'] = KURT_Right_AccX_Array
    feature_frame['KURT_Right_AccY_Array'] = KURT_Right_AccY_Array
    feature_frame['KURT_Right_AccZ_Array'] = KURT_Right_AccZ_Array
    feature_frame['Stride_Vel_Right_Array'] = Stride_Vel_Right_Array
    feature_frame['Step_Vel_Right_Array'] = Step_Vel_Right_Array
    feature_frame['Stride_Len_Right_Array'] = Stride_Len_Right_Array
    feature_frame['Step_Len_Right_Array'] = Step_Len_Right_Array
    feature_frame['Stride_Time_Right_Array'] = Stride_Time_Right_Array
    feature_frame['Step_Time_Right_Array'] = Step_Time_Right_Array

    # Replacing Null Values With Column Means
    feature_frame.fillna(feature_frame.mean(), inplace=True)

    return feature_frame


