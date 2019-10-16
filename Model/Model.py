# Developed By Code|<Ill at 10/15/2019
# Developed VM IP 203.241.246.158

from sklearn.externals import joblib
from Utilities import helper_functions
from scipy import stats
import numpy as np

def prob_generator(probs):
    off_prob_array = []
    on_prob_array = []
    for i in range(0, len(probs)):
        off_prob_array.append(probs[i][0])
        on_prob_array.append(probs[i][1])

    off_prob = np.max(off_prob_array)
    on_prob = np.max(on_prob_array)

    return off_prob, on_prob

def pred_generator(preds):
    pred_mode = stats.mode(preds)[0][0]

    if pred_mode == 0:
        return 'Off'
    else:
        return 'On'

def Model_on_off():
    model_path='Model/on-off.pkl'
    model=joblib.load(model_path)

    return model

def model_call(data):
    remover = ['grouper', 'IQR_Left_AccY_Array', 'IQR_Right_AccY_Array', 'Stride_Len_Left_Array',
               'Stride_Len_Right_Array', 'Stride_Time_Left_Array', 'Stride_Time_Right_Array']

    data.drop(remover, axis=1, inplace=True)
    print("Feature Data",data.shape)
    model = Model_on_off()
    predictions = model.predict(data)
    print(predictions)

    prediction_mode = pred_generator(predictions)

    prediction_probs = model.predict_proba(data)
    off_probability, on_probability = prob_generator(prediction_probs)

    return prediction_mode, off_probability, on_probability







