import numpy as np

gender_map = {'Male': 0, 'Female': 1}
smoking_map = {'never': 0, 'former': 1, 'current': 2, 'unknown': 3, 'other': 4}

def preprocess_input(data_dict):
    data_dict['gender'] = gender_map.get(data_dict['gender'], 2)
    data_dict['smoking_history'] = smoking_map.get(data_dict['smoking_history'], 4)

    entrada = np.array([[
        data_dict['gender'],
        data_dict['age'],
        data_dict['hypertension'],
        data_dict['heart_disease'],
        data_dict['smoking_history'],
        data_dict['bmi'],
        data_dict['HbA1c_level'],
        data_dict['blood_glucose_level']
    ]], dtype=float)
    
    return entrada
