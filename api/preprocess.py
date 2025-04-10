import numpy as np
# Valores categóricos para numéricos
gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
smoking_map = {'never': 0, 'former': 1, 'current': 2, 'not current': 3, 'No Info': 4}

def preprocess_input(data_dict):
    data_dict['gender'] = gender_map.get(data_dict['gender'], 2) #2 = 'Other', genero desconhecido
    data_dict['smoking_history'] = smoking_map.get(data_dict['smoking_history'], 4) # 4 = 'No Info', torna-se sem informacao, no caso de valor desconhecido
    # É como um .values, mas construído manualmente com os dados do dicionário
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
