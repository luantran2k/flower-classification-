#The trained model is saved as the classifier.h5
#The predictor.py shows the way to use that seperately. 

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

#Make sure the version is correct
classifier = load_model('./FlowersModel')


test_image = image.load_img('./test/tulip.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
print(np.argmax(result))
print(result)
{'dogs': 1, 'cats': 0}
if result[0][0] == 1:
    prediction = 'daisy'
elif result[0][1] == 1:
    prediction = 'dandelion'
elif result[0][2] == 1:
    prediction = 'rose'
elif result[0][3] == 1:
    prediction = 'sunflower'
else:
    prediction = 'tulip'
    
print("It's a "+prediction)
