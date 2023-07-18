# Installing dependencies:
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
#from fastapi.middleware.cors import CORSMiddleware
#from pyngrok import ngrok
#import nest_asyncio


#app creation:
app = FastAPI()

'''
# creating Middleware
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)
'''

# creating Basemodel:
class Features(BaseModel):
   SepalLengthCm:float
   SepalWidthCm:float
   PetalLengthCm:float
   PetalWidthCm:float

# load pickle saved file:
iris_model = pickle.load(open(r'c:\Users\mitin\Downloads\iris_model .pkl','rb'))

# operation creation for post to client:
@app.post('/iris_prediction')
def iris_pred(input_parameters:Features):
  input_parameters = input_parameters.dict()

  sl = input_parameters['SepalLengthCm']
  sw = input_parameters['SepalWidthCm']
  pl = input_parameters['PetalLengthCm']
  pw = input_parameters['PetalWidthCm']

  input_list = [sl,sw,pl,pw]

  prediction = iris_model.predict([input_list])
  if (prediction[0] == 0):
    return 'The species is Iris-versicolor'
  elif (prediction == 1):
    return 'The species is Iris-virginica'
  else:
    return 'The species is Iris-setosa'
  
'''
url = ngrok.connect(8000)
print('Public URL:', url.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
'''