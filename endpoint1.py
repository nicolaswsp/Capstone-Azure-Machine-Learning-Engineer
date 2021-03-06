import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://22ebf534-b6c9-428f-951f-2f85cff8cf82.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = ''



# One set of data to score, so we get two results back
data = {
  "data": [
    {
      "Time": 472,
      "V1": -3.0435406239976,
      "V2": -3.15730712090228,
      "V3": 1.08846277997285,
      "V4": 2.2886436183814,
      "V5": 1.35980512966107,
      "V6": -1.06482252298131,
      "V7": 0.325574266158614,
      "V8": -0.0677936531906277,
      "V9": -0.270952836226548,
      "V10": -0.838586564582682,
      "V11": -0.414575448285725,
      "V12": -0.503140859566824,
      "V13": 0.676501544635863,
      "V14": -1.69202893305906,
      "V15": 2.00063483909015,
      "V16": 0.666779695901966,
      "V17": 0.599717413841732,
      "V18": 1.72532100745514,
      "V19": 0.283344830149495,
      "V20": 2.10233879259444,
      "V21": 0.661695924845707,
      "V22": 0.435477208966341,
      "V23": 1.37596574254306,
      "V24": -0.293803152734021,
      "V25": 0.279798031841214,
      "V26": -0.145361714815161,
      "V27": -0.252773122530705,
      "V28": 0.0357642251788156,
      "Amount": 529
    }
  ]
}
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


