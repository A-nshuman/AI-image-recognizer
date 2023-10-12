from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
import requests

app = Flask(__name__)

model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval() 

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        image = Image.open(file)
        image = preprocess(image)
        image = image.unsqueeze(0)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = output.max(1)
        
        labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
        labels = requests.get(labels_url).json()
        recognized_label = labels[predicted.item()]
        
        return jsonify({'result': recognized_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
