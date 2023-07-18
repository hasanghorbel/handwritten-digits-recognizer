import argparse

import torch

from neuralnet import NeuralNet, hidden_size, input_size, num_classes
from number_writer import Writer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='recognize handwritten digits')
parser.add_argument('-p', '--path', default='model/model.pth',
                    type=str, help='path to model')
args = parser.parse_args()
model_path = args.path

loaded_model = NeuralNet(input_size, hidden_size, num_classes)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
loaded_model.eval()

# drawing part
output = Writer(28).draw()

# prediction part
with torch.no_grad():
    image = torch.Tensor(output).unsqueeze(0).to(device)
    output = loaded_model(image)
    _, predicted = torch.max(output, 1)
    print(predicted.item())
