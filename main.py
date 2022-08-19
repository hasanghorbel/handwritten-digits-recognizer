import torch

from number_writer import Writer
from neuralnet import NeuralNet, input_size, hidden_size, num_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './model/model.pth'

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
