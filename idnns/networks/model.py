import torch.nn as nn

class Model(nn.Module):
	def __init__(self, activation_function, layer_sizes, input_size, num_of_classes, save_file, covn_net):
		super(Model, self).__init__()
		self.save_file = save_file
		layer_sizes = [input_size] + layer_sizes
		self.linears = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
		self.last_layer = nn.Linear(layer_sizes[-1], num_of_classes)
		self.activation = nn.Tanh() if activation_function == 0 else nn.ReLU()
		self.softmax = nn.Softmax(dim=1) # for MI calculation only, not for loss calculation
        
	def forward(self, x):
		# x: shape (B, D)
		layers = []
		for layer in self.linears:
			x = layer(x)
			x = self.activation(x)
			layers.append(x)

		x = self.last_layer(x)
		layers.append(self.softmax(x))

		return x, layers
	
if __name__ == '__main__':
	import torch
	input = torch.randn(1, 12, requires_grad=True)
	model = Model(0, [10, 7, 5, 4, 3], 12, 2, None, None)
	pred, layers = model(input)
