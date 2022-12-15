import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable
from tensorly import tucker_to_tensor, tucker_to_unfolded, unfold
from tensorly.decomposition import tucker


class TensorRPCANet(nn.Module):
	'''
	Recurrent neural network for tensor RPCA.

	Parameters:
		z0_init: initial parameter value for z0--note that this value is passed through a softplus to get the actual z0 value
		z1_init: initial parameter value for z1--note that this value is passed through a softplus to get the actual z1 value
		eta_init: initial parameter value for step size--note that this value is passed through a softplus to get the actual step size
		decay_init: initial parameter value for decay rate--note that this value is passed through a sigmoid to get the actual decay rate
		device: device to load this model
		softplus_factor: a factor to multiple the softplus outputs for z0 and z1 to make the gradients nicer
		skip: modes to skip iterative updates for
		datatype: parameter datatypes
	'''
	def __init__(self, z0_init, z1_init, eta_init, decay_init, device, softplus_factor=0.01, skip=[], datatype=torch.float32):
		super().__init__()
		self.device = device
		self.z0 = nn.Parameter(Variable(torch.tensor(z0_init, dtype=datatype, device = device), requires_grad=True))
		self.z1 = nn.Parameter(Variable(torch.tensor(z1_init, dtype=datatype, device = device), requires_grad=True))
		self.eta = nn.Parameter(Variable(torch.tensor(eta_init, dtype=datatype, device = device), requires_grad=True))
		self.decay = nn.Parameter(Variable(torch.tensor(decay_init, dtype=datatype, device = device), requires_grad=True))

		self.softplus_factor = softplus_factor
		self.skip = skip
		self = self.to(device)

	def thre(self, inputs, threshold):
		'''
		Soft thresholding.

		Args:
			inputs: input tensor
			threshold: threshold value >=0
		Output:
			out: soft thresholding outputs
		'''
		out = torch.sign(inputs) * torch.relu( torch.abs(inputs) - threshold)
		return out

	def forward(self, Y, ranks, num_l, epsilon=1e-9):
		'''
		Forward method of network.

		Args:
			Y: input tensor
			ranks: multilinear rank of low rank tensor
			num_l: number of iterative updates of ScaledGD i.e. number of recurrent layers
			epsilon: bias term for matrix inverse stability
		Output:
			X: low rank tensor from last iteration
			S: sparse tensor from last iteration
		'''

		z0 = self.softplus_factor * F.softplus(self.z0)
		z1 = self.softplus_factor * F.softplus(self.z1)
		eta = F.softplus(self.eta)
		decay = torch.sigmoid(self.decay)
		
		## Initialization
		G_t, factors_t = tucker(Y - self.thre(Y, z0), rank=ranks)
		order = len(ranks)

		ATA_inverses_skipped = dict()
		ATA_skipped = dict()
		for k in self.skip:
			ATA_skipped[k] = factors_t[k].T @ factors_t[k]
			ATA_inverses_skipped[k] = torch.linalg.inv(ATA_skipped[k]) 

		## Main Loop in ScaledGD RPCA
		for t in range(num_l):
			X_t = tucker_to_tensor((G_t, factors_t))
			S_t1 = self.thre(Y- X_t, z1 * (decay**t))
			factors_t1 = []
			D = S_t1 - Y
			ATA_t = []
			for k in range(order):
				if k in self.skip:
					ATA_t.append(ATA_skipped[k])
				else:
					ATA_t.append(factors_t[k].T @ factors_t[k])

			for k in range(order):
				if k in self.skip:
					factors_t1.append(factors_t[k])
					continue 
				
				A_t = factors_t[k]
				factors_t_copy = factors_t.copy()
				factors_t_copy[k] = torch.eye(A_t.shape[1]).to(self.device)
				A_breve_t = tucker_to_unfolded((G_t, factors_t_copy), k).T

				ATA_t_copy = ATA_t.copy()
				ATA_t_copy[k] = torch.eye(A_t.shape[1]).to(self.device)
				AbTAb_t = tucker_to_unfolded((G_t, ATA_t_copy), k) @ unfold(G_t, k).T

				ker = torch.linalg.inv(AbTAb_t + epsilon * torch.eye(A_breve_t.shape[1]).to(self.device))
				A_t1 = (1 - eta) * A_t - eta * unfold(D, k) @ A_breve_t @ ker
				factors_t1.append(A_t1)
			G_factors_t = []
			for k in range(order):
				if k in self.skip:
					G_factors_t.append(ATA_inverses_skipped[k] @ factors_t[k].T)
				else:
					G_factors_t.append(torch.linalg.inv(ATA_t[k] + epsilon * torch.eye(factors_t[k].shape[1]).to(self.device)) @ factors_t[k].T)
			G_t1 = G_t - eta  * tucker_to_tensor((X_t + D, G_factors_t))
			factors_t = factors_t1
			G_t = G_t1
		
		return tucker_to_tensor((G_t, factors_t)), S_t1

