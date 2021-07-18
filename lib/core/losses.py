'''
 @Time : 2021/1/23 9:53 下午 
 @Author : Moming_Y
 @File : losses.py 
 @Software: PyCharm
'''

import torch
import torch.nn as nn

class Cross_entropy(nn.Module):
	"""docstring for F"""
	def __init__(self, alpha=5, eps=1e-7, size_average=True):
		super(Cross_entropy, self).__init__()
		self.eps = eps
		self.size_average = size_average
		self.alpha = alpha

	def forward(self, inputs, target):
		#y = one_hot(target,inputs.size(-1))
		inputs = inputs.clamp(self.eps, 1 - self.eps)
		loss = -(self.alpha * target * torch.log(inputs) + (1-target) * torch.log(1-inputs))
		#loss =-(self.alpha*target*torch.pow((1-inputs),2)*torch.log(inputs)+(1-self.alpha)*(1-target)*torch.pow(inputs,2)*torch.log(1-inputs))
		#loss = loss*(1-inputs)**self.gamma

		if self.size_average: return loss.mean()
		else: return loss.sum()