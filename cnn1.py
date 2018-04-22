import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

EPOCH = 2
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
	DOWNLOAD_MNIST=True

train_data= torchvision.datasets.MNIST(
	root='./mnist/',
	train=True,
	transform=torchvision.transforms.ToTensor(),
	download=DOWNLOAD_MNIST,
	)

train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE,shuffle=True)

test_data=torchvision.datasets.MNIST(
	root='./mnist/',
	train=False
	)
test_x=Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y=test_data.test_labels[:2000]

class CNN(nn.Module):
	"""docstring for CNN"""
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1= nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=8,
				kernel_size=3,
				stride=1,
				padding=1,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
		)
		self.conv2=nn.Sequential(nn.Conv2d(8,16,3,1,1),nn.ReLU(),nn.MaxPool2d(2))
		self.conv3=nn.Sequential(nn.Conv2d(16,32,3,1,1),nn.ReLU())
		self.out=nn.Linear(32*7*7,10)
		#self.out=nn.Softmax()

	def forward(self,x):
		x=self.conv1(x)
		x=self.conv2(x)
		x=self.conv3(x)
		x=x.view(x.size(0),-1)
		output=self.out(x)
		return output,x

cnn=CNN()
print(cnn)

optimizer=torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
	for step, (x,y) in enumerate(train_loader):
		b_x=Variable(x)
		b_y=Variable(y)

		output=cnn(b_x)[0]
		#print(output)
		loss=loss_func(output, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step%50==0:
			test_output, last_layer=cnn(test_x)
			pred_y=torch.max(test_output, 1)[1].data.squeeze()
			accuracy=sum(pred_y==test_y)/float(test_y.size(0))
			print('Epoch:',epoch,'| train loss: %.4f'%loss.data[0],'| test accuracy: %.2f' % accuracy)




		
