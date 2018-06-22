#from torch.utils.data import DataLoader

from torchvision import  datasets,transforms
import net

#超参数
batch_size=64
learn_rate=1e-2
num_epoches=20

#数据预处理，将数据标准化
#因为图像只有一个通道，所以transforms.Normalize([0.5],[0.5])，分别是均值和方差，如果三个通道，transforms.Normalize([a,b,c],[d,e,f])
data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

#下载训练集MNIST
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)

test_dataset=datasets.MNIST(root="./data",train=False,transform=data_tf)

# train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
# test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)