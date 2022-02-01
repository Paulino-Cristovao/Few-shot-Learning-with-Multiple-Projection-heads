#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from loss import cuda


class Net(nn.Module):
    def __init__(self, latent, num_classes=100, norm=True, scale=True):
        super(Net,self).__init__()
        self.extractor = Extractor()
        self.embedding = Embedding(latent)
        self.proj = Projection(latent)
        self.classifier = Classifier(latent,num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale
       
   
    def norm_emb(self, x):
        if self.norm:
            x = self.l2_norm(x)
        return x

    def scale_(self, x):  
        if self.scale:
            x = self.s * x
        return x

        

    def _forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x2_,x2, x1_,x1 = self.proj(x)

        x2_ = self.scale_(x2_)
        x2 = self.scale_(x2)

        x1_ = self.scale_(x1_)
        x1 = self.scale_(x1)   
      
        return x2_, x2, x1_,x1

    
    
    def forward(self, x):
        feature2, _,feature1,_ = self._forward(x)

        logit = self.classifier(feature2)
        
        return logit, feature2, feature1


    def helper_extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x2_,x2,x1_,x1 = self.proj(x)

        #x2_ = self.l2_norm(x2_)
        #x2 = self.l2_norm(x2)

        #x1_ = self.l2_norm(x1_)
        #x1 = self.l2_norm(x1)
      
        return x2_,x2,x1_,x1
    

    def forward_wi_fc1(self, x):
        _,_,_,x = self.helper_extract(x)
        logit = self.classifier(x)
        
        return logit, x
     
    def forward_wi_fc1_(self, x):
        _,_,x,_ = self.helper_extract(x)
        logit = self.classifier(x)
        
        return logit, x
    
    
    def forward_wi_fc2(self, x):
        _,x,_,_ = self.helper_extract(x)
        logit = self.classifier(x)
        
        return logit, x
    
    def forward_wi_fc2_(self, x):
        x,_,_,_ = self.helper_extract(x)
        logit = self.classifier(x)
        
        return logit, x
    

    def extract(self, x):
        x = self.helper_extract(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor,self).__init__()
        basenet = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])
                

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x



class Embedding(nn.Module):
    def __init__(self,latent):
        super(Embedding,self).__init__()
        self.fc = nn.Linear(2048, 512)
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        #x = self.relu(x)
        
        return  x


class Projection(nn.Module):
    def __init__(self,latent):
        super(Projection,self).__init__()
        self.fc1 = nn.Linear(512, latent)
        self.fc2 = nn.Linear(latent, latent)
        self.relu = nn.ReLU()
        #self.drop = nn.Dropout(0.5)
    

    def l2_norm1(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output


    def forward(self, x):
        fc1 = self.l2_norm1(self.fc1(x))
        fc1_ = self.relu(fc1)

        fc2 = self.fc2(fc1_)
        fc2_ = self.l2_norm1(fc2)
        
        return  fc2_, fc2, fc1_, fc1



class Classifier(nn.Module):
    def __init__(self, latent,num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(latent, num_classes, bias=False)
        
    def forward(self, x):
        x = self.fc(x)      
       
        return x
