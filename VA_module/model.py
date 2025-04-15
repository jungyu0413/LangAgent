from src.resnet import *
from torch.autograd import Variable



class Model(nn.Module):
    
    def __init__(self, conf, pretrained=True, num_classes=7):
        super(Model, self).__init__()
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        self.task = conf.task
        self.embedding = conf.feature_embedding
        self.num_classes = conf.num_classes
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  
        self.features2 = nn.Sequential(*list(resnet50.children())[-2:-1])  
        self.fc = nn.Linear(self.embedding, self.num_classes)  
        
        if self.task == 'va':
            self.val_fc = nn.Linear(self.embedding, 1)
            self.aro_fc = nn.Linear(self.embedding, 1)
                    
    def forward(self, x):        
        x = self.features(x)
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        #### 1, 2048, 1, 1
        
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)
        
        params = list(self.parameters())
        if self.task == 'va':
            fc_weights = params[-6].data
        else:
            fc_weights = params[-2].data
        fc_weights = fc_weights.view(1, self.num_classes, self.embedding, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad = False)

        # attention
        feat = x.unsqueeze(1) # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2) # N * self.num_labels * H * W
        
        if self.task == 'va':
            val = self.val_fc(feature)
            aro = self.aro_fc(feature)
        
            return output, val, aro, feature, hm
        else:
            
            return output,feature, hm