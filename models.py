from torch import nn
import torch
import torch.nn.functional as F
import timm
import torchxrayvision as xrv

from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

from torchvision.models import resnet101
from torchvision.models import ResNet101_Weights

from torchvision.models import mobilenet_v3_small
from torchvision.models import MobileNet_V3_Small_Weights

from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

from torchvision.models import squeezenet1_1
from torchvision.models import SqueezeNet1_1_Weights

from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights

from torchvision.models import densenet121
from torchvision.models import DenseNet121_Weights

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

import utils.utils as utils
import clip


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    



class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=False, residual=False) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.relu1 = nn.ReLU()

        if batch_norm:
            self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.relu2 = nn.ReLU()

        if batch_norm:
            self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)

        self.batch_norm = batch_norm

        self.residual = residual
        self.diff_channels = False

        if out_channels != in_channels:
            self.residual = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm:
            out = self.batch_norm_1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.batch_norm_2(out)
        
        if self.residual:
            out = torch.add(out, residual)

        out = self.relu2(out)

        return out

class CNN_PRO(nn.Module):
    def __init__(self, num_classes=1, in_channels=3,
                 cnn_size=[8, 16, 32], cnn_kernels=[7, 5, 3], avg_pool_size=8,
                 fc_size=[1024, 512, 128], 
                 batch_norm=False, dropout=False, residual=False) -> None:
        super().__init__()

        assert len(cnn_size) == len(cnn_kernels)

        feature_extractor_list = []

        for i in range(len(cnn_size)):
            if i == 0:
                feature_extractor_list.append(CNN_block(in_channels=in_channels, out_channels=cnn_size[i], 
                                                        kernel_size=cnn_kernels[i], batch_norm=batch_norm, residual=residual))
            else:
                feature_extractor_list.append(CNN_block(in_channels=cnn_size[i-1], out_channels=cnn_size[i], 
                                                        kernel_size=cnn_kernels[i], batch_norm=batch_norm, residual=residual))
            if i != len(cnn_size)-1:
                feature_extractor_list.append(nn.MaxPool2d(kernel_size=2))

        feature_extractor_list.append(nn.AdaptiveAvgPool2d(output_size=avg_pool_size))

        self.feature_extractor = nn.Sequential(*feature_extractor_list)


        classifier_list = []
        for i in range(len(fc_size)):
            if i == 0:
                classifier_list.append(nn.Linear(in_features=cnn_size[-1]*avg_pool_size*avg_pool_size, out_features=fc_size[i]))
            else:
                classifier_list.append(nn.Linear(in_features=fc_size[i-1], out_features=fc_size[i]))
 
            
            if batch_norm:
                    classifier_list.append(nn.BatchNorm1d(num_features=fc_size[i]))

            classifier_list.append(nn.ReLU())

            if dropout:
                classifier_list.append(nn.Dropout(p=0.5))

        classifier_list.append(nn.Linear(in_features=fc_size[-1], out_features=num_classes))

        self.classifier = nn.Sequential(*classifier_list)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)
        out = self.classifier(features)
        return out

    def num_of_params(self):
        total = 0
        for layer_params in self.feature_extractor.parameters():
            total += layer_params.numel()
        for layer_params in self.classifier.parameters():
            total += layer_params.numel()
        return total   
    

class MLP(nn.Module):
    def __init__(self, num_classes=1, in_features=3*224*224, fc_size=[4096, 2048, 1024, 512, 256, 128], 
                 batch_norm=True, dropout=True) -> None:
        super().__init__()

        classifier_list = []
        for i in range(len(fc_size)):
            if i == 0:
                classifier_list.append(nn.Linear(in_features=in_features, out_features=fc_size[i]))
            else:
                classifier_list.append(nn.Linear(in_features=fc_size[i-1], out_features=fc_size[i]))
 
            
            if batch_norm:
                    classifier_list.append(nn.BatchNorm1d(num_features=fc_size[i]))

            classifier_list.append(nn.ReLU())

            if dropout:
                classifier_list.append(nn.Dropout(p=0.5))

        classifier_list.append(nn.Linear(in_features=fc_size[-1], out_features=num_classes))

        self.classifier = nn.Sequential(*classifier_list)
        
    def forward(self, x):
        features = torch.flatten(x, start_dim=1)
        out = self.classifier(features)
        return out

    def num_of_params(self):
        total = 0
        for layer_params in self.classifier.parameters():
            total += layer_params.numel()
        return total


def get_vgg16(pretrained=False):
    if pretrained:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.features[26].parameters():
            param.requires_grad = True

        for param in model.features[28].parameters():
            param.requires_grad = True
        
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = True

    else:
        model = vgg16()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.classifier.add_module("7", torch.nn.Linear(1000, 1))
    
    return model


def get_vgg16_cnn(pretrained=False):
    if pretrained:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.features[26].parameters():
            param.requires_grad = True

        for param in model.features[28].parameters():
            param.requires_grad = True
        
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = True

    else:
        model = vgg16()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=25088,
            out_features=1,
            bias=True
        )
    )
    
    return model


def get_MLP(pretrained=False):
    model = MLP()
    
    return model

def get_resnet50(task, pretrained=False,num_classes=10):
    if task=='Classification':
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = resnet50()
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.fc = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=2048,
                out_features=num_classes,
                bias=True
            )
        )
    else:
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = resnet50()
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.fc = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=2048,out_features=1),
            # torch.nn.BatchNorm1d(num_features=128),
            # torch.nn.ELU(),
            # torch.nn.Linear(in_features=128,out_features=1),
            # torch.nn.Hardtanh(min_val=0.,max_val=4.)
        )

    
    return model



def get_resnet18(pretrained=False, num_classes=10):
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        # for param in model.layer4.parameters():
        #     param.requires_grad = True

        # for i, (name, layer) in enumerate(model.layer4.named_modules()):
        #     if isinstance(layer, torch.nn.Conv2d):
        #         layer.reset_parameters()

    else:
        model = resnet18()

    # Add on fully connected layers for the output of our model

    # model.fc = torch.nn.Sequential(
    #     nn.Dropout(0.2),
    #     torch.nn.Linear(
    #         in_features=512,
    #         out_features=num_classes,
    #         bias=True
    #     ) 
    # )

    model.fc = torch.nn.Linear(
            in_features=512,
            out_features=num_classes
    ) 

    return model

def get_resnet101(pretrained=False):
    if pretrained:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for i, (name, layer) in enumerate(model.layer4.named_modules()):
            if isinstance(layer, torch.nn.Conv2d):
                layer.reset_parameters()

    else:
        model = resnet101()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=2048,
            out_features=1,
            bias=True
        )
    )
    
    return model


def get_mobilenet_v3_small(pretrained=False):
    if pretrained:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(10):
            for param in model.features[i].parameters():
                param.requires_grad = False

    else:
        model = mobilenet_v3_small()

    # Add on fully connected layers for the output of our model

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=576,
            out_features=1,
            bias=True
        )
    )

    return model


def get_mobilenet_v3_large(pretrained=False):
    if pretrained:
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for i in range(13):
            for param in model.features[i].parameters():
                param.requires_grad = False

        for i in range(13, 17):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = mobilenet_v3_large()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=960,
            out_features=512,
            bias=True
        ),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=512,
            out_features=1,
            bias=True
        )
    )

    return model

def get_efficientnet_v2_s(pretrained=False):
    if pretrained:
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(6):
            for param in model.features[i].parameters():
                param.requires_grad = False

        for i in range(6, 8):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = efficientnet_v2_s()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.2),
        torch.nn.Linear(
            in_features=1280,
            out_features=1,
            bias=True
        )
    )
    return model

def get_squeezenet1_1(pretrained=False):
    if pretrained:
        model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(10):
            for param in model.features[i].parameters():
                param.requires_grad = False

    else:
        model = squeezenet1_1()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.5),
        torch.nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=1,
            stride=1
        ),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    )

    return model


def get_convnext_tiny(pretrained=False):
    if pretrained:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(6):
            for param in model.features[i].parameters():
                param.requires_grad = False

        for i in range(6, 8):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = convnext_tiny()

    def _is_contiguous(tensor: torch.Tensor) -> bool:
        # jit is oh so lovely :/
        # if torch.jit.is_tracing():
        #     return True
        if torch.jit.is_scripting():
            return tensor.is_contiguous()
        else:
            return tensor.is_contiguous(memory_format=torch.contiguous_format)

    class LayerNorm2d(nn.LayerNorm):
        r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
        """

        def __init__(self, normalized_shape, eps=1e-6):
            super().__init__(normalized_shape, eps=eps)

        def forward(self, x) -> torch.Tensor:
            if _is_contiguous(x):
                return F.layer_norm(
                    x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
            else:
                s, u = torch.var_mean(x, dim=1, keepdim=True)
                x = (x - u) * torch.rsqrt(s + self.eps)
                x = x * self.weight[:, None, None] + self.bias[:, None, None]
                return x

    model.classifier = torch.nn.Sequential(
        LayerNorm2d(768, eps=1e-06),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(
            in_features=768,
            out_features=1,
            bias=True
        )
    )

    return model
    
def get_deitS(task, pretrained=False,num_classes=10):
    if task == 'Classification':
        if pretrained:
            model = timm.create_model('deit_base_patch16_224.fb_in1k',pretrained=True)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = timm.create_model('deit_base_patch16_224.fb_in1k',pretrained=False)
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.head = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=model.head.in_features,
                out_features=num_classes,
                bias=True
            )
        )
    else:
        if pretrained:
            model = timm.create_model('deit_base_patch16_224.fb_in1k',pretrained=True)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = timm.create_model('deit_base_patch16_224.fb_in1k',pretrained=False)
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.head = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=model.head.in_features,
                out_features=1,
                bias=True
            )
        )

    return model


def get_medical_densnet121(pretrained=False,num_classes=2):
    if pretrained:
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False
        # for param in model.layer4.parameters():
        #     param.requires_grad = True

        # for i, (name, layer) in enumerate(model.layer4.named_modules()):
        #     if isinstance(layer, torch.nn.Conv2d):
        #         layer.reset_parameters()

    else:
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.classifier = torch.nn.Sequential(torch.nn.Linear(
            in_features=1024,
            out_features=num_classes,
            bias=True
        ))
    return model


def get_densnet121(task, pretrained=False,num_classes=2):
    if task == 'Classification':
        if pretrained:
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = densenet121()
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.classifier = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=1024,
                out_features=num_classes,
                bias=True
            )
        )
        
    else:
        if pretrained:
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = densenet121()
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.classifier = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=1024,
                out_features=1,
                bias=True
            )
        )
        
    return model



def get_CLIP(classnames, pretrained=False, num_classes=10):
    classnames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    base_model, preprocess = clip.load('ViT-B/32', 'cuda', jit = False)
    template = utils.openai_imagenet_template
    clf = utils.zeroshot_classifier(base_model, classnames, template, 'cuda')
    feature_dim = base_model.visual.output_dim
    model = utils.ModelWrapper(base_model, feature_dim, num_classes, normalize=True, initial_weights=clf)
    for p in model.parameters():
        p.data = p.data.float()

    return model


def get_model(model_name, task = 'Classification', pretrained=False, num_classes=10, classnames = None):
    if model_name == "ResNet50":
        return get_resnet50(task, pretrained, num_classes)
    elif model_name == "ResNet18":
        return get_resnet18(pretrained, num_classes)
    elif model_name == "ResNet101":
        return get_resnet101(pretrained)
    elif model_name == "MobileNetV3Small":
        return get_mobilenet_v3_small(pretrained)
    elif model_name == "MobileNetV3Large":
        return get_mobilenet_v3_large(pretrained)
    elif model_name == "EfficientNetV2S":
        return get_efficientnet_v2_s(pretrained)
    elif model_name == "SqueezeNet1_1":
        return get_squeezenet1_1(pretrained)
    elif model_name == "ConvNeXtTiny":
        return get_convnext_tiny(pretrained)
    elif model_name == "DeiT-S":
        return get_deitS(task, pretrained, num_classes)
    elif model_name == "MIMIC-DenseNet121":
        return get_medical_densnet121(pretrained, num_classes)
    elif model_name == "DenseNet121":
        return get_densnet121(task, pretrained, num_classes)
    elif model_name == "CLIP":
        return get_CLIP(classnames, pretrained, num_classes)
        
    elif model_name == "LeNet5":
        return LeNet5(2)
    elif model_name == "CNN_PRO":
        return CNN_PRO(cnn_size=[16, 16, 32, 32, 32, 32], cnn_kernels=[7, 5, 5, 3, 3, 3], 
                        avg_pool_size=8, fc_size=[1024, 512, 256, 128], 
                        dropout=True, batch_norm=True)
    elif model_name == "VGG16":
        return get_vgg16(pretrained)
    elif model_name == "VGG16_CNN":
        return get_vgg16_cnn(pretrained)
    elif model_name == "MLP":
        return get_MLP(pretrained)
    else:
        raise Exception("Model not implemented")