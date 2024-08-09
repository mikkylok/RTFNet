# coding:utf-8
# Original By Yuxiang Sun, Aug. 2, 2019
# Modified By Meixi Lu

import torch
import torch.nn as nn 
import torchvision.models as models 


class RTFNet(nn.Module):

    def __init__(self, n_class, num_resnet_layers=50, num_lstm_layers=1, lstm_hidden_size=512):
        super(RTFNet, self).__init__()

        self.num_resnet_layers = num_resnet_layers

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048

        ########  Thermal ENCODER  ########
 
        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
 
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        # LSTM module
        self.lstm = nn.LSTM(input_size=self.inplanes, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)

        # Classifier
        self.classifier = nn.Linear(lstm_hidden_size, n_class)

    def forward(self, input):

        batch_size, num_frames, _, _, _ = input.size()  # assuming input size is (batch, frames, channels, height, width)

        # Initialize an empty list to store features for each frame
        features = []

        for t in range(num_frames):
            rgb = input[:, t, :3]
            thermal = input[:, t, 3:]

            # encoder

            ######################################################################

            print("rgb.size() original: ", rgb.size())  # (480, 640)
            print("thermal.size() original: ", thermal.size()) # (480, 640)

            ######################################################################

            rgb = self.encoder_rgb_conv1(rgb)
            print("rgb.size() after conv1: ", rgb.size()) # (240, 320)
            rgb = self.encoder_rgb_bn1(rgb)
            print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
            rgb = self.encoder_rgb_relu(rgb)
            print("rgb.size() after relu: ", rgb.size())  # (240, 320)

            thermal = self.encoder_thermal_conv1(thermal)
            print("thermal.size() after conv1: ", thermal.size()) # (240, 320)
            thermal = self.encoder_thermal_bn1(thermal)
            print("thermal.size() after bn1: ", thermal.size()) # (240, 320)
            thermal = self.encoder_thermal_relu(thermal)
            print("thermal.size() after relu: ", thermal.size())  # (240, 320)

            rgb = rgb + thermal

            rgb = self.encoder_rgb_maxpool(rgb)
            print("rgb.size() after maxpool: ", rgb.size()) # (120, 160)

            thermal = self.encoder_thermal_maxpool(thermal)
            print("thermal.size() after maxpool: ", thermal.size()) # (120, 160)

            ######################################################################

            rgb = self.encoder_rgb_layer1(rgb)
            print("rgb.size() after layer1: ", rgb.size()) # (120, 160)
            thermal = self.encoder_thermal_layer1(thermal)
            print("thermal.size() after layer1: ", thermal.size()) # (120, 160)

            rgb = rgb + thermal

            ######################################################################

            rgb = self.encoder_rgb_layer2(rgb)
            print("rgb.size() after layer2: ", rgb.size()) # (60, 80)
            thermal = self.encoder_thermal_layer2(thermal)
            print("thermal.size() after layer2: ", thermal.size()) # (60, 80)

            rgb = rgb + thermal

            ######################################################################

            rgb = self.encoder_rgb_layer3(rgb)
            print("rgb.size() after layer3: ", rgb.size()) # (30, 40)
            thermal = self.encoder_thermal_layer3(thermal)
            print("thermal.size() after layer3: ", thermal.size()) # (30, 40)

            rgb = rgb + thermal

            ######################################################################

            rgb = self.encoder_rgb_layer4(rgb)
            print("rgb.size() after layer4: ", rgb.size()) # (15, 20)
            thermal = self.encoder_thermal_layer4(thermal)
            print("thermal.size() after layer4: ", thermal.size()) # (15, 20)

            fuse = rgb + thermal

            ######################################################################

            # Global Average Pooling
            fuse = nn.AdaptiveAvgPool2d((1, 1))(fuse)
            fuse = fuse.view(batch_size, -1)  # Flatten

            features.append(fuse)

        # Stack the features along the time dimension
        features = torch.stack(features, dim=1)  # shape (batch, frames, features)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features)

        # Classification
        output = self.classifier(lstm_out[:, -1, :])  # Use the last LSTM output for classification

        return output


def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
    rtf_net = RTFNet(9).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    rtf_net(input)
    #print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()
