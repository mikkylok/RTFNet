# coding:utf-8
# Original By Yuxiang Sun, Aug. 2, 2019
# Modified By Meixi Lu


import torch
import torch.nn as nn 
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import torch.nn.utils.rnn as rnn_utils


class RTFNet(nn.Module):

    def __init__(self, n_class=3,
                 num_resnet_layers=50,
                 num_lstm_layers=1,
                 lstm_hidden_size=512,
                 device=torch.device('cuda:0')):
        super(RTFNet, self).__init__()
        self.device = device  # Assign the device

        self.num_resnet_layers = num_resnet_layers

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            resnet_raw_model2 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            # resnet_raw_model1 = models.resnet50(pretrained=True)
            # resnet_raw_model2 = models.resnet50(pretrained=True)
            resnet_raw_model1 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            resnet_raw_model2 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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
        self.lstm.flatten_parameters()
        # Classifier
        self.classifier = nn.Linear(lstm_hidden_size, n_class)

        # Move all modules to the specified device
        self.to(self.device)

    def forward(self, rgb_images, thermal_images, lengths):

        # Initialize an empty list to store features for each frame
        features = []

        for t in range(rgb_images.size(1)):  # iterate over time dimension
            rgb = rgb_images[:, t]
            thermal = thermal_images[:, t]

            # encoder

            rgb = self.encoder_rgb_conv1(rgb)
            rgb = self.encoder_rgb_bn1(rgb)
            rgb = self.encoder_rgb_relu(rgb)

            thermal = self.encoder_thermal_conv1(thermal)
            thermal = self.encoder_thermal_bn1(thermal)
            thermal = self.encoder_thermal_relu(thermal)

            rgb = rgb + thermal

            rgb = self.encoder_rgb_maxpool(rgb)
            thermal = self.encoder_thermal_maxpool(thermal)

            rgb = self.encoder_rgb_layer1(rgb)
            thermal = self.encoder_thermal_layer1(thermal)

            rgb = rgb + thermal

            rgb = self.encoder_rgb_layer2(rgb)
            thermal = self.encoder_thermal_layer2(thermal)

            rgb = rgb + thermal

            rgb = self.encoder_rgb_layer3(rgb)
            thermal = self.encoder_thermal_layer3(thermal)

            rgb = rgb + thermal

            rgb = self.encoder_rgb_layer4(rgb)
            thermal = self.encoder_thermal_layer4(thermal)

            fuse = rgb + thermal

            # Global Average Pooling
            fuse = nn.AdaptiveAvgPool2d((1, 1))(fuse)
            fuse = fuse.view(fuse.size(0), -1)  # Flatten

            features.append(fuse)

        # Stack the features along the time dimension
        features = torch.stack(features, dim=1)  # shape (batch, frames, features)

        # Pack the sequences
        packed_features = rnn_utils.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)

        # packed_lstm_out, _ = self.lstm(packed_features)
        # lstm_out, _ = rnn_utils.pad_packed_sequence(packed_lstm_out, batch_first=True)
        # output = self.classifier(lstm_out[:, -1, :])

        packed_lstm_out, (hn, _) = self.lstm(packed_features)
        final_output = hn[-1]  # Take the last hidden state
        output = self.classifier(final_output)  # Pass through the classifier

        return output


def unit_test():
    device_ids = [0, 1]  # Use GPU 0 and GPU 1
    num_minibatch = 2
    num_frames = 5  # Set a number of frames per sequence

    # Create dummy data for RGB and Thermal images
    rgb = torch.randn(num_minibatch, num_frames, 3, 480, 640, dtype=torch.float32)
    thermal = torch.randn(num_minibatch, num_frames, 1, 480, 640, dtype=torch.float32)

    # Move input data to the first device (GPU 0)
    rgb = rgb.cuda(device_ids[0])
    thermal = thermal.cuda(device_ids[0])

    # Initialize the model
    rtf_net = RTFNet(n_class=3).to(device_ids[0])  # Adjust `n_class` based on your use case

    # Wrap the model with DataParallel and move it to the first GPU
    rtf_net = nn.DataParallel(rtf_net, device_ids=device_ids).cuda(device_ids[0])

    # Provide lengths for the sequences, here they are all 5 since num_frames=5
    lengths = [num_frames] * num_minibatch
    lengths = torch.tensor(lengths, dtype=torch.int64).cpu()  # Ensure lengths is a CPU tensor with int64 type

    # Perform forward pass
    try:
        output = rtf_net(rgb, thermal, lengths)
        print(output)
        print(output.shape)  # Should print the shape of the output
    except RuntimeError as e:
        print("RuntimeError:", e)
        print("CUDA_LAUNCH_BLOCKING set to 1 for debugging")
        raise e


if __name__ == '__main__':
    unit_test()

