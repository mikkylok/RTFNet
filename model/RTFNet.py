# coding:utf-8
# Original By Yuxiang Sun, Aug. 2, 2019
# Modified By Meixi Lu


import torch
import torch.nn as nn 
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights


class RTFNet(nn.Module):

    def __init__(self, n_class=3,
                 num_resnet_layers=50,
                 num_lstm_layers=1,
                 lstm_hidden_size=512,
                 attention_heads=8,  # Add multi-head attention
                 attention_dim=256,  # Dimension for attention embeddings
                 device=torch.device('cuda:0')):
        super(RTFNet, self).__init__()
        self.device = device

        self.num_resnet_layers = num_resnet_layers

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            resnet_raw_model2 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            resnet_raw_model2 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            resnet_raw_model2 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            resnet_raw_model2 = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
            resnet_raw_model2 = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
            self.inplanes = 2048

        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
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

        # Ensure this variable is set properly
        self.cross_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=attention_heads)

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear projection to match dimensions (from 256 to 2048)
        self.linear_proj = nn.Linear(self.inplanes, attention_dim)

        # LSTM module
        self.lstm = nn.LSTM(input_size=attention_dim, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers,
                            batch_first=True)
        self.lstm.flatten_parameters()

        # Classifier
        self.classifier = nn.Linear(lstm_hidden_size, n_class)

        # Move all modules to the specified device
        self.to(self.device)

    def forward(self, rgb_images, thermal_images):

        # Initialize an empty list to store features for each frame
        features = []
        all_attention_weights = []

        for t in range(rgb_images.size(1)):  # iterate over time dimension
            rgb = rgb_images[:, t]
            thermal = thermal_images[:, t]

            # RGB encoder
            rgb = self.encoder_rgb_conv1(rgb)
            rgb = self.encoder_rgb_bn1(rgb)
            rgb = self.encoder_rgb_relu(rgb)
            rgb = self.encoder_rgb_maxpool(rgb)

            # Thermal encoder
            thermal = self.encoder_thermal_conv1(thermal)
            thermal = self.encoder_thermal_bn1(thermal)
            thermal = self.encoder_thermal_relu(thermal)
            thermal = self.encoder_thermal_maxpool(thermal)

            # Layer 1
            rgb = self.encoder_rgb_layer1(rgb)
            thermal = self.encoder_thermal_layer1(thermal)

            # Layer 2
            rgb = self.encoder_rgb_layer2(rgb)
            thermal = self.encoder_thermal_layer2(thermal)

            # Layer 3
            rgb = self.encoder_rgb_layer3(rgb)
            thermal = self.encoder_thermal_layer3(thermal)

            # Layer 4
            rgb = self.encoder_rgb_layer4(rgb)
            thermal = self.encoder_thermal_layer4(thermal)

            # Apply global average pooling after Layer 4
            rgb_global = self.global_avg_pool(rgb).view(rgb.size(0), 1, -1)  # Shape: [batch_size, 1, feature_dim]
            thermal_global = self.global_avg_pool(thermal).view(thermal.size(0), 1, -1)  # Shape: [batch_size, 1, feature_dim]

            # Concatenate RGB and Thermal features along the sequence dimension for attention
            combined = torch.cat((rgb_global, thermal_global), dim=1).transpose(0, 1)  # Shape: [seq_len=2, batch_size, feature_dim]

            # Project 2048-dim features down to attention_dim=128 before attention
            combined_proj = self.linear_proj(combined)  # Shape: [seq_len=2, batch_size, attention_dim=256]

            # Apply multi-head attention (cross-modality attention)
            attended_features, attn_weights = self.cross_attention(combined_proj, combined_proj, combined_proj)

            all_attention_weights.append(attn_weights)

            # Extract RGB and thermal weighted features after attention
            rgb_weighted = attended_features[0].view(rgb.size(0), -1)  # Attended RGB features
            thermal_weighted = attended_features[1].view(thermal.size(0), -1)  # Attended Thermal features

            # Combine using weighted sum
            fuse = rgb_weighted + thermal_weighted

            features.append(fuse)

        # Stack the features along the time dimension
        features = torch.stack(features, dim=1)  # shape (batch, frames, features)

        # Pass the stacked features through the LSTM
        lstm_out, (hn, _) = self.lstm(features)
        final_output = hn[-1]  # Take the last hidden state
        output = self.classifier(final_output)

        return output, all_attention_weights, lstm_out


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
        print(output.shape)
    except RuntimeError as e:
        print("RuntimeError:", e)
        print("CUDA_LAUNCH_BLOCKING set to 1 for debugging")
        raise e


if __name__ == '__main__':
    unit_test()

