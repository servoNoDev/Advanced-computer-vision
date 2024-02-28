from torch import nn
import torch

from torch import nn
import torch

class UNet(nn.Module):
    # Encoder Blocks:
    # The encoder consists of four convolutional blocks (conv_block) which decrease the spatial dimensions while increasing the number of channels. Each block performs two convolutional operations followed by ReLU activation.

    # Middle (Bottleneck) Layer:
    # The middle layer is a deconvolutional block (deconv_block) which increases the spatial dimensions while reducing the number of channels.

    # Decoder Blocks:
    # The decoder consists of four deconvolutional blocks (deconv_block) which increase the spatial dimensions while reducing the number of channels. Each block also concatenates feature maps from the corresponding encoder layer before performing the deconvolution operation.

    # Last Deconvolutional Block:
    # The final deconvolutional block (last_deconv_block) is similar to the other deconvolutional blocks but ends with a convolutional layer with a sigmoid activation function to output the segmentation mask.

    # Forward Method:
    # In the forward pass, the input passes through the encoder to extract features. Then, the features are passed through the middle layer. Next, the decoder receives the features along with the corresponding features from the encoder (skip connections) to reconstruct the original spatial dimensions. Finally, the output is produced by the last deconvolutional block.

    # This architecture is effective for semantic segmentation tasks, where the goal is to classify each pixel in the input image. The skip connections help to preserve spatial information and improve segmentation accuracy.
    def __init__(self):
        super(UNet, self).__init__()
        print("new model - stupido")
        # Encoder blocks
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Middle (bottleneck) layer
        self.middle = self.deconv_block(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Decoder blocks
        self.decoder4 = self.deconv_block(1024, 512)
        self.decoder3 = self.deconv_block(512, 256)
        self.decoder2 = self.deconv_block(256, 128)
        self.decoder1 = self.last_deconv_block(128, 64)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    # def deconv_block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2)
    #     )
    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Adjusted channels here
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)  # Adjusted channels here
        )

    def last_deconv_block(self, in_channels, out_channels):
        """
        Final deconvolutional block consisting of two convolutional layers with ReLU activation,
        followed by a convolutional layer with sigmoid activation to output segmentation mask.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        # Middle (bottleneck) layer
        middle = self.middle(self.maxpool(enc4))

        # Decoder
        dec4 = self.decoder4(middle)
        dec3 = self.decoder3(dec4)
        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)

        return dec1


class UNetSkiped(nn.Module):
    # Encoder Blocks:
    # The encoder consists of four convolutional blocks (conv_block) which decrease the spatial dimensions while increasing the number of channels. Each block performs two convolutional operations followed by ReLU activation.

    # Middle (Bottleneck) Layer:
    # The middle layer is a deconvolutional block (deconv_block) which increases the spatial dimensions while reducing the number of channels.

    # Decoder Blocks:
    # The decoder consists of four deconvolutional blocks (deconv_block) which increase the spatial dimensions while reducing the number of channels. Each block also concatenates feature maps from the corresponding encoder layer before performing the deconvolution operation.

    # Last Deconvolutional Block:
    # The final deconvolutional block (last_deconv_block) is similar to the other deconvolutional blocks but ends with a convolutional layer with a sigmoid activation function to output the segmentation mask.

    # Forward Method:
    # In the forward pass, the input passes through the encoder to extract features. Then, the features are passed through the middle layer. Next, the decoder receives the features along with the corresponding features from the encoder (skip connections) to reconstruct the original spatial dimensions. Finally, the output is produced by the last deconvolutional block.

    # This architecture is effective for semantic segmentation tasks, where the goal is to classify each pixel in the input image. The skip connections help to preserve spatial information and improve segmentation accuracy.
    def __init__(self):
        super(UNetSkiped, self).__init__()
        
        # Encoder blocks
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Middle (bottleneck)
        self.middle = self.deconv_block(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Decoder blocks
        self.decoder4 = self.deconv_block(1024, 512)
        self.decoder3 = self.deconv_block(512, 256)
        self.decoder2 = self.deconv_block(256, 128)
        self.decoder1 = self.last_deconv_block(128, 64)

    def conv_block(self, in_channels, out_channels):
        """
        Convolutional block consisting of two convolutional layers with ReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True)
        )

    def deconv_block(self, in_channels, out_channels):
        """
        Deconvolutional block consisting of two convolutional layers with ReLU activation
        followed by a transposed convolutional layer.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2)
        )

   

    def last_deconv_block(self, in_channels, out_channels):
        """
        Final deconvolutional block consisting of two convolutional layers with ReLU activation,
        followed by a convolutional layer with sigmoid activation to output segmentation mask.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        # Middle (bottleneck)
        middle = self.middle(self.maxpool(enc4))

        # Decoder with skip connections
        dec4 = torch.cat([enc4, middle], dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)

        dec3 = torch.cat([enc3, dec4], dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)

        dec2 = torch.cat([enc2, dec3], dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)

        dec1 = torch.cat([enc1, dec2], dim=1)  # Skip connection
        output = self.decoder1(dec1)

        return output

# Batch normalization is a technique used to improve the training of artificial neural networks by normalizing the inputs of each layer in a mini-batch.
# It was introduced to address the issue of internal covariate shift, which refers to the change in the distribution of network activations due to parameter updates during training.
# Here's how batch normalization works:
# Normalization: For each mini-batch during training, batch normalization normalizes the activations of each layer by subtracting the mean and dividing by the standard deviation of the mini-batch. This is akin to standardizing the inputs to have zero mean and unit variance.
# Scaling and Shifting: After normalization, batch normalization applies two additional learnable parameters: a scaling parameter (gamma) and a shifting parameter (beta). These parameters allow the model to adaptively scale and shift the normalized activations, enabling the network to learn the optimal representation for each layer.
# Stabilizing Training: By normalizing the inputs to each layer, batch normalization helps stabilize the training process. It reduces the internal covariate shift, making the optimization landscape smoother and enabling higher learning rates. This leads to faster convergence and improved generalization performance.


class UNetSkipedBatch(nn.Module):
    def __init__(self):
        super(UNetSkipedBatch, self).__init__()
        
        # Encoder blocks
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Middle (bottleneck)
        self.middle = self.deconv_block(512, 1024)
        

        # Decoder blocks
        self.decoder4 = self.deconv_block(1024, 512)
        self.decoder3 = self.deconv_block(512, 256)
        self.decoder2 = self.deconv_block(256, 128)
        self.decoder1 = self.last_deconv_block(128, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def conv_block(self, in_channels, out_channels):
        """
        Convolutional block consisting of two convolutional layers with ReLU activation
        and batch normalization.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def deconv_block(self, in_channels, out_channels):
        """
        Deconvolutional block consisting of two convolutional layers with ReLU activation,
        batch normalization, followed by a transposed convolutional layer.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2)
        )

    def last_deconv_block(self, in_channels, out_channels):
        """
        Final deconvolutional block consisting of two convolutional layers with ReLU activation,
        batch normalization, followed by a convolutional layer with sigmoid activation to output segmentation mask.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        # Middle (bottleneck)
        middle = self.middle(self.maxpool(enc4))

        # Decoder with skip connections
        dec4 = torch.cat([enc4, middle], dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)

        dec3 = torch.cat([enc3, dec4], dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)

        dec2 = torch.cat([enc2, dec3], dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)

        dec1 = torch.cat([enc1, dec2], dim=1)  # Skip connection
        output = self.decoder1(dec1)

        return output



class UNetPlusPlusFake(nn.Module):
    def __init__(self):
        super(UNetPlusPlusFake, self).__init__()
        
        # Encoder blocks
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Middle (bottleneck)
        self.middle = self.deconv_block(512, 1024)
        
        # Decoder blocks
        self.decoder4 = self.deconv_block(1024, 512)
        self.decoder3 = self.deconv_block(512, 256)
        self.decoder2 = self.deconv_block(256, 128)
        self.decoder1 = self.last_deconv_block(128, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def conv_block(self, in_channels, out_channels):
        """
        Convolutional block consisting of two convolutional layers with ReLU activation
        and batch normalization.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def deconv_block(self, in_channels, out_channels):
        """
        Deconvolutional block consisting of two convolutional layers with ReLU activation,
        batch normalization, followed by a transposed convolutional layer.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2)
        )

    def last_deconv_block(self, in_channels, out_channels):
        """
        Final deconvolutional block consisting of two convolutional layers with ReLU activation,
        batch normalization, followed by a convolutional layer with sigmoid activation to output segmentation mask.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        # Middle (bottleneck)
        middle = self.middle(self.maxpool(enc4))

        # Decoder with skip connections and feature supervision
        dec4 = torch.cat([enc4, middle], dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)
        
        # Supervision 1
        feat1 = dec4
        
        dec3 = torch.cat([enc3, dec4], dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)
        
        # Supervision 2
        feat2 = dec3
        
        dec2 = torch.cat([enc2, dec3], dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)
        
        # Supervision 3
        feat3 = dec2
        
        dec1 = torch.cat([enc1, dec2], dim=1)  # Skip connection
        output = self.decoder1(dec1)
        
        # Supervision 4 (output)
        feat4 = output

        return output, [feat1, feat2, feat3, feat4]  # Return segmentation output and intermediate features
