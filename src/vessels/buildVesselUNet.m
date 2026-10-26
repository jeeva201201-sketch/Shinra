function lgraph = buildVesselUNet(input_size, num_classes)
% BUILDVESSELUNET Construct a Pure MATLAB-Native U-Net for Retinal Vessel Segmentation
% Defines a fully convolutional encoder-decoder architecture with skip connections,
% batch normalization, ReLU activations, 1x1 convolution segmentation head, and
% a PixelClassificationLayer.
%
% Input Size: [512, 512, 1] (Single-channel enhanced green channel)
% Classes: ["Background", "Vessel"] (2 classes)
%
% Guaranteed:
% - Zero dependency on SqueezeNet or pretrained classification networks.
% - No GlobalAveragePooling2DLayer.
% - No ClassificationOutputLayer.

    if nargin < 1 || isempty(input_size)
        input_size = [512, 512, 1];
    end
    if nargin < 2 || isempty(num_classes)
        num_classes = 2;
    end

    classes = ["Background", "Vessel"];

    % 1. Try MATLAB-Native unetLayers (Deep Learning Toolbox)
    try
        lgraph = unetLayers(input_size, num_classes, ...
            'EncoderDepth', 3, ...
            'NumFirstEncoderFilters', 16, ...
            'FilterSize', 3);
        
        % Ensure the terminal pixel classification layer explicitly specifies class names
        output_layer = pixelClassificationLayer('Classes', classes, 'Name', 'output');
        lgraph = replaceLayer(lgraph, lgraph.Layers(end).Name, output_layer);
    catch
        % 2. Pure Manual DAG LayerGraph with Full Skip Connections
        layers = [
            imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'none')
            
            % Encoder Level 1 (512x512)
            convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'enc1_conv1')
            batchNormalizationLayer('Name', 'enc1_bn1')
            reluLayer('Name', 'enc1_relu1')
            convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'enc1_conv2')
            batchNormalizationLayer('Name', 'enc1_bn2')
            reluLayer('Name', 'enc1_relu2')
            maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc1_pool')
            
            % Encoder Level 2 (256x256)
            convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'enc2_conv1')
            batchNormalizationLayer('Name', 'enc2_bn1')
            reluLayer('Name', 'enc2_relu1')
            convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'enc2_conv2')
            batchNormalizationLayer('Name', 'enc2_bn2')
            reluLayer('Name', 'enc2_relu2')
            maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc2_pool')
            
            % Bottleneck Level 3 (128x128)
            convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'neck_conv1')
            batchNormalizationLayer('Name', 'neck_bn1')
            reluLayer('Name', 'neck_relu1')
            transposedConv2dLayer(2, 32, 'Stride', 2, 'Name', 'dec2_up')
        ];
        
        lgraph = layerGraph(layers);
        
        % Decoder Level 2 (256x256) with Skip Concatenation from enc2_relu2
        dec2_layers = [
            depthConcatenationLayer(2, 'Name', 'dec2_concat')
            convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'dec2_conv1')
            batchNormalizationLayer('Name', 'dec2_bn1')
            reluLayer('Name', 'dec2_relu1')
            transposedConv2dLayer(2, 16, 'Stride', 2, 'Name', 'dec1_up')
        ];
        lgraph = addLayers(lgraph, dec2_layers);
        lgraph = connectLayers(lgraph, 'dec2_up', 'dec2_concat/in1');
        lgraph = connectLayers(lgraph, 'enc2_relu2', 'dec2_concat/in2');
        
        % Decoder Level 1 (512x512) with Skip Concatenation from enc1_relu2
        dec1_layers = [
            depthConcatenationLayer(2, 'Name', 'dec1_concat')
            convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'dec1_conv1')
            batchNormalizationLayer('Name', 'dec1_bn1')
            reluLayer('Name', 'dec1_relu1')
            convolution2dLayer(1, num_classes, 'Padding', 'same', 'Name', 'final_conv')
            softmaxLayer('Name', 'softmax')
            pixelClassificationLayer('Classes', classes, 'Name', 'output')
        ];
        lgraph = addLayers(lgraph, dec1_layers);
        lgraph = connectLayers(lgraph, 'dec1_up', 'dec1_concat/in1');
        lgraph = connectLayers(lgraph, 'enc1_relu2', 'dec1_concat/in2');
    end
end
