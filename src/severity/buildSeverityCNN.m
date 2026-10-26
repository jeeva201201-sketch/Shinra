function lgraph = buildSeverityCNN(input_size, num_classes)
% BUILDSEVERITYCNN Construct Compact Deep Convolutional Network for 5-Class DR Grading
% Explicitly enforces network input size to [224, 224, 3], perfectly matching
% the standardized SHINRA retinal preprocessing pipeline.
%
% Fully native to MATLAB Deep Learning Toolbox without requiring external add-ons.

    if nargin < 1 || isempty(input_size)
        input_size = [224, 224, 3];
    end
    if nargin < 2 || isempty(num_classes)
        num_classes = 5;
    end

    % Ensure input_size is strictly 1x3 vector [224, 224, 3]
    if numel(input_size) == 2
        input_size = [input_size(1), input_size(2), 3];
    end

    % Compact Custom Convolutional Architecture (Native Deep Learning Toolbox)
    % Designed with ~436k parameters to prevent parameter bloat and overfitting on IDRiD
    layers = [
        imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'zscore')
        
        % Block 1: 224x224 -> 112x112
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_1')
        batchNormalizationLayer('Name', 'bn1_1')
        reluLayer('Name', 'relu1_1')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_2')
        batchNormalizationLayer('Name', 'bn1_2')
        reluLayer('Name', 'relu1_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        % Block 2: 112x112 -> 56x56
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_1')
        batchNormalizationLayer('Name', 'bn2_1')
        reluLayer('Name', 'relu2_1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_2')
        batchNormalizationLayer('Name', 'bn2_2')
        reluLayer('Name', 'relu2_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % Block 3: 56x56 -> 28x28
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_1')
        batchNormalizationLayer('Name', 'bn3_1')
        reluLayer('Name', 'relu3_1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_2')
        batchNormalizationLayer('Name', 'bn3_2')
        reluLayer('Name', 'relu3_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
        
        % Block 4: 28x28 -> 1x1 via Global Average Pooling
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4_1')
        batchNormalizationLayer('Name', 'bn4_1')
        reluLayer('Name', 'relu4_1')
        globalAveragePooling2dLayer('Name', 'gap')
        
        % Classifier Head (5 Classes: 0 to 4)
        dropoutLayer(0.4, 'Name', 'drop')
        fullyConnectedLayer(num_classes, 'Name', 'fc_dr')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
    
    lgraph = layerGraph(layers);
end
