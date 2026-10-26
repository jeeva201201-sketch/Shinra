function [lgraph, net_base] = buildSqueezeNetSeverityV3(num_classes)
% BUILDSQUEEZENETSEVERITYV3 SqueezeNet V3 Transfer Learning Architecture
% Implements:
% 1. ImageNet-pretrained SqueezeNet backbone.
% 2. Frozen low/mid-level feature layers (conv1 through fire7).
% 3. Fine-tuned high-level feature layers (fire8 and fire9, LearnRateFactor = 1).
% 4. Direct logit path to global average pooling: drop9 -> new_conv10 -> pool10 -> prob -> output.
%    (Explicitly removes relu_conv10 to eliminate dead-ReLU gradient blockage).
% 5. New 5-class convolution head (new_conv10 with LearnRateFactor = 10).
%
% Input dimensions: 227 x 227 x 3 (native SqueezeNet resolution)

    if nargin < 1 || isempty(num_classes)
        num_classes = 5;
    end

    % 1. Load Pretrained SqueezeNet
    try
        net_base = squeezenet();
    catch ME
        error('SHINRA:SqueezeNetUnavailable', ...
            'Failed to load pretrained SqueezeNet: %s\nEnsure Deep Learning Toolbox Model for SqueezeNet is available.', ...
            ME.message);
    end

    lgraph = layerGraph(net_base);

    % 2. Freeze conv1 through fire7 (Layers 1 to fire7)
    % Keep fire8 and fire9 trainable with base learn rate factor = 1
    layers = lgraph.Layers;
    for i = 1:numel(layers)
        layer_name = layers(i).Name;
        if isa(layers(i), 'nnet.cnn.layer.Convolution2DLayer')
            % Check if layer belongs to conv1 through fire7
            is_frozen = strcmp(layer_name, 'conv1') || ...
                        startsWith(layer_name, 'fire2_') || ...
                        startsWith(layer_name, 'fire3_') || ...
                        startsWith(layer_name, 'fire4_') || ...
                        startsWith(layer_name, 'fire5_') || ...
                        startsWith(layer_name, 'fire6_') || ...
                        startsWith(layer_name, 'fire7_');
            
            is_finetuned = startsWith(layer_name, 'fire8_') || ...
                           startsWith(layer_name, 'fire9_');

            if is_frozen
                conv_layer = layers(i);
                conv_layer.WeightLearnRateFactor = 0;
                conv_layer.BiasLearnRateFactor = 0;
                lgraph = replaceLayer(lgraph, layer_name, conv_layer);
            elseif is_finetuned
                conv_layer = layers(i);
                conv_layer.WeightLearnRateFactor = 1;
                conv_layer.BiasLearnRateFactor = 1;
                lgraph = replaceLayer(lgraph, layer_name, conv_layer);
            end
        end
    end

    % 3. Construct New 5-Class Classification Convolution Layer
    new_conv = convolution2dLayer([1, 1], num_classes, ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10, ...
        'WeightL2Factor', 1, ...
        'BiasL2Factor', 0, ...
        'Name', 'new_conv10');

    new_class_output = classificationLayer('Name', 'new_classoutput');

    % 4. Rewire Head: drop9 -> new_conv10 -> pool10 -> prob -> new_classoutput
    % Remove original conv10, relu_conv10, and ClassificationLayer_predictions
    lgraph = removeLayers(lgraph, {'conv10', 'relu_conv10', 'ClassificationLayer_predictions'});

    % Add new classification layers
    lgraph = addLayers(lgraph, new_conv);
    lgraph = addLayers(lgraph, new_class_output);

    % Connect new_conv10 directly between drop9 and pool10 (bypassing relu_conv10)
    lgraph = connectLayers(lgraph, 'drop9', 'new_conv10');
    lgraph = connectLayers(lgraph, 'new_conv10', 'pool10');
    lgraph = connectLayers(lgraph, 'prob', 'new_classoutput');
end
