function [lgraph, net_base] = buildSqueezeNetSeverity(num_classes, freeze_backbone)
% BUILDSQUEEZENETSEVERITY Construct SqueezeNet Transfer-Learning Network for 5-Class DR
% Loads pretrained SqueezeNet (ImageNet), freezes the early/middle fire modules,
% and replaces the 1000-class classification head with a new 5-class DR head.
%
% Input dimensions: 227 x 227 x 3 (native SqueezeNet input resolution)

    if nargin < 1 || isempty(num_classes)
        num_classes = 5;
    end
    if nargin < 2 || isempty(freeze_backbone)
        freeze_backbone = true; % Conservative transfer learning on small dataset
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

    % 2. Freeze Backbone Layers if requested (Layers 1 through drop9)
    if freeze_backbone
        layers = lgraph.Layers;
        for i = 1:numel(layers)
            layer_name = layers(i).Name;
            % Freeze all convolutional layers except the new classification head
            if isa(layers(i), 'nnet.cnn.layer.Convolution2DLayer') && ~strcmp(layer_name, 'conv10')
                frozen_conv = layers(i);
                frozen_conv.WeightLearnRateFactor = 0;
                frozen_conv.BiasLearnRateFactor = 0;
                lgraph = replaceLayer(lgraph, layer_name, frozen_conv);
            end
        end
    end

    % 3. Replace Classification Head with 5-Class Retinal DR Head
    % Replace 1000-class conv10 with 5-class conv layer (Higher LearnRate for fast adaptation)
    new_conv = convolution2dLayer([1, 1], num_classes, ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10, ...
        'WeightL2Factor', 1, ...
        'BiasL2Factor', 0, ...
        'Name', 'new_conv10');

    new_class_output = classificationLayer('Name', 'new_classoutput');

    lgraph = replaceLayer(lgraph, 'conv10', new_conv);
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', new_class_output);
end
