classdef plotInfo
    %PLOTINFO Contains the information needed to create a plot
    %   Detailed explanation goes here
    
    properties(Access = public)
        epochs
        val_ers
        train_ers
        test_er
        val_er
        hidden_layers
        learning_rate
        momentum
        label
    end
    
    methods
        function obj = plotInfo(network,label)
            if label == 0.006
                label = 0.005;
            end
            obj.label = label;
            obj.epochs = network.epochs;
            obj.val_ers = network.val_ers;
            obj.train_ers = network.train_ers;
            obj.test_er = network.nn.test_er;
            obj.val_er = network.nn.val_er;
            obj.hidden_layers = network.nn.size(2:end-1);
            obj.learning_rate = network.nn.learningRate;
            obj.momentum = network.nn.momentum;
        end
    end
    
end

