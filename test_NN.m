function [bestNN] = test_NN(ds,mlp_opts)
%% ex6 neural net with sigmoid activation and plotting of validation and training error
% split training data into training and validation data
global batch_size;
global test_interval;
assert(mod(mlp_opts.epochs,test_interval) == 0);

rand('state',0)
bestErr = 1;
nns = [];
opts_list = [];
bestNN_i = 1;
nn = nnsetup([size(ds.X,2), mlp_opts.layers, size(ds.Y,2)]);
nn.learningRate = mlp_opts.learning_rate;
nn.momentum = mlp_opts.momentum;
nn.output               = 'softmax';                   %  use softmax output
nn.activation_function = 'sigm';
opts.numepochs          = test_interval;                  %  Number of full sweeps through data
opts.batchsize          = batch_size;                  %  Take a mean gradient step over this many samples
opts.plot               = 0;                           %  enable plotting
nn.trained_epochs = 0;

for nn_index = 1:mlp_opts.epochs / test_interval
    nn = nntrain(nn, ds.train_x, ds.train_y, opts);                %  nntrain takes validation set as last two arguments (optionally)
    [er, bad] = nntest(nn, ds.test_x, ds.test_y);
    nn.trained_epochs = nn.trained_epochs + opts.numepochs;
    disp(er);
    if er<bestErr
        bestErr = er;
        bestNN_i = nn_index;
    end
    nn.finalErr = er;
    nns = [nns, nn];
    opts_list = [opts_list, opts];
end

bestNN = nns(bestNN_i);

% 
% res.nns.trained_epochs = [nns.trained_epochs];
% res.nns.final_er = [nns.finalErr];
% res.size = nns(1).size;
% res.activation_function = nns(1).activation_function;
% res.learning_rate = nns(1).learningRate;
% res.momentum = nns(1).momentum;
% res.scaling_learningRate = nns(1).scaling_learningRate;
% res.weightPenaltyL2 = nns(1).weightPenaltyL2;
% res.nonSparsityPenalty = nns(1).nonSparsityPenalty;
% res.sparsityTarget = nns(1).sparsityTarget;
% res.inputZeroMaskedFraction = nns(1).inputZeroMaskedFraction;
% res.dropoutFraction = nns(1).dropoutFraction;
% res.output = nns(1).output;
% res.testing = nns(1).testing;
% save('NN_epochs_test_5.mat', 'res', 'opts_list', 'bestErr', 'bestNN');

%assert(er < 0.02, 'Too big error');
end