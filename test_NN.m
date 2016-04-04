function [nns] = test_NN(ds,mlp_opts)

global batch_size;
global test_interval;
assert(mod(mlp_opts.epochs,test_interval) == 0);

rand('state',0)
bestErr = 1;
nns = [];
opts_list = [];
bestNN_i = 1;
nn = nnsetup([size(ds.train_x,2), mlp_opts.layers, size(ds.train_y,2)]);
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
    mlp=struct;
    mlp.trained_epochs = nn.trained_epochs;
    mlp.final_er = er;
    mlp.size = nn.size;
    mlp.activation_function = nn.activation_function;
    mlp.learningRate = nn.learningRate;
    mlp.momentum = nn.momentum;
    mlp.output = nn.output;
    if er<bestErr
        bestErr = er;
        bestNN_i = nn_index;
    end
    nn.final_er = er;
    nns = [nns, mlp];
    opts_list = [opts_list, opts];
end

bestNN = nns(bestNN_i);

end