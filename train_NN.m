function [res] = train_NN(ds,mlp_opts,initial_nn)

%%Setup
global batch_size;
global test_interval;
assert(mod(mlp_opts.epochs,test_interval) == 0);

rand('state',0)
bestErr = 1;
switch nargin
    case 2
        nn = nnsetup([size(ds.train_x,2), mlp_opts.layers, size(ds.train_y,2)]);
    case 3
        nn = initial_nn;
end
nn.learningRate = mlp_opts.learning_rate;
nn.momentum = mlp_opts.momentum;
nn.output               = 'softmax';%'sigm';%                   %  use softmax output
nn.activation_function  = 'sigm';                       
opts.numepochs          = test_interval;                  %  Number of full sweeps through data
opts.batchsize          = batch_size;                  %  Take a mean gradient step over this many samples
opts.plot               = 0;                       %  enable plotting
opts.alpha     =  1;
nn.trained_epochs = 0;
epochs_since_best_nn = 0;
best_nn = struct;
best_nn.val_er = 1;
epochs = [];
val_ers = [];
train_ers = [];
early_stop_epochs = 10;

%% Algorithm
for nn_index = 1:mlp_opts.epochs / test_interval
    nn = nntrain(nn, ds.train_x, ds.train_y, opts);                %  nntrain takes validation set as last two arguments (optionally)
    [val_er, ~] = nntest(nn, ds.val_x, ds.val_y);
    [train_er,~] = nntest(nn, ds.train_x, ds.train_y);
    val_ers = [val_er,val_ers];
    train_ers = [train_er,train_ers];
    epoch = nn_index * test_interval;
    fprintf('epoch = %d\n',epoch);
    epochs = [epoch,epochs];
    nn.trained_epochs = nn.trained_epochs + opts.numepochs;
    fprintf('val_er = %g \ntrain_er = %g\n',val_er,train_er);
    if val_er < best_nn.val_er
        disp('New best mlp');
        best_nn = nn;
        best_nn.val_er = val_er;
        epochs_since_best_nn = 0;
    elseif epochs_since_best_nn <= early_stop_epochs
        fprintf('epochs since best_nn = %d\n',epochs_since_best_nn);
       epochs_since_best_nn = epochs_since_best_nn + test_interval;
    else
        fprintf('%d epochs of no better results, stoping early\n',epochs_since_best_nn);
        break;
    end
end

%% Results
[test_er,~] = nntest(best_nn,ds.test_x,ds.test_y);
best_nn.test_er = test_er;
res = struct;
res.nn = best_nn;
res.epochs = epochs;
res.val_ers = val_ers;
res.train_ers = train_ers;
end