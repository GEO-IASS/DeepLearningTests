
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
% rand('state',0)
% nn = nnsetup([784 100 10]);
% opts.numepochs =  1;   %  Number of full sweeps through data
% opts.batchsize = 100;  %  Take a mean gradient step over this many samples
% [nn, L] = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% 
% assert(er < 0.08, 'Too big error');
% 
% %% ex2 neural net with L2 weight decay
% rand('state',0)
% nn = nnsetup([784 100 10]);
% 
% nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% opts.numepochs =  1;        %  Number of full sweeps through data
% opts.batchsize = 100;       %  Take a mean gradient step over this many samples
% 
% nn = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% assert(er < 0.1, 'Too big error');
% 
% 
% %% ex3 neural net with dropout
% rand('state',0)
% nn = nnsetup([784 100 10]);
% 
% nn.dropoutFraction = 0.5;   %  Dropout fraction 
% opts.numepochs =  1;        %  Number of full sweeps through data
% opts.batchsize = 100;       %  Take a mean gradient step over this many samples
% 
% nn = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% assert(er < 0.1, 'Too big error');
% 
% %% ex4 neural net with sigmoid activation function
% rand('state',0)
% nn = nnsetup([784 100 10]);
% 
% nn.activation_function = 'sigm';    %  Sigmoid activation function
% nn.learningRate = 1;                %  Sigm require a lower learning rate
% opts.numepochs =  1;                %  Number of full sweeps through data
% opts.batchsize = 100;               %  Take a mean gradient step over this many samples
% 
% nn = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% assert(er < 0.1, 'Too big error');
% 
% %% ex5 plotting functionality
% rand('state',0)
% nn = nnsetup([784 20 10]);
% opts.numepochs         = 5;            %  Number of full sweeps through data
% nn.output              = 'softmax';    %  use softmax output
% opts.batchsize         = 1000;         %  Take a mean gradient step over this many samples
% opts.plot              = 1;            %  enable plotting
% 
% nn = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% assert(er < 0.1, 'Too big error');

%% ex6 neural net with sigmoid activation and plotting of validation and training error
% split training data into training and validation data
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);

rand('state',0)
bestErr = 1;
nns = [];
opts_list = [];
bestNN = 1;
for nn_index = 1:5
nn                      = nnsetup([784 500 150 10]);     
nn.output               = 'softmax';                   %  use softmax output
opts.numepochs          = 100*nn_index;                %  Number of full sweeps through data
opts.batchsize          = 100;                        %  Take a mean gradient step over this many samples
opts.plot               = 0;                           %  enable plotting
nn = nntrain(nn, tx, ty, opts, vx, vy);                %  nntrain takes validation set as last two arguments (optionally)

[er, bad] = nntest(nn, test_x, test_y);
if er<bestErr
	bestErr = er;
	bestNN = nn_index;
end
nn.finalErr = er;
nns = [nns, nn];
opts_list = [opts_list, opts];

end
save('NN_epochs_test.mat');

%assert(er < 0.02, 'Too big error');
