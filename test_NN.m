
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
%[train_x, mu, sigma] = zscore(train_x);
%test_x = normalize(test_x, mu, sigma);


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
nn                      = nnsetup([784 500 150 10]);
nn.learningRate = 0.1;
%nn.momentum = 0;
%nn.output               = 'softmax';                   %  use softmax output
nn.activation_function = 'sigm';
opts.numepochs          = 5;                %  Number of full sweeps through data
opts.batchsize          = 100;                        %  Take a mean gradient step over this many samples
opts.plot               = 0;                           %  enable plotting
nn.trained_epochs = 0;
for nn_index = 1:75

nn = nntrain(nn, train_x, train_y, opts);                %  nntrain takes validation set as last two arguments (optionally)
[er, bad] = nntest(nn, test_x, test_y);
nn.trained_epochs = nn.trained_epochs + opts.numepochs;

if er<bestErr
	bestErr = er;
	bestNN = nn_index;
end
nn.finalErr = er;
nns = [nns, nn];
opts_list = [opts_list, opts];

end


res.nns.trained_epochs = [nns.trained_epochs];
res.nns.final_er = [nns.finalErr];
res.size = nns(1).size;
res.activation_function = nns(1).activation_function;
res.learning_rate = nns(1).learningRate;
res.momentum = nns(1).momentum;
res.scaling_learningRate = nns(1).scaling_learningRate;
res.weightPenaltyL2 = nns(1).weightPenaltyL2;
res.nonSparsityPenalty = nns(1).nonSparsityPenalty;
res.sparsityTarget = nns(1).sparsityTarget;
res.inputZeroMaskedFraction = nns(1).inputZeroMaskedFraction;
res.dropoutFraction = nns(1).dropoutFraction;
res.output = nns(1).output;
res.testing = nns(1).testing;
save('NN_epochs_test_5.mat', 'res', 'opts_list', 'bestErr', 'bestNN');

%assert(er < 0.02, 'Too big error');
