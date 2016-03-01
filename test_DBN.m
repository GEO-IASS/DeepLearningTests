load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
%rand('state',0)
%dbn.sizes = [100];
%opts.numepochs =   1;
%opts.batchsize = 100;
%opts.momentum  =   0;
%opts.alpha     =   1;
%dbn = dbnsetup(dbn, train_x, opts);
%dbn = dbntrain(dbn, train_x, opts);
%figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
bestErr = 1; % 100% error before first
dbns = [];
opts_list = [];
bestNN = 1;

	dbn = struct;
	nn = struct;
	opts = struct;
	%rng(0, 'twister');
	%n_layers = randi([1,6],1,1);
	%train dbn
	dbn.sizes = [500 150];
	opts.numepochs =   30;
	%dbn.sizes = randi([100,1000],1,n_layers);
	%opts.numepochs = randi([3,5], 1,1);

	opts.batchsize = 100;
	opts.momentum  =  0.5;
	opts.alpha     =  1;
	dbn = dbnsetup(dbn, train_x, opts);
	dbn = dbntrain(dbn, train_x, opts);

	%unfold dbn to nn
	nn = dbnunfoldtonn(dbn, 10);
	nn.activation_function = 'sigm';
    nn.learningRate = 0.1;
	nn.trained_epochs = 0;

for nn_index =  1:75
	%train nn
	opts.numepochs =  5;
	opts.batchsize = 100;
	nn = nntrain(nn, train_x, train_y, opts);
	nn.trained_epochs = nn.trained_epochs + opts.numepochs;
	[er, bad] = nntest(nn, test_x, test_y);


	if er<bestErr
		bestErr = er;
		bestNN = nn_index;
	end
	nn.final_er = er;
	dbns = [dbns, nn];
	opts_list = [opts_list, opts];
end

res.dbns.trained_epochs = [dbns.trained_epochs];
res.dbns.final_er = [dbns.final_er];
res.size = dbns(1).size;
res.activation_function = dbns(1).activation_function;
res.learning_rate = dbns(1).learningRate;
res.momentum = dbns(1).momentum;
res.scaling_learningRate = dbns(1).scaling_learningRate;
res.weightPenaltyL2 = dbns(1).weightPenaltyL2;
res.nonSparsityPenalty = dbns(1).nonSparsityPenalty;
res.sparsityTarget = dbns(1).sparsityTarget;
res.inputZeroMaskedFraction = dbns(1).inputZeroMaskedFraction;
res.dropoutFraction = dbns(1).dropoutFraction;
res.output = dbns(1).output;
res.testing = dbns(1).testing;
save('DBN_epochs_test_5.mat', 'res', 'opts_list', 'bestErr', 'bestNN');

%assert(er < 0.10, 'Too big error');
