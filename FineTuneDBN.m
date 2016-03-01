load mnist_uint8;

load('DBN_500-150_pre200_batch100_m0.1_a1.mat');

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);


%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
bestErr = 1; % 100% error before first
dbns = [];
opts_list = [];
bestDBN = 1;


nn = struct;
opts = struct;
opts.momentum  =  0.5;
opts.alpha     =  1;    
opts.batchsize = 100;
opts.numepochs =  5;
%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';
nn.learningRate = 0.5;
nn.trained_epochs = 0;


for nn_index =  1:75
	%train nn
	nn = nntrain(nn, train_x, train_y, opts);
	nn.trained_epochs = nn.trained_epochs + opts.numepochs;
	[er, bad] = nntest(nn, test_x, test_y);


	if er<bestErr
		bestErr = er;
		bestDBN = nn_index;
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
%save('Results/DBN_test_3_batch100_pre200_lr0.5_mom0.5.mat', 'res', 'opts_list', 'bestErr', 'bestDBN');

%assert(er < 0.10, 'Too big error');
