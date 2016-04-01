function [bestDBN] = FineTuneDBN(dbn,ds,nn_opts)
global batch_size;
global test_interval;
assert(mod(nn_opts.epochs,test_interval) == 0)
%load('DBN_500-150_pre200_batch100_m0.1_a1.mat');

%train_x = double(train_x) / 255;
%test_x  = double(test_x)  / 255;
%train_y = double(train_y);
%test_y  = double(test_y);

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
bestErr = 1; % 100% error before first
dbns = [];
opts_list = [];
bestDBN_index = 1;


nn = struct;
opts = struct;
opts.momentum  =  nn_opts.momentum;
opts.alpha     =  1;
opts.batchsize = batch_size;
opts.numepochs =  test_interval;
%unfold dbn to nn
nn = dbnunfoldtonn(dbn, size(ds.Y,2));
nn.activation_function = 'sigm';
nn.learningRate = nn_opts.learning_rate;
nn.trained_epochs = 0;


for nn_index =  1:nn_opts.epochs / test_interval
	%train nn
	nn = nntrain(nn, ds.train_x, ds.train_y, opts);
	nn.trained_epochs = nn.trained_epochs + opts.numepochs;
	[er, bad] = nntest(nn, ds.test_x, ds.test_y);

    disp(er);
	if er<bestErr
		bestErr = er;
		bestDBN_index = nn_index;
	end
	nn.final_er = er;
	dbns = [dbns, nn];
	opts_list = [opts_list, opts];
end

bestDBN = dbns(bestDBN_index);

% res.dbns.trained_epochs = [dbns.trained_epochs];
% res.dbns.final_er = [dbns.final_er];
% res.size = dbns(1).size;
% res.activation_function = dbns(1).activation_function;
% res.learning_rate = dbns(1).learningRate;
% res.momentum = dbns(1).momentum;
% res.scaling_learningRate = dbns(1).scaling_learningRate;
% res.weightPenaltyL2 = dbns(1).weightPenaltyL2;
% res.nonSparsityPenalty = dbns(1).nonSparsityPenalty;
% res.sparsityTarget = dbns(1).sparsityTarget;
% res.inputZeroMaskedFraction = dbns(1).inputZeroMaskedFraction;
% res.dropoutFraction = dbns(1).dropoutFraction;
% res.output = dbns(1).output;
% res.testing = dbns(1).testing;
% %save('Results/DBN_test_3_batch100_pre200_lr0.5_mom0.5.mat', 'res', 'opts_list', 'bestErr', 'bestDBN');
% plot(res.dbns.trained_epochs,res.dbns.final_er);
% %assert(er < 0.10, 'Too big error');
end
