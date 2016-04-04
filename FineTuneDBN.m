function [dbns] = FineTuneDBN(dbn,ds,nn_opts)
global batch_size;
global test_interval;
assert(mod(nn_opts.epochs,test_interval) == 0)

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
nn = dbnunfoldtonn(dbn, size(ds.train_y,2));
nn.activation_function = 'sigm';
nn.learningRate = nn_opts.learning_rate;
nn.trained_epochs = 0;


for nn_index =  1:nn_opts.epochs / test_interval
	%train nn
	nn = nntrain(nn, ds.train_x, ds.train_y, opts);
	nn.trained_epochs = nn.trained_epochs + opts.numepochs;
	[er, bad] = nntest(nn, ds.test_x, ds.test_y);
    dbn=struct;
    dbn.trained_epochs = nn.trained_epochs;
    dbn.final_er = er;
    dbn.size = nn.size;
    dbn.activation_function = nn.activation_function;
    dbn.learningRate = nn.learningRate;
    dbn.momentum = nn.momentum;
    dbn.output = nn.output;
    disp(er);
	if er<bestErr
		bestErr = er;
		bestDBN_index = nn_index;
	end
	nn.final_er = er;
	dbns = [dbns, dbn];
	opts_list = [opts_list, opts];
end


end
