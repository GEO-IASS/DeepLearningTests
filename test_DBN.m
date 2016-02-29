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
	opts.numepochs =   20;
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
	nn.trained_epochs = 0;

for nn_index =  1:5
	%train nn
	opts.numepochs =  100;
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
save('DBN_test.mat');

%assert(er < 0.10, 'Too big error');