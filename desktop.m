% use this file to do all the messy work

%% initialization
load '20news_w100.mat';
global batch_size;
global test_interval;
batch_size = 20;
test_interval = 5;
gd_epochs = 400;
Y = softmaxifyY(newsgroups');
X = double(full(documents'));
ds = createDataSet(X,Y,1,length(X)/400);
gd_opts = struct;
gd_opts.layers = [100];
gd_opts.epochs = 400;

%% train MLP
gd_opts.momentum = 1/8;
gd_opts.learning_rate = .2;

mlp = test_NN(ds,gd_opts);
mlp_final_err = mlp.finalErr;
%% pre train DBN
dbn_opts = struct;
dbn_opts.momentum = 1/2;
dbn_opts.learning_rate = 1;
dbn_opts.epochs = 50;

dbn_opts.layers = gd_opts.layers;
dbn = test_DBN(ds,dbn_opts);
%% fine tune DBN
gd_opts.momentum = .1;
gd_opts.learning_rate = .1;

nn = FineTuneDBN(dbn,ds,gd_opts);
dbn_final_err = nn.final_er;