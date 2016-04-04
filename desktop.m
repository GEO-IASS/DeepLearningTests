% use this file to do all the messy work

pre_loaded_dbn = 1; %set to 1 if the pre training of dbn is already done

%% initialization
load 'mnist_uint8.mat';
global batch_size;
global test_interval;
batch_size = 50;
test_interval = 2;
%Y = softmaxifyY(newsgroups');

%X = double(full(documents'));
%ds = createDataSet(X,Y,1,length(X)/400);
ds.test_x = double(test_x) / 255;
ds.test_y = double(test_y);
ds.train_x = double(train_x(1:100,:)) / 255;
ds.train_y = double(train_y(1:100,:));
ds.train_x_unlabled = double(train_x) / 255;
gd_opts = struct;
gd_opts.layers = [2500 2000 1500 1000 500];
gd_opts.epochs = 350;

%% train MLP
gd_opts.momentum = 1/2;
gd_opts.learning_rate = .01;

nns = test_NN(ds,gd_opts);
save_results(nns, gd_opts, 'Results/NN_9.mat');
%mlp_final_err = mlp.finalErr;
%% pre train DBN
dbn_opts = struct;
dbn_opts.momentum = 1/2;
dbn_opts.learning_rate = 2;
dbn_opts.epochs = 100;
if(pre_loaded_dbn==0)
    dbn_opts.layers = gd_opts.layers;
    dbn = test_DBN(ds,dbn_opts);
end
%% fine tune DBN
gd_opts.momentum = .5;
gd_opts.learning_rate = 2;

dbns = FineTuneDBN(dbn,ds,gd_opts);
%dbn_final_err = nn.final_er;
save_results(dbns, struct, 'Results/DBN_9.mat');

%% Plot
plot_res(nns,dbns);