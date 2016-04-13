% use this file to do all the messy work
clear;
dbn1=load('DBN_500-150_pre200_batch100_m0.1_a1.mat');
dbn2=load('DBN_hinton_100epochs.mat');
dbn3=load('DBN_deep_pre200.mat');
dbns = [dbn1.dbn, dbn2.dbn, dbn3.dbn];
pre_loaded_dbn = 1; %set to 1 if the pre training of dbn is already done
%% Load data
if not(exist('train_x') || exist('test_x'))
    load('mnist_uint8.mat');
end
ds.test_x = double(test_x) / 255;
ds.test_y = double(test_y);
ds.val_x = double(train_x(50001:60000,:)) / 255;
ds.val_y = double(train_y(50001:60000,:));
train_x = double(train_x(1:50000,:)) / 255;
train_y = double(train_y(1:50000,:));
ds.train_x_unlabled = double(train_x(1:50000,:)) / 255;
%% initialization
global batch_size;
global test_interval;
batch_size = 50;
test_interval = 2;
portion = 1;
validation = .2;
n_networks = 3;
h_layers = {[500,150],[500, 500, 2000],[2500, 2000, 1500, 1000, 500]};
mlp_learning_rates = [0.25, 0.03, 0.02];
dbn_learning_rates = [1, 1, 1];
for i = 1:n_networks
    hidden_layers = h_layers{i};
    %% pre train DBN
    dbn = struct;
    if(pre_loaded_dbn==0)
        dbn_opts = struct;
        dbn_opts.momentum = 0.5;
        dbn_opts.learning_rate = 1;
        dbn_opts.epochs = 2;
        dbn_opts.layers = hidden_layers;
        dbn = test_DBN(ds,dbn_opts);
        save(sprintf('%s %s lr=%g ep=%g mo=%g.mat','Results/MNIST/DBN_Pre'...
            ,mat2str(hidden_layers),dbn_opts.learning_rate...
            ,dbn_opts.epochs,dbn_opts.momentum),'dbn');
    else
        dbn = dbns(i);
    end
    for lables = [50,100, 1000, 50000, 25000]
        fprintf('starting\n\thidden_layers = %s\n\tlables = %d\n',mat2str(hidden_layers),lables); 
        gd_opts = struct;
        gd_opts.layers = hidden_layers;
        gd_opts.epochs = 350;
        ds.train_x = train_x(1:lables,:);
        ds.train_y = train_y(1:lables,:);

        %% train MLP
        gd_opts.learning_rate = mlp_learning_rates(i);
        gd_opts.momentum = 1/2*gd_opts.learning_rate;

        nn_res = train_NN(ds,gd_opts);
        save(sprintf('%s %s lr=%g lab=%d ep=%g mo=%g.mat','Results/MNIST/NN'...
            ,mat2str(hidden_layers),gd_opts.learning_rate...
            ,lables,gd_opts.epochs,gd_opts.momentum),'nn_res');
        %mlp_final_err = mlp.finalErr;

        %% fine tune DBN
        gd_opts.learning_rate = dbn_learning_rates(i);
        gd_opts.momentum = 1/2*gd_opts.learning_rate;
        dbn_res = train_NN(ds,gd_opts,dbnunfoldtonn(dbn,size(ds.train_y,2)));
        save(sprintf('%s %s lr=%g lab=%d ep=%g mo=%g.mat','Results/MNIST/DBN_Post'...
            ,mat2str(hidden_layers),gd_opts.learning_rate...
            ,lables,gd_opts.epochs,gd_opts.momentum),'dbn_res');
    end
end