% use this file to do all the messy work

pre_loaded_dbn = 0; %set to 1 if the pre training of dbn is already done
%% Load data
if not(exist('train') || exist('test'))
    train = load('rectangles_im_train.amat');
    test = load('rectangles_im_test.amat');
end
%% initialization
global batch_size;
global test_interval;
batch_size = 50;
test_interval = 2;
portion = 1;
validation = .2;
ds = load_rectangles(test,train,1,validation,portion);
for hidden_layers = {[500,500,1000],[1500,2000,3000]}
    hidden_layers = hidden_layers{1};
    %% pre train DBN
        dbn_opts = struct;
        dbn_opts.momentum = 0;
        dbn_opts.learning_rate = .1;
        dbn_opts.epochs = 100;
        dbn_opts.layers = hidden_layers;
        dbn = test_DBN(ds,dbn_opts);
        save(sprintf('%s [%d,%d,%d] lr=%g po=%g ep=%g mo=%g.mat','Results/Rectangles/DBN_Pre'...
            ,dbn_opts.layers(1),dbn_opts.layers(2),dbn_opts.layers(3),dbn_opts.learning_rate...
            ,portion,dbn_opts.epochs,dbn_opts.momentum),'dbn');
    for lables = [.006,.01,.1,1]
        fprintf('starting\n\thidden_layers = %s\n\tlables = %d\n',mat2str(hidden_layers),lables); 
        ds = load_rectangles(test,train,lables,validation,portion);
        gd_opts = struct;
        gd_opts.layers = hidden_layers;
        gd_opts.epochs = 500;

        %% train MLP
        gd_opts.momentum = 0;
        gd_opts.learning_rate = .01;

        nn_res = train_NN(ds,gd_opts);
        save(sprintf('%s [%d,%d,%d] lr=%g po=%g ep=%g mo=%g lb=%g.mat','Results/Rectangles/NN'...
            ,gd_opts.layers(1),gd_opts.layers(2),gd_opts.layers(3),gd_opts.learning_rate...
            ,portion,gd_opts.epochs,gd_opts.momentum,lables),'nn_res');
        %mlp_final_err = mlp.finalErr;

        %% fine tune DBN
        gd_opts.momentum = 0;
        gd_opts.learning_rate = .1;
        dbn_res = train_NN(ds,gd_opts,dbnunfoldtonn(dbn,size(ds.train_y,2)));
        save(sprintf('%s [%d,%d,%d] lr=%g po=%g ep=%g mo=%g lb=%g.mat','Results/Rectangles/DBN_Post'...
            ,gd_opts.layers(1),gd_opts.layers(2),gd_opts.layers(3),gd_opts.learning_rate...
            ,portion,gd_opts.epochs,gd_opts.momentum,lables),'dbn_res');
    end
end
%% Plot
plot_res(nns,dbns);