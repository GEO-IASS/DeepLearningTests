% use this file to do all the messy work

pre_loaded_dbn = 0; %set to 1 if the pre training of dbn is already done
%% Load data
if not(exist('train'))
    train = load('rectangles_im_train.amat');
end
if not(exist('test'))
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
%% Run
for hidden_layers = {[500,500,1000],[1500,2000,3000]}
    hidden_layers = hidden_layers{1};
    %% pre train DBN
    dbn_file_name = sprintf('%s [%d,%d,%d] lr=%g po=%g ep=%g mo=%g.mat','Results/Rectangles/DBN_Pre'...
        ,dbn_opts.layers(1),dbn_opts.layers(2),dbn_opts.layers(3),dbn_opts.learning_rate...
        ,portion,dbn_opts.epochs,dbn_opts.momentum);
    if exists(dbn_file_name,'file') == 2
        dbn = load(dbn_file_name);
    else
        dbn_opts = struct;
        dbn_opts.momentum = 0;
        dbn_opts.learning_rate = .1;
        dbn_opts.epochs = 100;
        dbn_opts.layers = hidden_layers;
        dbn = test_DBN(ds,dbn_opts);
        save(dbn_file_name,'dbn');
    end
    %% Train over different amount of lables
    for lables = [.006,.01,.1,1]
        %% setup gd
        fprintf('starting\n\thidden_layers = %s\n\tlables = %d\n',mat2str(hidden_layers),lables); 
        ds = load_rectangles(test,train,lables,validation,portion);
        gd_opts = struct;
        gd_opts.layers = hidden_layers;
        gd_opts.epochs = 500;
        gd_opts.early_stop = 20;
        run_folder = sprintf('%s %s lb=%g es=%d ep=%d po=%g val=%g bs=%d ti=%d','Results/Rectangles'...
            ,mat2str(gd_opts.layers),lables,gd_opts.early_stop,gd_opts.epochs,portion,validation...
            ,batch_size,test_interval);
         %% train MLPs
        for learning_rate = []
           
            gd_opts.momentum = 0;
            gd_opts.learning_rate = learning_rate;
            nn_res = train_NN(ds,gd_opts);
            save(sprintf('%s/NN lr=%g mo=%g.mat',run_folder...
                ,gd_opts.learning_rate,gd_opts.momentum),'nn_res');
        end
        %% fine tune DBNs
        
        for learning_rate = []
            gd_opts.momentum = learning_rate;
            gd_opts.learning_rate = .1;
            dbn_res = train_NN(ds,gd_opts,dbnunfoldtonn(dbn,size(ds.train_y,2)));
            save(sprintf('%s/DDBN lr=%g mo=%g.mat',run_folder...
                ,gd_opts.learning_rate,gd_opts.momentum),'dbn_res');
        end
    end
end