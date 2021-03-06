% This script will run all architectures with all lables amounts with the specifyed
% learning rates for the respective architecture and save the results in a
% manner so that only comparable results are saved in the same dir.
% if previous results already exist, it will not redo them.


%% Load data
if not(exist('train'))
    train = load('rectangles_im_train.amat');
end
if not(exist('test'))
    test = load('rectangles_im_test.amat');
end
%% Configuration
global batch_size;
global test_interval;
batch_size = 50;
test_interval = 2;
portion = 1;
validation = .2;
hidden_layers = {[500,500,1000],[1500,2000,3000],[3000,4000,6000]};
mlp_learning_rates = {[.0001,.001,.01,.1],[.0001,.001,.01,.1],[.0001,.001,.01,.1]};
dbn_learning_rates = {[.1],[.1],[.1]};
ddbn_learning_rates = mlp_learning_rates;
labels = [.006,.01,.1,1,.2,.5];
base_folder = 'Results/Rectangles';
dbn_base_opts = struct;
dbn_base_opts.momentum = 0;
dbn_base_opts.epochs = 150;

%% initialization
ds = load_rectangles(test,train,1,validation,portion);

%Checking for configuration mistakes
assert(all(size(hidden_layers) == size(mlp_learning_rates)));
assert(all(size(mlp_learning_rates) == size(dbn_learning_rates)));
assert(all(size(dbn_learning_rates) == size(ddbn_learning_rates)));
%% Run
for i = 1:size(hidden_layers,2)
    layer_base_folder = sprintf('%s/%s',base_folder,mat2str(hidden_layers{i}));
    mkdir(layer_base_folder);
    %% pre train DBN
    dbn_opts = dbn_base_opts;
    dbn_opts.layers = hidden_layers{i};
    for dbn_learning_rate = dbn_learning_rates{i}
        dbn_opts.learning_rate = dbn_learning_rate;
        dbn_file_name = sprintf('%s/DBN lr=%g po=%g ep=%g mo=%g.mat',layer_base_folder...
            ,dbn_opts.learning_rate...
            ,portion,dbn_opts.epochs,dbn_opts.momentum);
        % Load DBN if it exists
        if exist(dbn_file_name,'file') == 2
            fprintf('\"%s\"\nfound. Loading...\n',dbn_file_name);
            dbn = (load(dbn_file_name));
            dbn = dbn.dbn;
        else
            fprintf('\"%s\"\n not found. Starting cd1 training\n',dbn_file_name);
            dbn = test_DBN(ds,dbn_opts);
            save(dbn_file_name,'dbn');
        end
    end
    %% Train over different amount of lables
    for lable = labels
        %% setup gd
        fprintf('starting\n\thidden_layers = %s\n\tlables = %d\n',mat2str(hidden_layers{i}),lable); 
        ds = load_rectangles(test,train,lable,validation,portion);
        gd_opts = struct;
        gd_opts.layers = hidden_layers{i};
        gd_opts.epochs = 500;
        gd_opts.early_stop = 30;
        lables_base_folder = sprintf('%s/lb=%g/es=%d ep=%d po=%g val=%g bs=%d ti=%d',layer_base_folder...
            ,lable,gd_opts.early_stop,gd_opts.epochs,portion,validation...
            ,batch_size,test_interval);
        mkdir(lables_base_folder);
         %% train MLPs
        for learning_rate = mlp_learning_rates{i}
            gd_opts.momentum = .1;
            gd_opts.learning_rate = learning_rate;
            mlp_file_name = sprintf('%s/MLP lr=%g mo=%g.mat',lables_base_folder...
                ,gd_opts.learning_rate,gd_opts.momentum);
            if exist(mlp_file_name,'file') == 2
                fprintf('%s already exists, ignoring\n',mlp_file_name);
            else
                fprintf('%s does not exists, starting gd training\n',mlp_file_name);
                nn_res = train_NN(ds,gd_opts);
                save(mlp_file_name,'nn_res');
            end
        end
        %% fine tune DBNs
        
        for learning_rate = ddbn_learning_rates{i}
            gd_opts.momentum = .1;
            gd_opts.learning_rate = learning_rate;
            ddbn_file_name = sprintf('%s/DDBN lr=%g mo=%g.mat',lables_base_folder...
                ,gd_opts.learning_rate,gd_opts.momentum);
            if exist(ddbn_file_name,'file') == 2
                fprintf('%s already exists, ignoring\n',ddbn_file_name);
            else
                fprintf('%s does not exists, starting gd training\n',ddbn_file_name);
                dbn_res = train_NN(ds,gd_opts,dbnunfoldtonn(dbn,size(ds.train_y,2)));
                save(ddbn_file_name,'dbn_res');
            end
        end
    end
end