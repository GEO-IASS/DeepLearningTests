load mnist_uint8;
load_news;
%train_x = double(train_x) / 255;
%test_x  = double(test_x) / 255;
%train_y = double(train_y);
% test_y  = double(test_y);
% train_x = lda_train_data(1:11200,:);
% labels = train_labels(1:11200,:);
% train_y = zeros(length(labels),20);
% 
% 
% for i = 1:length(train_y)
%     train_y(i,labels(i)) = 1;
% end
% test_x = lda_test_data;
% labels = test_labels;
% test_y = zeros(length(labels),20);
% for i = 1:length(test_y)
%     test_y(i,labels(i)) = 1;
% end

load '20news_w100.mat';
batch_size = 20;
train_test_ratio = .7;
Y = softmaxifyY(newsgroups');
X = double(full(documents'));

[no_data,~] = (size(X));
no_test_data = floor(no_data*train_test_ratio);
no_test_data = no_test_data - mod(no_test_data,batch_size);
train_x = X(1 : no_test_data,:);
train_y = Y(1 : no_test_data,:);
test_x = X(no_test_data + 1 : end,:);
test_y = Y(no_test_data + 1 : end,:);

assert(length(train_x) + length(test_x) == no_data);

% normalize
%[train_x, mu, sigma] = zscore(train_x);
%test_x = normalize(test_x, mu, sigma);


%% ex6 neural net with sigmoid activation and plotting of validation and training error
% split training data into training and validation data

vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);

rand('state',0)
bestErr = 1;
nns = [];
opts_list = [];
bestNN = 1;
nn = nnsetup([100 200 size(Y,2)]);
nn.learningRate = .1    ;
nn.momentum = .5;
nn.output               = 'softmax';                   %  use softmax output
nn.activation_function = 'sigm';
opts.numepochs          = 5;                %  Number of full sweeps through data
opts.batchsize          = batch_size;                        %  Take a mean gradient step over this many samples
opts.plot               = 0;                           %  enable plotting
nn.trained_epochs = 0;
for nn_index = 1:25

nn = nntrain(nn, train_x, train_y, opts);                %  nntrain takes validation set as last two arguments (optionally)
[er, bad] = nntest(nn, test_x, test_y);
nn.trained_epochs = nn.trained_epochs + opts.numepochs;

if er<bestErr
	bestErr = er;
	bestNN = nn_index;
end
nn.finalErr = er;
nns = [nns, nn];
opts_list = [opts_list, opts];

end


res.nns.trained_epochs = [nns.trained_epochs];
res.nns.final_er = [nns.finalErr];
res.size = nns(1).size;
res.activation_function = nns(1).activation_function;
res.learning_rate = nns(1).learningRate;
res.momentum = nns(1).momentum;
res.scaling_learningRate = nns(1).scaling_learningRate;
res.weightPenaltyL2 = nns(1).weightPenaltyL2;
res.nonSparsityPenalty = nns(1).nonSparsityPenalty;
res.sparsityTarget = nns(1).sparsityTarget;
res.inputZeroMaskedFraction = nns(1).inputZeroMaskedFraction;
res.dropoutFraction = nns(1).dropoutFraction;
res.output = nns(1).output;
res.testing = nns(1).testing;
save('NN_epochs_test_5.mat', 'res', 'opts_list', 'bestErr', 'bestNN');

%assert(er < 0.02, 'Too big error');
