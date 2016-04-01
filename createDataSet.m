function [dataSet] = createDataSet(X,Y,train_test_ratio,labled)

global batch_size;

switch nargin
    case 3
        labled = size(X,1);
    case 4
end
ix = randperm(size(X,1));
X = X(ix,:);
Y = Y(ix,:);
dataSet = struct;
no_test_data = floor(labled*train_test_ratio);
no_test_data = no_test_data - mod(no_test_data,batch_size);
dataSet.X = X(1:end - mod(end,batch_size),:);
dataSet.Y = Y(1:end - mod(end,batch_size),:);
dataSet.train_x = dataSet.X(1 : no_test_data,:);
dataSet.train_y = dataSet.Y(1 : no_test_data,:);
dataSet.test_x = dataSet.X(no_test_data + 1 : end,:);
dataSet.test_y = dataSet.Y(no_test_data + 1 : end,:);
end

