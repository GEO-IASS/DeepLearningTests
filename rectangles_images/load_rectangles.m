function [ds] = load_rectangles(test, train, labled, validation, portion)
    %Validation  is the portion of the training data that is to be reserved
    %for the validation dataset
    global batch_size;
    assert(0 <= labled && labled <= 1);
    assert(0 <= validation && validation <= 1);
    assert(0 <= portion && portion <= 1);
    ds = struct;
    n_train = floor(size(train,1)*labled*portion);
    n_val = floor(n_train * validation);
    n_train = n_train - n_val;
    n_train = n_train - mod(n_train,batch_size);
    n_train_unlabled = floor(size(train,1)*portion);
    n_train_unlabled = n_train_unlabled - mod(n_train_unlabled,batch_size);
    n_test = floor(size(test,1)*portion);
    assert(n_train > 0);
    assert(n_val > 0);
    assert(n_test > 0);
    ds.train_x = train(1:n_train,1:end - 1);
    ds.val_x = train(n_train:n_train + n_val,1:end - 1);
    ds.train_x_unlabled = train(1:n_train_unlabled,1:end - 1);
    ds.train_y = softmaxifyY(train(1:n_train,end));
    ds.val_y = softmaxifyY(train(n_train:n_train + n_val,end));
    ds.test_x = test(1:n_test,1:end - 1);
    ds.test_y = softmaxifyY(test(1:n_test,end));
end

