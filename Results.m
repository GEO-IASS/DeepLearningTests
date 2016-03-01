load('Results/NN_test_1.mat');
nn.res = res;
load('Results/DBN_test_2_bigBatch.mat');

dbn_res = plot(res.dbns.trained_epochs, res.dbns.final_er);
xlabel('trained epochs');
ylabel('test error rate');
hold on
nn_res = plot(nn.res.nns.trained_epochs, nn.res.nns.final_er);

legend('DBN', 'NN');
%legend(nn_res, 'NN');
%legend(

