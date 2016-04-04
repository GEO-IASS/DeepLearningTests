function plot_res(nns, dbns)
    plot([dbns.trained_epochs], [dbns.final_er]);
    xlabel('trained epochs');
    ylabel('test error rate');
    hold on
    plot([nns.trained_epochs], [nns.final_er], '--');
    legend('DBN', 'NN');
end