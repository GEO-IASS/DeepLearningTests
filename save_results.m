function [] = save_results(nns, opts, name)
    res.trained_epochs = [nns.trained_epochs];
    res.errors = [nns.final_er];
    res.size = nns(1).size;
    res.activation_function = nns(1).activation_function;
    res.learning_rate = nns(1).learningRate;
    res.momentum = nns(1).momentum;
    %res.scaling_learningRate = nns(1).scaling_learningRate;
    %res.weightPenaltyL2 = nns(1).weightPenaltyL2;
    %res.nonSparsityPenalty = nns(1).nonSparsityPenalty;
    %res.sparsityTarget = nns(1).sparsityTarget;
    %res.inputZeroMaskedFraction = nns(1).inputZeroMaskedFraction;
    %res.dropoutFraction = nns(1).dropoutFraction;
    res.output = nns(1).output;
    [bestErr, bestNN] = min([nns.final_er]);
    save(name, 'res', 'opts', 'bestErr', 'bestNN');
end