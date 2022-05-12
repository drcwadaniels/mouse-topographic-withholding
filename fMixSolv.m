function [ param_fit, LL_out, exit_flag] = fMixSolv(data_matrix, param_init, param_lb, param_ub, FI)
    
    SAoptions = saoptimset('simulannealbnd');
    SAoptions = saoptimset(SAoptions, 'TemperatureFcn', @temperatureboltz, 'AnnealingFcn', @annealingboltz, 'MaxFunEvals', 100000,  'Display', 'iter', 'TolFun',1e-008);



    [param_fit, fval, exit_flag] = simulannealbnd(@(param)-mix_expgamma(data_matrix, param, FI), param_init,param_lb,param_ub, SAoptions);
    
    [param_fit, fval, exit_flag] = patternsearch(@(param)-mix_expgamma(data_matrix, param, FI), param_fit, [], [], [], [], param_lb, param_ub, SAoptions);
    

     
    LL_out = -fval;
    
    


end