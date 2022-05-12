%Script to run Mixture Models

%When subtracting min it is min(x) - 0.005

%Edit Per Rat
FI = 6;
Subject = 8; 
Rat = unnamed(:,Subject);

data = Rat;
data(isnan(data) == 1) = [];
data(data == 0) = [];
m = min(data)-.005;
data = data - m; 

%Set initial parameters (if necessary)
param_init = [.90 .6 12 .5 .1 1];

param_lb = [0  0 0 0 0 0];
param_ub = [1 1 Inf Inf Inf Inf];

%Model 1 has parameters q, theta, c, and k, with param_ub = [1 Inf Inf Inf]
%Model 2 has parameters q, q2, theta, c, k_s, k_l, with param_ub [1 1 Inf Inf Inf Inf]

for i = 1:2
   
    [p_fit, LL_out, exitFlag] = fMixSolv(data, param_init, param_lb, param_ub, FI); 
    
    param_init = p_fit;
    pcollector = p_fit; 
    
end
   

    [y,x] = ecdf(data); 
    plot(x,y, 'bo');
    hold;
    MixVisual = p_fit(1)* gamcdf(x,p_fit(3)*(FI/p_fit(4)),p_fit(4)) + p_fit(2) * expcdf((x),p_fit(5)) + (1-p_fit(1)-p_fit(2))* expcdf((x), p_fit(6));
    %MixVisual = p_fit(1) * gamcdf((x), p_fit(2) * FI/p_fit(3), p_fit(3)) + (1-p_fit(1))*expcdf(x,p_fit(4));
    
    
    %p_fit(1)* gamcdf(x,p_fit(3)*(FI/p_fit(4)),p_fit(4)) + p_fit(2) * expcdf((x),p_fit(5)) + (1-p_fit(1)-p_fit(2))* expcdf((x), p_fit(6)); 
    %p_fit(1) * gamcdf((x), p_fit(2) * FI/p_fit(3), p_fit(3)) + (1-p_fit(1))*expcdf(x,p_fit(4));
    
    
    plot(x,MixVisual,'b-');
    


    








 
 