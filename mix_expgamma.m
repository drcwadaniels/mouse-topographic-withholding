function [LL] = mix_expgamma(data_matrix, param, FI)





x = data_matrix;

%Mixture Model 
q  = param(1);
q2 = param(2); 
theta = param(3);
c = param(4);
k = param(5); 
k2 = param(6); 




% Mixture Model
    fit_matrix = q * gampdf((x), theta * FI/c, c) + q2*exppdf(x,k) + (1-q-q2)*exppdf(x,k2);
    
    %Simple Model
    %q * gampdf((x), theta * FI/c, c)+ (1-q) * exppdf(x,k);
    %Complex Model
    %q * gampdf((x), theta * FI/c, c) + q2*exppdf(x,k) + (1-q-q2)*exppdf(x,k2);


       

       LL = sum(log(fit_matrix));
       
       if LL == Inf
           LL = -Inf;
       elseif (theta*(FI/c)) < 1
           LL = -Inf;
       elseif q + q2 > 1
          LL = -Inf;
       elseif k2 < k 
           LL = -Inf;
       elseif q + q2 + (1-q-q2)>1 
            LL = -Inf; 
     
       end
end