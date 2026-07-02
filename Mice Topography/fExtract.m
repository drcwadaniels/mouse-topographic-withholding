function [ C_array] = fExtract (textDataFileName, sessions)
%Note use the following to run this function
% [A_arrayM, A_arrayF, C_array] = fExtract('FILE NAME GOES HERE')

%Extracts Data from Array A in med pc data files

%Defined Variables
Csta = 314159265;
Cend = 123456789; 
i2 = 0; 
i3 = 1; 


FILE = fopen(textDataFileName);
FIC = textscan(FILE, '%s');
numData = zeros(length(FIC{1}), 1); 
FICu = FIC{1,1}(:,:);
n = numel(FICu); 
FICa = zeros(n,1);
tic
FICa = str2double(FICu(:,1));
toc
fclose(FILE); 

%Find Cend
for i = 1:sessions
    if i == 1
        FICa = FICa;
        continue
    end
[ia,ib] = ismember(Cend,FICa); 
FICa = FICa((ib+1):end,:); 
end

%Constructs C Array for Session
i2 = 0; 
sC = find(FICa(:,1) == Csta);
sC = sC(2); 
eC = find(FICa(:,1) == Cend); 
if isempty(eC) == 1 
     LIST = [ 1 2 3 4 5 6]; 
    [C, ia, ib] = intersect(LIST, FICa);
    eC = ib(1); 
end
sC_t = sC; 

C_array = FICa((sC_t+1):(eC-1)); 

    
end







