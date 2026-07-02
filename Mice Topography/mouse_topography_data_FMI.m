function [ Latencies, FR2irt ] = mouse_topography_data_FMI( carray, topog, FR )



%Extract C_array Data
C_array =  carray; 
event = mod(C_array,1);
time = (C_array - event);
event_F = round((mod(C_array,1))*1000);
TbT_1 = [time,event_F];
if topog == 1
    RESP = 600;
elseif topog == 2
    RESP = 800;
end

trial_starts = find(event_F == 100);
corrects = find(event_F == 212);
incorrects = find(event_F == 211); 
response = find(event_F == RESP); 

allends = transpose(sort([transpose(corrects),transpose(incorrects)])); 

if length(trial_starts) > length(allends)
    trial_starts = trial_starts(1:(end-1));
end
%Trial by Trial Event Data
for i = 1:length(trial_starts) 
    trialevents = transpose(TbT_1(trial_starts(i):allends(i),2));
    eventtimes = transpose(TbT_1(trial_starts(i):allends(i),1));
    FMI_trialevents(i) = {trialevents};
    FMI_eventtimes(i) = {eventtimes}; 
    
end

%Latencies
for i = 1:length(trial_starts);
    events = cell2mat(FMI_trialevents(i));
    times = cell2mat(FMI_eventtimes(i)); 
    trialstart = times(1);
    Firstresp = find(events == RESP);
   if isempty(Firstresp) == 1
       Firstresp = length(times); 
    elseif isempty(Firstresp) == 0
    Firstresp = Firstresp(1);
    end
    Latencies(i) = (times(Firstresp) - trialstart)/100;
end

%IRTs
for i = 1:length(trial_starts);
    events = cell2mat(FMI_trialevents(i));
    times = cell2mat(FMI_eventtimes(i)); 
    trialstart = times(1);
    Firstresp = find(events == RESP);
    Secondresp = find(events == 212);
    if isempty(Secondresp) == 1
    Secondresp_alt = find(events == 211);
    end
   if isempty(Firstresp) == 1
       Firstresp = length(times); 
    elseif isempty(Firstresp) == 0
    Firstresp = Firstresp(1);
   end
    
   if isempty(Secondresp) == 1
       Secondresp = length(times);
   elseif isempty(Secondresp) == 0
       Secondresp = Secondresp(1); 
   end
    FR2irt(i) = (times(Secondresp) - times(Firstresp))/100;
end



end

