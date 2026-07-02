function [ Latencies, FR2irt ] = mouse_topography_data_DRL( carray, topog, FR )



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

i1 = 0;
i2 = 0;
i3 = 0;
i4 = 0;
TR_HEirt = []; 

for i = 1:length(TbT_1)
   if TbT_1(i,2) == 100 && i1 == 0
      i1 = 1;
      stime = TbT_1(i,1);
   elseif TbT_1(i,2) == RESP  && i1 == 1
      etime  = TbT_1(i,1); 
      i1 = 2; 
   end
   
   if i1 == 2
       i2 = i2 + 1;
       Latencies(i2) = (etime-stime)/100;
       if Latencies(i2) == 0
           Latencies(i2) = .0001;
       end
       i1 = 0; 
   end 
end

i1 = 0;
i2 = 0;
count = 0; 

for i = 1:length(TbT_1)
   if TbT_1(i,2) == 100 && i1 == 0
      i1 = 1;
   end
   if TbT_1(i,2) == RESP  && TbT_1(i+1,2) == 212
        stime  = TbT_1(i,1); 
        i1 = 2;
   end
    if TbT_1(i,2) == 200
      etime = TbT_1(i,1);
      i1 = 3;
   elseif TbT_1(i,2) == 211
       stime = 0;
       etime = 0; 
       i1 = 0;
   end
   if i1 == 3
      i2 = i2 + 1;
      TR_HEirt(i2) = (etime - stime)/100;
      i1 = 0;
   end
    
    
    
end

i1 = 0;
i2 = 0;
count = 0;

for i = 1:length(TbT_1)
    
   if TbT_1(i,2) == 100 && i1 == 0
      i1 = 1;
   elseif TbT_1(i,2) == 212 && i1 == 1
       i1 = 0; 
       count = 0;
   end
   if TbT_1(i,2) == RESP  && i1 == 1 && count == 0
       stime  = TbT_1(i,1);
       count = count + 1;
   elseif TbT_1(i,2) == RESP && count == 1
       etime = TbT_1(i,1);
       i2 = i2 + 1;
       FR2irt(i2) = (etime - stime)/100;
       stime = etime; 
   end
       
end



end

