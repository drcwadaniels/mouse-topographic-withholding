%script to Pull Mouse PreTraining Data


[ Latencies_1, FR2irt_1] = mouse_topography_data_DRL( R1_carray, 1, 2 );
[ Latencies_2,FR2irt_2] = mouse_topography_data_DRL( R2_carray, 1, 2 );
[ Latencies_3, FR2irt_3] = mouse_topography_data_DRL( R3_carray, 1, 2);
[ Latencies_4, FR2irt_4] = mouse_topography_data_DRL( R4_carray, 2, 2 );
[ Latencies_5, FR2irt_5] = mouse_topography_data_DRL( R5_carray, 2, 2 );
[ Latencies_6,FR2irt_6] = mouse_topography_data_DRL( R6_carray, 2, 2 );
[ Latencies_7, FR2irt_7] = mouse_topography_data_DRL( R7_carray, 1, 2 );
[ Latencies_8, FR2irt_8] = mouse_topography_data_DRL( R8_carray, 2, 2 );
[ Latencies_9, FR2irt_9] = mouse_topography_data_DRL( R9_carray, 2, 2 );
[ Latencies_10, FR2irt_10] = mouse_topography_data_DRL( R10_carray, 2, 2 );
[ Latencies_11, FR2irt_11] = mouse_topography_data_DRL( R11_carray, 1, 2 );
[ Latencies_12, FR2irt_12] = mouse_topography_data_DRL( R12_carray, 1, 2);
[ Latencies_13, FR2irt_13] = mouse_topography_data_DRL( R13_carray, 1, 2 );
[ Latencies_14, FR2irt_14] = mouse_topography_data_DRL( R14_carray, 1, 2 );
[ Latencies_15, FR2irt_15] = mouse_topography_data_DRL( R15_carray, 2, 2 );
% [ Latencies_16, FR2irt_16] = mouse_topography_data_FMI( R16_carray, 2, 2 );
Latencies_16=[]; TR_HEirt_16 = []; FR2irt_16 = [];


%Matrix

LATS(1:length(Latencies_1),1) = Latencies_1;
LATS(1:length(Latencies_2),2) = Latencies_2;
LATS(1:length(Latencies_3),3) = Latencies_3;
LATS(1:length(Latencies_4),4) = Latencies_4;
LATS(1:length(Latencies_5),5) = Latencies_5;
LATS(1:length(Latencies_6),6) = Latencies_6;
LATS(1:length(Latencies_7),7) = Latencies_7;
LATS(1:length(Latencies_8),8) = Latencies_8;
LATS(1:length(Latencies_9),9) = Latencies_9;
LATS(1:length(Latencies_10),10) = Latencies_10;
LATS(1:length(Latencies_11),11) = Latencies_11;
LATS(1:length(Latencies_12),12) = Latencies_12;
LATS(1:length(Latencies_13),13) = Latencies_13;
LATS(1:length(Latencies_14),14) = Latencies_14;
LATS(1:length(Latencies_15),15) = Latencies_15;
LATS(1:length(Latencies_16),16) = Latencies_16;
LATS(LATS == 0) = NaN; 



FR2IRT(1:length(FR2irt_1),1) = FR2irt_1;
FR2IRT(1:length(FR2irt_2),2) = FR2irt_2;
FR2IRT(1:length(FR2irt_3),3) = FR2irt_3;
FR2IRT(1:length(FR2irt_4),4) = FR2irt_4;
FR2IRT(1:length(FR2irt_5),5) = FR2irt_5;
FR2IRT(1:length(FR2irt_6),6) = FR2irt_6;
FR2IRT(1:length(FR2irt_7),7) = FR2irt_7;
FR2IRT(1:length(FR2irt_8),8) = FR2irt_8;
FR2IRT(1:length(FR2irt_9),9) = FR2irt_9;
FR2IRT(1:length(FR2irt_10),10) = FR2irt_10;
FR2IRT(1:length(FR2irt_11),11) = FR2irt_11;
FR2IRT(1:length(FR2irt_12),12) = FR2irt_12;
FR2IRT(1:length(FR2irt_13),13) = FR2irt_13;
FR2IRT(1:length(FR2irt_14),14) = FR2irt_14;
FR2IRT(1:length(FR2irt_15),15) = FR2irt_15;
FR2IRT(1:length(FR2irt_16),16) = FR2irt_16;
FR2IRT(FR2IRT == 0) = NaN; 
