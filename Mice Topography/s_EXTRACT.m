%Script for Extracting Response Topography datat

clear all; 
%EXp 1 notes
%%Session 33 = 9/9/2015 for rats 1-7, 9-11, 13-16
%%Session 32 = 9/9/2015 for rats 8 and 12



session = 3;

%exp 2, session 1 = program error (Friday, 4/7/17)


%Exp 1 @ 32
% session_r8 = session;
% session_r12 = session_r8; 
session_r8 = session;
session_r12 = session;

%Mouse 1
textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M1_DRLtraining2.txt';
[R1_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M2_DRLtraining2.txt';
[R2_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M3_DRLtraining2.txt';
[R3_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M4_DRLtraining2.txt';
[R4_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M5_DRLtraining2.txt';
[R5_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M6_DRLtraining2.txt';
[R6_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M7_DRLtraining2.txt';
[R7_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M8_DRLtraining2.txt';
[R8_carray] = fExtract(textDataFileName, session_r8);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M9_DRLtraining2.txt';
[R9_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M10_DRLtraining2.txt';
[R10_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M11_DRLtraining2.txt';
[R11_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M12_DRLtraining2.txt';
[R12_carray] = fExtract(textDataFileName, session_r12);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M13_DRLtraining2.txt';
[R13_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M14_DRLtraining2.txt';
[R14_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M15_DRLtraining2.txt';
[R15_carray] = fExtract(textDataFileName, session);

textDataFileName ='C:\Users\cwdan\Google Drive\Mutant Mice Projects\Topography Project\Exp 2 txt data\M16_DRLtraining2.txt';
[R16_carray] = fExtract(textDataFileName, session);

