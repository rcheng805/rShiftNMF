load('example_EMG')

%% Load muscle synergies using rShiftNMF
load('ShiftSynergy_10Channels_cross')
int = 1:4000;
cross_int = 4001:8000;

%% Get VAF from muscle synergies extracted by rShiftNMF
VAF1 = getVAF(W1,H1,T1,EMG(:,int));
VAF2 = getVAF(W2,H2,T2,EMG(:,int));
VAF3 = getVAF(W3,H3,T3,EMG(:,int));
VAF4 = getVAF(W4,H4,T4,EMG(:,int));
VAF1_cross = getVAF(W1_cross,H1_cross,T1_cross,EMG(:,cross_int));
VAF2_cross = getVAF(W2_cross,H2_cross,T2_cross,EMG(:,cross_int));
VAF3_cross = getVAF(W3_cross,H3_cross,T3_cross,EMG(:,cross_int));
VAF4_cross = getVAF(W4_cross,H4_cross,T4_cross,EMG(:,cross_int));
VAF = [0, mean(VAF1),mean(VAF2),mean(VAF3),mean(VAF4)];
VAF_cross = [0, mean(VAF1_cross),mean(VAF2_cross),mean(VAF3_cross),mean(VAF4_cross)];
VAF_diff = [VAF(:,2)-VAF(:,1), VAF(:,3)-VAF(:,2), VAF(:,4)-VAF(:,3), VAF(:,5)-VAF(:,4)];
VAF_cross_diff = [VAF_cross(:,2)-VAF_cross(:,1), VAF_cross(:,3)-VAF_cross(:,2), VAF_cross(:,4)-VAF_cross(:,3), VAF_cross(:,5)-VAF_cross(:,4)];


%% Load muscle synergies using NNMF
load('NNMF_Synergy')

VAF_nnmf = [0, mean(VAF1), mean(VAF2), mean(VAF3), mean(VAF4)];
VAF_nnmf_cross = [0, mean(VAF1_cross), mean(VAF2_cross), mean(VAF3_cross), mean(VAF4_cross)];

%% Plot results
afigure;
hold on
plot(0:4,VAF)
plot(0:4,VAF_cross)
plot(0:4,VAF_nnmf)
plot(0:4,VAF_nnmf_cross)
hold off
xlabel('Number of Synergies')
ylabel('VAF')
title('VAF vs. Number of Synergies')
legend('rShiftNMF','Cross-Validated rShiftNMF', 'NNMF','Cross-Validated NNMF')
set(gca,'FontSize',14)
ylim([0,1])
