clear all
load('example_EMG')

%% Set options for rShiftNMF
opts1.maxiter=300;
opts1.conv_crit = 1e-12;
opts.T = [23; 23; 30; 30; 19; 19; 20; 20; 13; 13] + 4*rand(10,1);
opts1.runit=1;
opts1.auto_corr=0;
opts1.dispiter=1;

opts2.maxiter=2500;
opts2.runit=1;
opts2.T = repmat([23; 23; 30; 30; 19; 19; 20; 20; 13; 13],[1,2]) + 4*rand(10,2);
opts2.auto_corr=0;
opts2.dispiter=1;  

opts3.maxiter=3000;
opts3.runit=1;
opts3.T = repmat([23; 23; 30; 30; 19; 19; 20; 20; 13; 13],[1,3]) + 4*rand(10,3);
opts3.auto_corr=0;
opts3.dispiter=1;

opts4.maxiter=3000;
opts4.runit=1;
opts4.T = repmat([23; 23; 30; 30; 19; 19; 20; 20; 13; 13],[1,4]) + 4*rand(10,4);
opts4.auto_corr=0;
opts4.dispiter=1;

L = 4000;
channels = 10;

%Normalize EMG data
normalization = zeros(10,1);
for j = 1:10
    normalization(j) = sqrt(var(EMG(j,:)));
    EMG(j,:) = EMG(j,:)/normalization(j);
end
EMG_cross = EMG(:,4001:8000);
EMG = EMG(:,1:4000);

%% Calculate synergies with 1 Synergy
[W1, H1, T1]=ShiftNMF_reg(EMG, 1, 15, opts1);
opts1_cross.maxiter=300;
opts1_cross.conv_crit = 1e-12;
opts1_cross.T = T1;
opts1_cross.W = W1;
opts1_cross.runit=1;
opts1_cross.auto_corr=0;
opts1_cross.dispiter=1;
opts1_cross.constT = 1;
opts1_cross.constW = 1;
% Cross-Validate Synergy (Fixed W, T)
[W1_cross, H1_cross, T1_cross]=ShiftNMF(EMG_cross, 1, opts1_cross);
    
%% Calculate synergies with 2 Synergies
[W2, H2, T2]=ShiftNMF_reg(EMG, 2, 20, opts2);
opts2_cross.maxiter=2500;
opts2_cross.T = squeeze(T2);
opts2_cross.W = squeeze(W2);
opts2_cross.runit=1;
opts2_cross.auto_corr=0;
opts2_cross.dispiter=1;
opts2_cross.constT = 1;
opts2_cross.constW = 1;
% Cross-Validate Synergy (Fixed W, T)
[W2_cross, H2_cross, T2_cross]=ShiftNMF(EMG_cross, 2, opts2_cross);

%% Calculate synergies with 3 Synergies
[W3, H3, T3]=ShiftNMF_reg(EMG, 3, 22, opts3);
opts3_cross.maxiter=3000;
opts3_cross.T = squeeze(T3);
opts3_cross.W = squeeze(W3);
opts3_cross.runit=1;
opts3_cross.auto_corr=0;
opts3_cross.dispiter=1;
opts3_cross.constT = 1;
opts3_cross.constW = 1;
% Cross-Validate Synergy (Fixed W, T)
[W3_cross, H3_cross, T3_cross]=ShiftNMF(EMG_cross, 3, opts3_cross);

%% Calculate synergies with 4 Synergies
[W4, H4, T4]=ShiftNMF_reg(EMG, 4, 25, opts4);
opts4_cross.maxiter=3000;
opts4_cross.T = squeeze(T4);
opts4_cross.W = squeeze(W4);
opts4_cross.runit=1;
opts4_cross.auto_corr=0;
opts4_cross.dispiter=1;
opts4_cross.constT = 1;
opts4_cross.constW = 1;
% Cross-Validate Synergy (Fixed W, T)
[W4_cross, H4_cross, T4_cross]=ShiftNMF(EMG_cross, 4, opts4_cross);

% Compensate for normalization
for j = 1:10
    W1(j)=W1(j)*normalization(j);
    W1_cross(j)=W1_cross(j)*normalization(j);
    for k = 1:2
        W2(j,k)=W2(j,k)*normalization(j);
        W2_cross(j,k)=W2_cross(j,k)*normalization(j);
    end
    for k = 1:3
        W3(j,k)=W3(j,k)*normalization(j);
        W3_cross(j,k)=W3_cross(j,k)*normalization(j);
    end
    for k = 1:4
        W4(j,k)=W4(j,k)*normalization(j);
        W4_cross(j,k)=W4_cross(j,k)*normalization(j);
    end
end

%% Normalize activation pattern W to have norm 1
for i = 1:N1
    norm_factor_pre = norm(W1);
    W1 = W1/norm_factor_pre;
    H1 = norm_factor_pre*H1;
    norm_factor_pre = norm(W1_cross);
    W1_cross = W1_cross/norm_factor_pre;
    H1_cross = norm_factor_pre*H1_cross;
end
for j = 1:2
    norm_factor_pre = norm(W2(:,j));
    W2(:,j) = W2(:,j)/norm_factor_pre;
    H2(j,:) = norm_factor_pre*H2(j,:);
    norm_factor_pre = norm(W2_cross(:,j));
    W2_cross(:,j) = W2_cross(:,j)/norm_factor_pre;
    H2_cross(j,:) = norm_factor_pre*H2_cross(j,:);
end
for j = 1:3
    norm_factor_pre = norm(W3(:,j));
    W3(:,j) = W3(:,j)/norm_factor_pre;
    H3(j,:) = norm_factor_pre*H3(j,:);
    norm_factor_pre = norm(W3_cross(:,j));
    W3_cross(:,j) = W3_cross(:,j)/norm_factor_pre;
    H3_cross(j,:) = norm_factor_pre*H3_cross(j,:);
end
for j = 1:4
    norm_factor_pre = norm(W4(:,j));
    W4(:,j) = W4(:,j)/norm_factor_pre;
    H4(j,:) = norm_factor_pre*H4(j,:);
    norm_factor_pre = norm(W4_cross(:,j));
    W4_cross(:,j) = W4_cross(:,j)/norm_factor_pre;
    H4_cross(j,:) = norm_factor_pre*H4_cross(j,:);
end

%% Save Results
save('ShiftSynergy_10Channels_cross','W1','W2','H1','H2',...
    'T1','T2','W3','H3','T3','W4','H4','T4','W1_cross','W2_cross', ...
    'W3_cross','W4_cross','H1_cross','H2_cross','H3_cross','H4_cross', ...
    'T1_cross','T2_cross','T3_cross','T4_cross','normalization');
