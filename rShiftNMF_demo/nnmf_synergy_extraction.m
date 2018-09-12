%% Synergy Extraction by NNMF
load('example_EMG')

EMG_cross = squeeze(EMG(:,4001:8000));
EMG = squeeze(EMG(:,1:4000));

L = 4000;
channels = 10;

%% Extract NNMF muscle synergies from EMG 
[W1_nnmf, H1_nnmf]=nnmf(EMG,1);
[W2_nnmf, H2_nnmf]=nnmf(squeeze(EMG),2);
[W3_nnmf, H3_nnmf]=nnmf(squeeze(EMG),3);
[W4_nnmf, H4_nnmf]=nnmf(squeeze(EMG),4);

%% Get VAF from each set of muscle syneriges
re = W1_nnmf*H1_nnmf;
for j = 1:channels
    VAF1(j) = 1 - norm(squeeze(EMG(j,:)) - re(j,:))/norm(squeeze(EMG(j,:)));
end
re = squeeze(W2_nnmf)*squeeze(H2_nnmf);
for j = 1:channels
    VAF2(j) = 1 - norm(squeeze(EMG(j,:)) - re(j,:))/norm(squeeze(EMG(j,:)));
end
re = squeeze(W3_nnmf)*squeeze(H3_nnmf);
for j = 1:channels
    VAF3(j) = 1 - norm(squeeze(EMG(j,:)) - re(j,:))/norm(squeeze(EMG(j,:)));
end
re = squeeze(W4_nnmf)*squeeze(H4_nnmf);
for j = 1:channels
    VAF4(j) = 1 - norm(squeeze(EMG(j,:)) - re(j,:))/norm(squeeze(EMG(j,:)));
end

%% Get cross-validated (fixed W) VAF from each set of muscle synergies
iter = 2000;
V = squeeze(EMG_cross);
H = squeeze(H1_nnmf);
for j = 1:iter
    W = squeeze(W1_nnmf);
    a = W'*V;
    b = W'*W*H;
    H = H.*(a./b);
end
W1_nnmf_cross = W;
H1_nnmf_cross = H;
re = W*H;
for j = 1:channels
    VAF1_cross(j) = 1 - norm(squeeze(EMG_cross(j,:)) - re(j,:))/norm(squeeze(EMG_cross(j,:)));
end

H = squeeze(H2_nnmf);
for j = 1:iter
    W = squeeze(W2_nnmf);
    a = W'*V;
    b = W'*W*H;
    H = H.*(a./b);
end
W2_nnmf_cross = W;
H2_nnmf_cross = H;
re = W*H;
for j = 1:channels
    VAF2_cross(j) = 1 - norm(squeeze(EMG_cross(j,:)) - re(j,:))/norm(squeeze(EMG_cross(j,:)));
end

H = squeeze(H3_nnmf);
for j = 1:iter
    W = squeeze(W3_nnmf);
    a = W'*V;
    b = W'*W*H;
    H = H.*(a./b);
end
W3_nnmf_cross = W;
H3_nnmf_cross = H;
re = W*H;
for j = 1:channels
    VAF3_cross(j) = 1 - norm(squeeze(EMG_cross(j,:)) - re(j,:))/norm(squeeze(EMG_cross(j,:)));
end

H = squeeze(H4_nnmf);
for j = 1:iter
    W = squeeze(W4_nnmf);
    a = W'*V;
    b = W'*W*H;
    H = H.*(a./b);
end
W4_nnmf_cross = W;
H4_nnmf_cross = H;
re = W*H;
for j = 1:channels
    VAF4_cross(j) = 1 - norm(squeeze(EMG_cross(j,:)) - re(j,:))/norm(squeeze(EMG_cross(j,:)));
end

%% Save Results
save('NNMF_Synergy','W1_nnmf','W2_nnmf','W3_nnmf','W4_nnmf',...
    'VAF1','VAF2','VAF3','VAF4','VAF1_cross','VAF2_cross','VAF3_cross',...
    'VAF4_cross')

