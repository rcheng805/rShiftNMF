%% Get VAF of the muscle synergies for EMG activity
function VAF = getVAF(W,H,T,EMG)
% Set variables
[channels, K] = size(W);
L = length(EMG);
VAF = zeros(channels,1);

% Get reconstruction if there is one muscle synergy
if K == 1
    re_EMG = W*H;
    for j = 1:channels
        re_EMG(j,:) = circshift(re_EMG(j,:),[1, round(-T(j))]);
    end
    for j = 1:channels
        VAF(j)= 1 - norm(EMG(j,:) - re_EMG(j,:))/norm(EMG(j,:));
    end
    
% Get reconstruction if there are multiple muscle synergies
else
    re_EMG = zeros(K,channels, L);
    for k = 1:K
        for j = 1:channels
            re_EMG(k,j,:) = circshift(W(j,k)*H(k,:), [1, round(-T(j,k))]);
        end
    end
    a = squeeze(sum(re_EMG));
    for j = 1:channels
        VAF(j)= 1 - norm(EMG(j,:) - a(j,:))/norm(EMG(j,:));
    end
end