function [W, H, T, varexpl, cost]=ShiftNMF(X,noc,varargin)
%
% algorithm for Shifted Non-negative Matrix factorization described in
% Morten M�rup, Kristoffer Hougaard Madsen and Lars Kai Hansen " Shifted
% Non-negative Matrix Factorization", submitted to MLSP2007.
% Please refer to this article when publishing results obtained by the
% algortihm.
%
% Usage:
% [W, S, T, varexpl]=shiftNMF(X,noc,opts)
%
% Input:
% X             I x J matrix
% noc           number of components
% opts. 
%    W          Initial W
%    H          Initial H
%    T          Initial T
%    maxiter    maximum number of iteration (default: 1000)
%    conv_crit  Convergence Criterion (default relative change of costfunction of 1e-6)
%    ConstW     1: Constant W else estiamte
%    ConstH     1: Constant H
%    ConstT     1: Constant T
%    alpha      initial stepsize of H update
%    dispiter   1: Display result for iteration 
%    auto_corr   1: use autocorrelation every 20th iteration to estimate delays
%               0: don't use autocorrelation (default autocorr=1 ).
%    lambda     Strength of constraint imposed (default no constraints, i.e. lambda=0)
%    smoothtype 'curv'          : curvature
%               'grad'          : gradient, i.e. smoothness
%               'regularization': Regularize solution by imposing penalty
%                                 on H
%    smoothnorm 'L1'        : smoothtype imposed by absolute values (L1-norm)
%    smoothnorm 'L2'        : smoothtype imposed by Frobenius norm  (L2-norm)
%   
%
% Output:
% W             I x noc mixing matrix 
% H             noc x J matrix of sources
% T             I x noc matrix of time delays
%     
% varexpl   vector of Variation explained for ShiftNMF solutions through
%           the iterations
% cost      vector of cost function value through the iterations

%
% Copyright (C) Morten M�rup, Kristoffer Hougaard Madsen and Technical University of Denmark, 
% April 2007
  
if nargin>=3, opts = varargin{1}; else opts = struct; end
varexplold=0;
runit=mgetopt(opts,'runit',0);
if ~isfield(opts,'H') & runit~=1 
    disp(['Finding the best out of ' num2str(10) ' initial solution'])
    for k=1:10
        if k==1
            disp(['Now Estimating ' num2str(k) 'st solution ']);
        elseif k==2
            disp(['Now Estimating ' num2str(k) 'nd solution ']);
        elseif k==3
            disp(['Now Estimating ' num2str(k) 'rd solution ']);
        else
            disp(['Now Estimating ' num2str(k) 'th solution ']);
        end
        optsn=opts;
        optsn.dispiter=0;
        optsn.runit=1;    
        optsn.maxiter=25;
        [Wq, Hq, Tq, varexpl]=ShiftNMF(X,noc,optsn);
        if varexpl>varexplold
            disp(['Best variation explained ' num2str(varexpl) ]);
            varexplold=varexpl;
            W=Wq;
            H=Hq;
            T=Tq;
        end
    end
else
    mx=max(abs(X(:)));
    W=mgetopt(opts,'W',mx*rand(size(X,1),noc));
    H=mgetopt(opts,'H',mx*rand(noc,size(X,2)));
    T=mgetopt(opts,'T',zeros(size(W)));
end
%W=W./repmat(sqrt(sum(W.^2,1)),[size(W,1),1]);
 
nyT=mgetopt(opts,'nyT',1);
maxiter=mgetopt(opts,'maxiter',1000);
conv_crit=mgetopt(opts,'convcrit',1e-6);
SST=norm(X,'fro')^2;
constH=mgetopt(opts,'constH',0);
constW=mgetopt(opts,'constW',0);
constT=mgetopt(opts,'constT',0);
lambda=mgetopt(opts,'lambda',0);
auto_corr=mgetopt(opts,'auto_corr',1);
smoothtype=mgetopt(opts,'smoothtype','none');
smoothnorm=mgetopt(opts,'smoothnorm','L2');
alpha=mgetopt(opts,'alpha',1);
dispiter=mgetopt(opts,'dispiter',1);

switch smoothtype
    case 'curv'
        L=-2*eye(size(H,2));
        L(2:end,1:end-1)=L(2:end,1:end-1)+eye(size(H,2)-1);
        L(1:end-1,2:end)=L(1:end-1,2:end)+eye(size(H,2)-1);
        L(:,1)=0;
        L(:,end)=0;
    case 'grad' 
        L=-eye(size(H,2));
        L(2:end,1:end-1)=L(2:end,1:end-1)+eye(size(H,2)-1);
        L(:,end)=0;
    case 'regularization'
        L=eye(size(H,2));
    otherwise
        L=0;
end
L=sparse(L);
N=size(X,2);
f=j*2*pi*[0:N-1]/N;
Xf=fft(X,[],2);
Xf=Xf(:,1:floor(size(Xf,2)/2)+1);
Hf=fft(H,[],2);
Hf=Hf(:,1:floor(size(Hf,2)/2)+1);
f=f(1:size(Xf,2));


% Initial Cost
for i=1:size(W,1)
   Hft=Hf.*exp(T(i,:)'*f); 
   if mod(size(X,2),2)==0
       Hft=[Hft conj(Hft(:,end-1:-1:2))];
   else
       Hft=[Hft conj(Hft(:,end:-1:2))];
   end
   Ht=real(ifft(Hft,[],2));
   Rec(i,:)=W(i,:)*Ht;
end
cost=0.5*norm(X-Rec,'fro')^2
varexpl=(SST-2*cost)/SST;

if ~isempty(smoothtype)
    HL=H*L;
    if strcmp(smoothnorm,'L1') 
        smoothcost=lambda*sum(sum(abs(HL)));
    elseif ~isempty(smoothtype)
        smoothcost=lambda*0.5*norm(HL,'fro')^2;
    end 
end
cost_oldt=cost+smoothcost;
dcost=inf;
told=cputime;
iter=0;

% Display algorithm progress
if dispiter
    disp([' '])
    disp(['Shifted Non-negative matrix factorization'])
    disp(['A ' num2str(noc) ' component model will be fitted']);
    disp(['To stop algorithm press control C'])
    disp(['Smoothness by ' smoothtype ' using ' smoothnorm '-norm, imposed with strenght ' num2str(lambda)]);
    dheader = sprintf('%12s | %12s | %12s | %12s | %12s | %12s | %12s','Iteration','Expl. var.','Cost func.','Delta costf.',' Time(s)   ',' H-stepsize');
    dline = sprintf('-------------+--------------+--------------+--------------+--------------+--------------+');
    disp(dline);
    disp(dheader);
    disp(dline);
end


while iter<maxiter & (dcost>=cost*conv_crit | mod(iter,20)==0 ) 

    iter=iter+1;
    
    if mod(iter,10)==0 & dispiter
        disp(dline);
        disp(dheader);
        disp(dline);
    end
        
    % Update H
    if ~constH
        gradnH=zeros(size(Hf));
        gradpH=zeros(size(Hf));
        for i=1:size(Hf,2)
            Wf=(W.*exp(T*f(i)));
            gradnH(:,i)=Wf'*Xf(:,i);
            gradpH(:,i)=Wf'*Wf*Hf(:,i);
        end
        if mod(size(X,2),2)==0
        gradnH=[gradnH conj(gradnH(:,end-1:-1:2))];
        gradnH=real(ifft(gradnH,[],2));
        gradpH=[gradpH conj(gradpH(:,end-1:-1:2))];
        gradpH=real(ifft(gradpH,[],2));
        else            
            gradnH=[gradnH conj(gradnH(:,end:-1:2))];
            gradnH=real(ifft(gradnH,[],2));
            gradpH=[gradpH conj(gradpH(:,end:-1:2))];
            gradpH=real(ifft(gradpH,[],2));
        end
        gradnH(gradnH<0)=0;
        gradpH(gradpH<0)=0;
        if ~strcmp(smoothtype,'none')
            if strcmp(smoothnorm,'L1') 
                gradconstr=lambda*sign(HL).*repmat(sum(L,2)',[size(H,1),1]);
            else
                gradconstr=lambda*HL*L';
            end
            indp=find(gradconstr>0);
            indn=find(gradconstr<0);
            gradconp=zeros(size(H));
            gradconn=zeros(size(H));
            gradconp(indp)=gradconstr(indp);
            gradconn(indn)=-gradconstr(indn);          
            grad=(gradnH+gradconn)./(gradpH+gradconp+eps);
        else
            grad=(gradnH)./(gradpH+eps);
        end
        Hold=H;
        Recf=zeros(size(Xf));
        keepgoing=1;
        while keepgoing
             H=Hold.*(grad.^alpha);
             Hf=fft(H,[],2);
             Hf=Hf(:,1:floor(size(Hf,2)/2)+1);
             for i=1:size(W,1)
                Hft=Hf.*exp(T(i,:)'*f); 
                Recf(i,:)=W(i,:)*Hft;
             end    
             if mod(size(X,2),2)==0
                 cost=1/size(H,2)*(0.5*norm(Xf(:,[1 end])-Recf(:,[1 end]),'fro')^2+norm(Xf(:,[2:end-1])-Recf(:,[2:end-1]),'fro')^2);
             else
                  cost=1/size(H,2)*(0.5*norm(Xf(:,1)-Recf(:,1),'fro')^2+norm(Xf(:,[2:end])-Recf(:,[2:end]),'fro')^2);
             end
             if ~isempty(smoothtype)
                 HL=H*L;
                 if strcmp(smoothnorm,'L1')
                    smoothcost=lambda*sum(sum(abs(HL)));
                 else
                    smoothcost=lambda*0.5*norm(HL,'fro')^2;
                 end
                 cost=cost+smoothcost;
             end
             if cost>cost_oldt + 1e-2
                 alpha=alpha/2;
                 keepgoing=1;
             else
                 alpha=alpha*1.2;
                 keepgoing=0;
             end
         end
    end
    if ~isempty(smoothtype)
        cost=cost-smoothcost;
    end

    % Update W and calculate cost function
%      if ~constW | (~constH & constT) 
     if ~constW
        for i=1:size(W,1)
           Hft=Hf.*exp(T(i,:)'*f); 
           if mod(size(X,2),2)==0
               Hft=[Hft conj(Hft(:,end-1:-1:2))];
           else
               Hft=[Hft conj(Hft(:,end:-1:2))];
           end
           Ht=real(ifft(Hft,[],2));
           Ht(Ht<0)=0;
           gradnW(i,:)=X(i,:)*Ht';
           gradpW(i,:)=W(i,:)*(Ht*Ht');
        end
        if ~constW & ~strcmp(smoothtype,'none')
            tx = sum(gradpW.*W,1);
            ty = sum(gradnW.*W,1);
            gradnW = gradnW + repmat(tx,[size(W,1),1]).*W;
            gradpW = gradpW + repmat(ty,[size(W,1),1]).*W;                                                
            W=W.*(gradnW)./(gradpW+eps);
            W=W./repmat(sqrt(sum(W.^2,1)),[size(W,1),1]);
        else
            W=W.*(gradnW)./(gradpW+eps);
        end
        if  (~constH & constT) % Calculate Reconstruction if costfunction is needed
           for i=1:size(W,1)
               Hft=Hf.*exp(T(i,:)'*f); 
               if mod(size(X,2),2)==0
                   Hft=[Hft conj(Hft(:,end-1:-1:2))];
               else
                    Hft=[Hft conj(Hft(:,end:-1:2))];
               end
               Ht=real(ifft(Hft,[],2));                  
               Rec(i,:)=W(i,:)*Ht; 
           end
        end
    end

    % Update T and calculate cost
    
    if ~constT
        if mod(iter,20)==0 & iter<maxiter-20 & auto_corr
            disp('Reestimating shifts by cross-correlation')
            T=estTimeAutCor(Xf,W,Hf,T,f,smoothtype);
        end
        P.W=W;
        P.Xf=Xf;
        P.Hf=Hf;
        P.sizeX2=size(X,2);
        P.w=ones(1,length(f));
        if mod(size(X,2),2)==0
            P.w(2:end-1)=2;
        else
            P.w(2:end)=2;
        end
        P.f=f;
        P.nyT=nyT;
        [T,nyT,cost]=update_TmH(T,P);
        cost=cost/size(H,2); % Use Parseval identity for cost function
        sse=2*cost;
    elseif ~constW
        sse=norm(X-Rec,'fro')^2;
        cost=0.5*sse;
    else
        sse=2*cost;
    end
    if ~isempty(smoothtype)
        cost=cost+smoothcost;
    end
    
    % Display output of iteration
    dcost=cost_oldt-cost;
    cost_oldt=cost;
    costiter(iter)=cost;
    varexpl(iter)=(SST-sse)/SST;
    if rem(iter,1)==0 & dispiter
        t=cputime;
        tim=t-told;
        told=t;
        disp(sprintf('%12.0f | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f ',iter, varexpl(iter),cost,dcost,tim,alpha));
    end

end
cost=costiter;

% Align H and T
tmean=min(T);
T=T-repmat(tmean,[size(T,1),1]);
Hf=Hf.*exp(tmean'*f);
if mod(size(X,2),2)==0
    Hf=[Hf conj(Hf(:,end-1:-1:2))];
else
    Hf=[Hf conj(Hf(:,end:-1:2))];
end
H=real(ifft(Hf,[],2));
H(H<0)=0;


% -------------------------------------------------------------------------
% Parser for optional arguments
function var = mgetopt(opts, varname, default, varargin)
if isfield(opts, varname)
    var = getfield(opts, varname); 
else
    var = default;
end
for narg = 1:2:nargin-4
    cmd = varargin{narg};
    arg = varargin{narg+1};
    switch cmd
        case 'instrset',
            if ~any(strcmp(arg, var))
                fprintf(['Wrong argument %s = ''%s'' - ', ...
                    'Using default : %s = ''%s''\n'], ...
                    varname, var, varname, default);
                var = default;
            end
        otherwise,
            error('Wrong option: %s.', cmd);
    end
end

% -------------------------------------------------------------------------
function [T,W]=estTimeAutCor(Xf,W,Hf,T,f,smoothtype)
noc=size(W,2);
sX=2*size(Hf,2)-2;
Xft=Xf;
t1=randperm(size(Xf,1));
t2=randperm(noc);

for k=t1
  for d=t2
        nocm=setdiff(1:noc,d);
        Xft(k,:)=Xf(k,:)-W(k,nocm)*(Hf(nocm,:).*exp(T(k,nocm)'*f(1,:)));
        C=(conj(Xft(k,:)).*Hf(d,:));
        C=[C conj(C(end-1:-1:2))];
        C=real(ifft(C));
        [y,ind]=max(C);
        T(k,d)=(ind-sX)-1;
        if strcmp(smoothtype,'none')
            W(k,d)=C(ind)/((sum(2*(Hf(d,:).*conj(Hf(d,:))))-sum(Hf(d,[1 end]).*conj(Hf(d,[1 end]))))/sX);
        end
        if abs(T(k,d))>(sX/2)
            if T(k,d)>0
                T(k,d)=T(k,d)-sX;
            else 
                 T(k,d)=T(k,d)+sX;
            end
        end                  
  end
end

% -------------------------------------------------------------------------
function [T,nyT,cost]=update_TmH(T,P)
    nyT=P.nyT;
    Hf=P.Hf;
    W=P.W;
    sizeX2=P.sizeX2;
    Xf=P.Xf;
    f=P.f;
    w=P.w;
    Recfd=zeros(size(W,1),size(Hf,2),size(W,2));
    for d=1:size(W,2)
        Recfd(:,:,d)=(repmat(W(:,d),[1 length(f)]).*exp(T(:,d)*f)).*repmat(Hf(d,:),[size(W,1),1]);
    end
    Recf=sum(Recfd,3);        
    Q=Recfd.*repmat(conj(Xf-Recf),[1,1,size(Recfd,3)]);
    Hdiag=2*squeeze(sum(repmat((w.*(f.^2)),[size(Q,1),1,size(Q,3)]).*real(Q),2));    
    Hall=zeros(size(W,1),size(W,2),size(W,2));
    for d=1:size(W,2)
        Hall(:,:,d)=2*squeeze(sum(repmat((w.*(f.^2)),[size(Q,1),1,size(Q,3)]).*real(Recfd.*repmat(conj(Recfd(:,:,d)),[1,1,size(W,2)])),2));
    end    
    grad=squeeze(sum(repmat((w.*f),[size(Q,1),1,size(Q,3)]).*(conj(Q)-Q),2)); 
    for i=1:size(grad,1)
        ind=find(abs(grad(i,:))>1e-6);
        grad(i,ind)=grad(i,ind)/(-diag(Hdiag(i,ind))-squeeze(Hall(i,ind,ind)));
    end

    ind1=find(w==2); % Areas used twice
    ind2=find(w==1); % Areas used once
    cost_old=norm(Xf(:,ind1)-Recf(:,ind1),'fro')^2;
    cost_old=cost_old+0.5*norm(Xf(:,ind2)-Recf(:,ind2),'fro')^2;
    keepgoing=1;
    Told=T;
    while keepgoing
        T=Told-nyT*grad;
        for d=1:size(W,2)
            Recfd(:,:,d)=(repmat(W(:,d),[1 length(f)]).*exp(T(:,d)*f)).*repmat(Hf(d,:),[size(W,1),1]);
        end
        Recf=sum(Recfd,3);   
        cost=norm(Xf(:,ind1)-Recf(:,ind1),'fro')^2;
        cost=cost+0.5*norm(Xf(:,ind2)-Recf(:,ind2),'fro')^2;
        if cost<=cost_old
            keepgoing=0;
            nyT=nyT*1.2;
        else
            keepgoing=1;
            nyT=nyT/2;
        end
    end
    T=mod(T,sizeX2);
    ind=find(T>floor(sizeX2/2));
    T(ind)=T(ind)-sizeX2;
    
    
