%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is to fit the normal-incidence reflection coefficient of a material using vector fitting, which is the needed input for the DG simulation (more details in the boundary condition class of the API documentation and references therein)
%
%
% Copyright 2024 Huiqing Wang
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all

% --------------------
% Block 1: User input
% --------------------

c0 = 343; % Speed of sound in air (m/s)
rho0 = 1.213; % Density of air (Pa)
z0 = c0 * rho0; % Characteristic impedance of air

savename= 'carpet'; % filename to save the results of the fitting for the interested material, this file will be loaded in the DG simulation, make sure the filename is consistent
%%
%Frequency samples:
f_min =20;  % lower bound of frequency range
f_max = 2000; % upper bound of frequency range
frequency_step=10; % frequency step
Niter=20; % number of iterations for vector fitting, 20 is prefered based on experience
N =5; %  number of poles for the approximation
freq = [f_min:frequency_step:f_max]; % frequency range

%%  below is an example of the impedance (Miki model) to be approximated
sigma = 100000; % flow resistivity
x = freq/sigma;
Z_Miki = z0*( 1 + 5.50*(x*1000).^(-0.632) - 1i*8.43*(x*1000).^(-0.632) );
%% -------------------- -------------------- -------------------- --------------------


Zs_to_fit=Z_Miki; % impedance values of materials to be fitted



% --------------------------------------------------------------------------------
% Block 2: Use Vector Fitting to approximate the reflection Coefficients
% --------------------------------------------------------------------------------


ReflectionCoefficient= (Zs_to_fit/z0 -1) ./ (Zs_to_fit/z0 +1);

Ns=length(freq);
omega = 2*pi*freq;
s = 1i*omega;
opts.asymp=1;      % 1->there is no constant term in the approximation
% if opts.asymp = 2, there is a constant term (let VF finds)in the approximation
opts.cmplx_ss=1;   %0->Create real-only state space model (error in document)

%% Vector fitting algorithm
% Real starting poles :
poles=-linspace(freq(1),freq(end),N);

%Parameters for Vector Fitting :

weight=ones(1,Ns);
opts.relax=1;      %Use vector fitting with relaxed non-triviality constraint
opts.stable=1;     %Enforce stable poles

opts.skip_pole=0;  %Do NOT (0) skip pole identification
opts.skip_res=1;   %DO skip identification of residues (C,D,E), except in the last iteration
opts.spy1=0;       %No plotting for first stage of vector fitting
opts.spy2=1;       %Create magnitude plot for fitting of f(s)
opts.logx=0;       %Use linear abscissa axis
opts.logy=1;       %Use logarithmic ordinate axis
opts.errplot=0;    %Include deviation in magnitude plot
opts.phaseplot=0;  %Do NOT produce plot of phase angle
opts.legend=1;     %Include legends in plots

disp('vector fitting...')

for iter=1:Niter
    if iter==Niter, opts.skip_res=0;end
    disp(['   Iter ' num2str(iter)])
    [SER,poles,rmserr,fit]=vectfit3(ReflectionCoefficient,s,poles,weight,opts);
    rms(iter,1)=rmserr;
end
rms=max(abs(rms))

disp('Resulting state space model:');
A=full(SER.A);
B=SER.B;
C=SER.C;
D=SER.D;
E=SER.E;
rmserr ;


% ------------------------------------------------------------
% Block 3: Extract fitting coefficients and Save to file
% ------------------------------------------------------------
ireal = find(imag(diag(A))~=0,1,'first');
if find(diag(A)~=poles.')
    error('there exists complex pole, real pole alone is not sufficient')
end
Zapprox=zeros(1,length(omega));

if isempty(ireal)
    
    lambdaS = zeros(1,N);
    for iloop=1:N
        lambdaS(iloop)=-A(iloop,iloop);
    end
    AS = C(1:N);
    
else
    
    BS = zeros(1,(length(A)-ireal+1)/2);
    CS = zeros(1,(length(A)-ireal+1)/2);
    
    alphaS = zeros(1,(length(A)-ireal+1)/2);
    betaS = zeros(1,(length(A)-ireal+1)/2);
    
    lambdaS = zeros(1,ireal-1);
    for iloop=1:ireal-1
        lambdaS(iloop)=-A(iloop,iloop);
    end
    AS = C(1:ireal-1);
    
    for iloop = ireal:2:length(A)
        alphaS((iloop-ireal)/2+1) = -real(A(iloop,iloop));
        betaS((iloop-ireal)/2+1) = -imag(A(iloop,iloop));
        
        BS((iloop-ireal)/2+1) = real(C(iloop));
        CS((iloop-ireal)/2+1) = imag(C(iloop));
    end
    
end



if exist('BS')
    % exist complex pole
    save(['./',savename,'.mat'], ...
        'AS','lambdaS','BS','CS','alphaS','betaS' )
else
    save(['./',savename,'.mat'], ...
        'AS','lambdaS','freq' )
end

