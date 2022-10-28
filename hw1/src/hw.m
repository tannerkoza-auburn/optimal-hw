%% Optimal Estimation - Homework 1
% AUTHOR: Tanner Koza

clear
clc
close all

%% Problem 1
% STATEMENT: 
% Use the MATLAB convolve function to produce discrete
% probability functions (PDFs) for throws of six dice as follows
% (note: this is effectively the sum of 6 random variables)
% FIND: 
% a) 6 numbered 1,2,3,4,5,6
% b) 6 numbered 4,5,6,7,8,9
% c) 6 numbered 1,1,3,3,3,5
% d) 3 numbered 1,2,3,4,5,6 and 3 numbered 1,1,3,3,3,5

% Dice Definition
di1 = [1 2 3 4 5 6];
di2 = [4 5 6 7 8 9];
di3 = [1 1 3 3 3 5];

% Probability Mass Function/Discrete RV Definition
pmf1 = ones(1,6) * (1/6);
pmf2 = ones(1,6) * (1/6);
pmf3 = [(1/3) 0 (1/2) 0 (1/6) 0];

% Initialize PDF for Calculation
pdf1 = pmf1;
pdf2 = pmf2;
pdf3 = pmf3;
pdf4 = pmf1;

% Discrete PDF Calculations
for i = 1:5

    pdf1 = conv(pdf1,pmf1);

    pdf2 = conv(pdf2,pmf2);

    pdf3 = conv(pdf3,pmf3);

    if i < 3
        pdf4 = conv(pdf4,pmf1);
    else
        pdf4 = conv(pdf4,pmf3);
    end

end

% Remove Zeros PDFs
pdf3 = nonzeros(pdf3)';
pdf4 = nonzeros(pdf4)';

% PDF Sums
a_sum = sum(pdf1);
b_sum = sum(pdf2);
c_sum = sum(pdf3);
d_sum = sum(pdf4);

% Possible Outcome Calculations
a_out = unique(sum(setprod(di1, di1, di1, di1, di1, di1), 2));
b_out = unique(sum(setprod(di2, di2, di2, di2, di2, di2), 2));
c_out = unique(sum(setprod(di3, di3, di3, di3, di3, di3), 2));
d_out = unique(sum(setprod(di1, di1, di1, di3, di3, di3), 2));

% Mean Calculations
% NOTE: Cannot use mean(out) because the distribution might be skewed.
% For example, the distributions for Part C & D fall under this category.
a_mean = pdf1 * a_out; 
b_mean = pdf2 * b_out;
c_mean = pdf3 * c_out;
d_mean = pdf4 * d_out;

% Sigma Calculations
a_sigma = sqrt(pdf1 * (a_out - a_mean).^2);
b_sigma = sqrt(pdf2 * (b_out - b_mean).^2);
c_sigma = sqrt(pdf3 * (c_out - c_mean).^2);
d_sigma = sqrt(pdf4 * (d_out - d_mean).^2);

% Normal Distribution Creation
n = 100000; % number of random samples

a_dist = a_mean + a_sigma .* randn(n,1);
b_dist = b_mean + b_sigma .* randn(n,1);
c_dist = c_mean + c_sigma .* randn(n,1);
d_dist = d_mean + d_sigma .* randn(n,1);

% Histogram Creation
[a_N, a_edges] = histcounts(a_dist, 'Normalization', 'pdf');
[b_N, b_edges] = histcounts(b_dist, 'Normalization', 'pdf');
[c_N, c_edges] = histcounts(c_dist, 'Normalization', 'pdf');
[d_N, d_edges] = histcounts(d_dist, 'Normalization', 'pdf');


% Plotting
figure
histogram('BinEdges',a_edges,'BinCounts',a_N)
grid
hold on
stem(a_out,pdf1, 'r')
title('6 Normal Dice Discrete PDF')
xlabel('Possible Outcomes')
ylabel('Probability')
legend('Sampled PDF', 'Discrete PDF','Location','northwest')

figure
histogram('BinEdges',b_edges,'BinCounts',b_N)
grid
hold on
stem(b_out,pdf2,'r')
title('6 Dice (4-9) Discrete PDF')
xlabel('Possible Outcomes')
ylabel('Probability')
legend('Sampled PDF', 'Discrete PDF','Location','northwest')

% NOTE: The PDF doesn't match the corresponding normal distribution as
% this PDF is essentially a skewed version of a normal distribution. 
% The probabilities of outcomes closer to the mean are greater than what
% would provide a normal distribution. Therefore, comparing random normal
% samples to the PDF would show some discrepancies.
figure
histogram('BinEdges',c_edges,'BinCounts',c_N)
grid
hold on
stem(c_out,pdf3,'r')
title('6 Funny Dice Discrete PDF')
xlabel('Possible Outcomes')
ylabel('Probability')
legend('Sampled PDF', 'Discrete PDF','Location','northwest')

figure
histogram('BinEdges',d_edges,'BinCounts',d_N)
grid
hold on
stem(d_out,pdf4,'r')
title('3 Normal & 3 Funny Dice Discrete PDF')
xlabel('Possible Outcomes')
ylabel('Probability')
legend('Sampled PDF', 'Discrete PDF','Location','northwest')

clearvars

%% Problem 2
% STATEMENT: 
% What is the joint PDF for 2 fair dice (x1, x2) (make this a 6x6 matrix
% with the indices equal to the values of the random variables). 
% Note each row should add to the probability of the index for x1 and
% each column to the probability of the index for x2
% FIND:
% a) What are E(X1), E(X1-E(X1)), E(X1^2), E((X1-E(X1))^2), 
%                           and E(((X1-E(X1))*(X2-E(X2)))
% b) Form the covariance matrix for x1 and x2
% c) Now find the PDF matrix for the variables v1=x1 and v2=x1+x2.
% d) Now what is the mean, E(v1-E(v1)), rms, and variance of v1
% e) What is the mean, E(v2-E(v2)), rms and variance of v2
% f) What is the new covariance matrix P.

% Dice Definition
X1 = [1 2 3 4 5 6];
X2 = X1';

% Individual Di PDF Definition
pdf = ones(1,6) * (1/6);
len_pdf = length(pdf);

% Joint PDF (JPDF) Calculation
j_pdf = reshape(prod(setprodnu(pdf,pdf),2),[len_pdf len_pdf]);

% Marginal Density (Distribution) of JPDF Calculation
% NOTE: The Marginal Density is the PDF for X1 given any possible outcome
% in the PDF of X2 or vice versa. In other words, it's the provides the
% probability of each X1 element for any X2 element or vice versa.
mdX1 = sum(j_pdf,2); 
mdX2 = sum(j_pdf);

% Expectation Calculations
eX1 = X1 * mdX1; % expectation or mean of x1 E(X1) 

eX2 = mdX2 * X2; % expectation or mean of x2 E(X2)

zmX1 = floor( (X1 - eX1) * mdX1 ); % zero mean x1 E(X1-E(X1))

eX1sq = X1.^2 * mdX1; % sum of variance and mean squared E(X1^2)

varX1 = (X1 - eX1).^2 * mdX1; % variance of x1 % E((X1-E(X1))^2) 

varX2 = mdX2 * (X2 - eX2).^2; % variance of x2 % E((X2-E(X2))^2)

covX1X2 = X1 * j_pdf * X2 - (eX1 * eX2); % covariance of x1 and x2
                                         % E(((X1-E(X1))*(X2-E(X2)))

% Covariance Matrix Construction
P_X1X2 = [varX1 covX1X2;
        covX1X2 varX2];

clearvars j_pdf

% Dependent Dice Definition
V1 = X1;
V2 = [2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12];

% Joint PDF (JPDF) Calculation
j_pdf = [(1/36)*ones(1,6) zeros(1,5);
        0 (1/36)*ones(1,6) zeros(1,4);
        zeros(1,2) (1/36)*ones(1,6) zeros(1,3);
        zeros(1,3) (1/36)*ones(1,6) zeros(1,2);
        zeros(1,4) (1/36)*ones(1,6) 0;
        zeros(1,5) (1/36)*ones(1,6)];

% Marginal Density (Distribution) of JPDF Calculation
mdV1 = sum(j_pdf,2);
mdV2 = sum(j_pdf);

% Expectation Calculations
eV1 = V1 * mdV1;
eV2 = mdV2 * V2;

zmV1 = floor( (V1 - eV1) * mdV1 );
zmV2 = ceil( mdV2 * (V2 - eV2) );

rmsV1 = sqrt(V1.^2 * mdV1);
rmsV2 = sqrt(mdV2 * V2.^2);

varV1 = (V1 - eV1).^2 *mdV1; 
varV2 = mdV2 * (V2 - eV2).^2;

covV1V2 = V1 * j_pdf * V2 - (eV1 * eV2);

% Covariance Matrix Construction
P_V1V2 = [varV1 covV1V2;
          covV1V2 varV2];

clearvars

%% Problem 4

di = [-2.5 -1.5 -0.5 0.5 1.5 2.5];

pmf = (1/6) * ones(1, 6);

pdf = conv(pmf, pmf);

out = unique(sum(setprod(di, di), 2));

edi = ceil(pdf * out);
vardi = pdf * (out - edi).^2;

%% Problem 6

P = [2 1; 1 4];

rho = P(1)/sqrt(P(1)*P(4));

[V, s] = eig(P);

A = (s^-0.5) * V;

theta = 0:360;
a = [cosd(theta); sind(theta)];
a2 = 0.25 .* a;
a3 = 1.5 .* a; 

b = A\a;
b2 = A\a2;
b3 = A\a3;

scale = 5;

figure
plot(b(1,:),b(2,:),b2(1,:),b2(2,:), b3(1,:),b3(2,:))
hold on
plot(scale * [V(1,1) -V(1,1)], scale * [V(1,2) -V(1,2)])
plot(scale * [V(2,1) -V(2,1)], scale * [V(2,2) -V(2,2)])


%% Problem 7
% STATEMENT: Given x ~ N(0, sigmax^2) and y = 2x^2
% FIND: 
% a) Find the PDF of y 
% b) Draw the PDFs of x and y on the same plot for sx=2.0 
% c) How has the density function changed by this transformation 
% d) Is y a normal random variable? 

% Standard Deviation
sigma = 2;
 
% Possible Outcomes
x = -4*sigma:0.1:4*sigma;
dy = 0.00000001;
y = dy:0.001:4*50;

x_pdf = normpdf(x,0,sigma);
y_pdf = abs( (1/4) * sqrt (1./(y/2)) ) .* (1/sqrt(2*pi*sigma^2)) .* exp( (-1/2) .* ( (y/2)./sigma^2 ));

figure
plot(x,x_pdf,y,y_pdf)

xlim([0 20*sigma])

n = 100000;

x_dist = sigma * randn(n,1);
y_dist = 2 * (x_dist.^2);

[x_N, x_edges] = histcounts(x_dist, 'Normalization', 'pdf');
[y_N, y_edges] = histcounts(y_dist, 'Normalization', 'pdf');

figure
histogram('BinEdges', x_edges, 'BinCounts', x_N)
hold on
plot(x,x_pdf)
figure
histogram('BinEdges', y_edges, 'BinCounts', y_N)
hold on
plot(y,y_pdf)


