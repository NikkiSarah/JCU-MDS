%% Assessment 4 - Principal Component Analysis
% 13848336 Nikki Fitzherbert

%% Introduction
% The code in this assessment implements a principal component analysis
% using singular value decomposition, using a subset of the dataset
% aggregated for the 2017 Sustainable Infrastructure for the Tropics
% report.
%
% More information about the dataset can be found at: https://www.jcu.edu.au/state-of-the-tropics

%% Preparatory work
% cleaning the workspace
clear all
clc

% loading in the data and determining the its dimensions
[ndata, text, alldata] = xlsread('SotTCombined2010.xlsx');
Xtild = ndata;

sum(isnan(Xtild))
sum(isnan(Xtild), 'all')

% removing missing values
Xtild_nomissing = rmmissing(Xtild);

% centring and scaling the data to create matrix X
X = (Xtild_nomissing - mean(Xtild_nomissing)) ./ std(Xtild_nomissing,1);

%% The Principal Component (PC) vectors and proportion of variation
% performing the SVD; that is, decomposing matrix X into its three
% component matrices
[U,S,V] = svd(X);

% V is a matrix with orthonormal rows and shows the relationships between
% the 13 indicators; that is, the columns of V are the right singular
% vectors of matrix X (or the PC vectors)
V;

% extracting the first two PC vectors
PCvecs = [V(:,1:2)]'

% S is a diagonal matrix showing the singular values of matrix X, and
% ordered from largest to smallest
S;
diag(S)';

% calculating the eigenvalues of matrix X. These are used to calculate the
% proportion of variation in the dataset due to each PC vector.
eigvals = diag(S).^2
prop_var = eigvals / sum(eigvals)

eigvals_props = [diag(S)'; eigvals'; prop_var']

% calculating and graphically displaying the cumulative proportion of
% variation explained
cumprop_var = cumsum(prop_var)

pareto(prop_var)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

saveas(gcf, 'vars_plot', 'png')

%% The matrix of scores
% deriving the matrix of scores for each country/nation in the dataset.
% These are the principal components.
T = U*S
T2 = X*V;

%% Truncating the matrix
% deriving the reduced matrix by retaining only those PCs cumulatively
% explaining more than 80% of the variation in the data
k = 4;
Uhat = U(:, 1:k);
Shat = S(1:k, 1:k);
Vhat = V(:, 1:k);

Xhat = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)'

% comparing the reduced matrix to the original dataset. If a case has a
% large squared prediction error then it isn't well described by the k PCs
% retained in Xhat
SPE = sum((X-Xhat).^2,2);
SPE_vec = SPE'




