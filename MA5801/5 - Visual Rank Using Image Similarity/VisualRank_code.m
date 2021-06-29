%% Assessment 5 - VisualRank using image similarity
% 13848336 Nikki Fitzherbert
%
%% Introduction
% The following code demonstrates how a version of Google's PageRank
% algorithm can be applied to images (the 'VisualRank' algorithm). The
% algorithm makes uses of 'visual hyperlinks' between images to determine
% which image is most representative of the set provided as input.
%
% The code is divided up into three parts:
%  Part 1 ranks all 1400 images
%  Part 2 ranks only the heart-shaped images
%  Part 3 searches for images similar to a given image before refining the
%    results using the algorithm

%% Part 1: Ranking all 1400 shape images using VisualRank
% ensuring the workspace is clear before starting
clear all
clc

% loading the data file into matlab. This file contains a similarity
% matrix 'S' and a cell array of corresponding filenames
load('sim.mat');

% using 'S' to form the adjacency matrix 'A'
A = S;
% retaining only positively correlated visual hyperlinks between the images
A(A <= 0) = 0;
% removing elements corresponding to loop edges
A = A - diag(diag(A));

% using 'A' to create the hyperlink matrix 'H'
H = zeros(1400,1400);
[row, col] = size(A);
for i = 1:col
   H(i,:) = A(i,:) / sum(A(i,:));
end

% forming the random jump matrix 'J' where 'N' is the total number of
% images
N = row;
J = ones(row,col) / N;

% creating the modified visual hyperlink matrix with a dampening factor of
% d = 0.85
d = 0.85;
Htilde = d * H + (1 - d) * J;

% determining the VisualRank vector 'r' using an initial vector 'v0'
% setting the parameters required for the iteration including
% a break point
v0 = [0,1,zeros(1,1398)];
% v0 = ones(1,col) / N;
r(1,:) = v0 * Htilde;

for n = 2:50
    r(n,:) = v0 * (Htilde^n);
    % setting a break so the loop stops when the difference between two
    % successive iterations is marginal i.e. a steady-state is reached
    if(all(abs(r(n,:) - r(n-1,:)) <= 0.0001))
        break;
    end
end

rvec = r(end,:)*100;

% determining which image is ranked highest and displaying this image
[~,I] = max(rvec);
imshow(filenames{I});

%% Part 2: Ranking the 20 heart-shaped images using VisualRank
% clearing the workspace
clear all
clc

% the next four lines to create the full adjacency matrix are exactly the
% same as in the previous part (lines 23 to 30) so the detailed commentary
% has not been repeated
load('sim.mat');
A = S;
A(A <= 0) = 0;
A = A - diag(diag(A));

% making a smaller adjacency matrix using just the rows and columns
% corresponding to the heart-shaped images
A = A(81:100, 81:100);

% forming the corresponding visual hyperlink matrix and finding the
% VisualRank vector
% this code is (almost) exactly the same as in the previous part
% (lines 33 to 47) so the detailed commentary has not been repeated
[row, col] = size(A);
H = zeros(20,20);
for i = 1:col
   H(i,:) = A(i,:) / sum(A(i,:));
end

N = row;
J = ones(row,col) / N;

d = 0.85;
Htilde = d * H + (1 - d) * J;

% adjusting the starting vector for the smaller 20x20 matrix
v0 = [0,1,zeros(1,18)];
r(1,:) = v0 * Htilde;

for n = 2:50
    r(n,:) = v0 * (Htilde^n);
    % setting a break so the loop stops when the difference between two
    % successive iterations is marginal
    if(all(abs(r(n,:) - r(n-1,:)) <= 0.0001))
        break;
    end
end

rvec = r(end,:)*100;

% determining which heart-shaped image is most representative of the 20
% given heart-shaped images and displaying this image
[~,Imax] = max(rvec);
imshow(filenames{Imax + 80})
title('Most representative image')

% determining which heart-shaped image is the least representative (least
% 'heart-shaped') and displaying this image
[~,Imin] = min(rvec);
figure, imshow(filenames{Imin + 80});
title('Least representative image')

% ... or diplaying the two images in the same figure/array
figure

subplot(1,2,1)
imshow(filenames{Imax + 80})
title('Most representative image')

subplot(1,2,2)
imshow(filenames{Imin + 80})
title('Least representative image')

%% Part 3: Searching for similar images and refining the results using Visual Rank
% clearing the workspace
clear all
clc

% the next four lines to create the full adjacency matrix are exactly the
% same as in the previous part (lines 23 to 30) so the detailed commentary
% has not been repeated
load('sim.mat');
A = S;
A(A <= 0) = 0;
A = A - diag(diag(A));

% forming a new adjacency matrix for which the elements are the reciprocal
% of those in the original adjacency matrix 'A', and forming the graph 'G'
% corresponding to this new adjacency matrix
Arec = 1 ./ A;

G = digraph(Arec);
GG = digraph(Arec, 'omitselfloops');

% finding the 10 images nearest to 'device6-18.png'
position = find(strcmp(filenames, 'device6-18.png'));

[nodeIDs, dist] = nearest(G, position, 1.5);
top10 = nodeIDs(1:10);

% ranking the 10 nearest images using VisualRank and displaying them on a
% figure in their ranked order

% making a smaller adjacency matrix using just the rows and columns
% corresponding to the images identified by line 176 above
A = A(top10, top10);

% forming the corresponding visual hyperlink matrix and finding the
% VisualRank vector
% this code is (almost) exactly the same as in part A
% (lines 33 to 47) so the detailed commentary has not been repeated
[row, col] = size(A);
H = zeros(10,10);
for i = 1:col
   H(i,:) = A(i,:) / sum(A(i,:));
end

N = row;
J = ones(row,col) / N;

d = 0.85;
Htilde = d * H + (1 - d) * J;

% adjusting the starting vector for the smaller 10x10 matrix
v0 = [0,1,zeros(1,8)];
r(1,:) = v0 * Htilde;

for n = 2:50
    r(n,:) = v0 * (Htilde^n);
    % setting a break so the loop stops when the difference between two
    % successive iterations is marginal
    if(all(abs(r(n,:) - r(n-1,:)) <= 0.0001))
        break;
    end
end

rvec = r(end,:)*100;

% attaching the filename indexes onto 'r' and sorting into descending order
% (so highest ranked image is first etc)
r_ind = cat(1, top10', rvec);
r_ind = sort(r_ind, 2, 'descend');
r_ind_pos = r_ind(1,:)';

figure

%subplot(2,7,[1,9]);
%imshow(filenames{position})
%title('queried image')

subplot_i = 1;
for j = 1:5
    subplot(2,5,subplot_i);
    imshow(filenames{r_ind_pos(j)})
    title(['Rank = ', num2str(j)])
    
    subplot_i = subplot_i + 1;
end

subplot_ii = 6;
for jj = 6:10
    subplot(2,5,subplot_ii);
    imshow(filenames{r_ind_pos(jj)})
    title(['Rank = ', num2str(jj)])
    
    subplot_ii = subplot_ii + 1;
end