%% Initialisation
clear all
rng(1)

%% Testing the data generation:
% You should find the values,
%
% x =
%     0.4170
%     0.7203
% y =
%     3.5682
%     4.3824

[x,y] = randXY(2)

%% Testing the slope calculation:
% You should find the slope of
% slope = 2.6844

slope = findSlope(x,y)

%% Doubling the y input should double the slope.
% You should find:
% ratio = 2

slope2 = findSlope(x,2*y);

ratio = slope2 / slope

%% Try out a larger set of data.

[x,y] = randXY(100);
slope = findSlope(x,y);

disp('Slope of 100 data points is:')
slope

%% And to show the data we include this last piece.
figure
plot(x,y,'o')
hold all
coeffs = polyfit(x,y,1);
x = sort(x);
plot(x,polyval(coeffs,x))
