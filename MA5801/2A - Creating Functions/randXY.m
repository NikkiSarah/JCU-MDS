%creating a function, randXY, that will generate a random set of data
function [x,y] = randXY(N)
x = rand(N,1)
y = 5*x + 3 + 2*randn(N,1)
end
