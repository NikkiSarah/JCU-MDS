filename = imread('prague-astronomical-clock-detail-871291743639AGq.jpg');
% filename = imread('boat-in-caribbean-14884763094mZ.jpg');
Nretain = [5,10,50,100];

SVDcompress(filename, Nretain);