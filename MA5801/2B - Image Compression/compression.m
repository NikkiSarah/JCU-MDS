%% Introduction
% This worked example aims to demonstrate how we can perform image
% compression by using the singular value decomposition.
% 
% As we have already learned, the SVD breaks a matrix up into sets of
% singular value components, which each consist of
% a) one number which is the singular value
% b) a vector which is the left singular vector
% c) a vector which is the right singular vector
% The idea of image compression is to use the image as a matrix. We can
% then throw away the "least significant" singular value components.
% 
% This exercise runs with the following steps:
% 1) Load in some image data.
% 2) Extra a greyscale representation as a matrix.
% 3) Find the SVD.
% 4) Truncate the singular components.
% 5) Demonstrate the filesize changes.
% 
% We end with setting you a little task to extend the compression to a full
% colour image.

%% Loading the data
% Matlab supports loading images via a right-click menu in the "Current
% folder". Here we show how it can be done programmatically.

% There are two example images. We choose the caribbean image, but you may
% like to uncomment the clock image for comparison.
%pic = imread('prague-astronomical-clock-detail-871291743639AGq.jpg');
pic = imread('boat-in-caribbean-14884763094mZ.jpg');

% The image is loaded in as an integer format. Convert to floating point
% so SVD can be taken.
pic = double(pic);
% The loaded data is in the range from 0-255. Convert to the range 0-1.
pic = pic / 255;

%% Converting to greyscale
% The loaded data is a 3-dimensional "matrix". The red, green and blue
% components of the image are accessed with the last dimension.
red = pic(:,:,1);
green = pic(:,:,2);
blue = pic(:,:,3);

% To make a greyscale image we can take the average of this last dimension.
greyscale = mean(pic, 3);

% We now show what each of these parts look like.
figure
colormap gray

subplot(2,2,1)
imagesc(red)
axis equal tight off
title('Red')

subplot(2,2,2)
imagesc(green)
axis equal tight off
title('Green')

subplot(2,2,3)
imagesc(blue)
axis equal tight off
title('Blue')

subplot(2,2,4)
imagesc(greyscale)
axis equal tight off
title('Greyscale')

%% Perform the SVD
% This is a simple process in Matlab, perhaps the easiest part of the
% exercise.
[U,S,V] = svd(greyscale);

%% Compare with original
% By multiplying the matrices back together, we confirm that the
% decomposition has not lost any information in the image.

reconstructed = U*S*V';
figure
colormap gray
subplot(1,2,1)
imagesc(greyscale)
title('Original')
subplot(1,2,2)
imagesc(reconstructed)
title('Reconstructed')

%% Compare with fewer singular values
% Now we will "chop off" some of the parts of the individual matrices, U,
% S and V.

figure
colormap gray

subplot_i = 1;
for Nretain = [1, 2, 5, 10, 20, 30, 50, 100, size(S,1)]
    U2 = U(:,1:Nretain);
    V2 = V(:,1:Nretain);
    S2 = S(1:Nretain,1:Nretain);
    
    reconstructed = U2*S2*V2';
    
    % Here we work out a guess at the memory requirement.
    % This is a very simple estimate, and neglects the possibility of other
    % types of compression on top of the image data.
    %
    % The raw byte requirement is 8 bytes for each float, and 1+N+M floats
    % for each singular vector, where N and M are the sizes of the matrix.
    % We divide by 1024^2 to convert to MB
    memrequired = Nretain * (1 + size(U,1) + size(V,1)) * 8 / 1024^2;
    memrequired = round(memrequired, 2);
    
    subplot(3,3,subplot_i)
    imagesc(reconstructed)
    axis equal tight off
    title(['N = ', num2str(Nretain), ' Mem = ', num2str(memrequired)])
    
    subplot_i = subplot_i + 1;
end

%% What do the first few singular values look like?

figure
colormap gray
for ind = 1:9
    contribution = U(:,ind) * V(:,ind)';
    contribution = abs(contribution) / max(abs(contribution(:)));
    
    subplot(3,3,ind)
    imshow(contribution)
    axis equal tight off
    title(['i = ', num2str(ind), ' contribution']);
end

%% What would the compression look like?

% Save the full greyscale image matrix to file. We turn off Matlab mat file
% compression (which also requires the v7.3 format) to properly compare the
% SVD compression.
save full_greyscale.mat greyscale -v7.3 %-nocompression

% Consider 50 singular values only:
Nretain = 50;
U2 = U(:,1:Nretain);
V2 = V(:,1:Nretain);
S2 = S(1:Nretain,1:Nretain);
S2 = diag(S2);

% Save this information to file:
save compressed_greyscale.mat U2 V2 S2 -v7.3 %-nocompression

% You should now be able to see the difference between the two files.

%% Some comments
% Note that the SVD compression we present here cannot compete with
% sophisticated image compression techniques. This is mostly due to our
% represention using a matrix: we focus on individual pixels rather than
% image features.
%
% However, the compression we have just investigated is a demonstration of
% the utility of the SVD. It helps to pick out dominant features from a
% collection of data. This image compression example is useful because it
% is easy to visually see what is happening. In other, perhaps more useful,
% applications of the SVD, the process is often not as easy to visually
% identify.

