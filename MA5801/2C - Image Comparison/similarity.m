%% Similarity comparison with SVD
% In the image compression example, we learned that an SVD can break a
% matrix up into "singular components". Here we make use of those
% components as something like a "basis" for the matrix.
% 
% To compare two images we perform an SVD on the matrix formed from both
% images. The right-singular vectors represent one basis and the
% left-singular values another basis.
% 
% The "overlap" between two singular components, that is a component from A
% and a component from B, can be chosen to be:
% SA*SB * (VA'VB) * (UA'UB)
% The first part is made up of the singular values - these are the "amount" of each
% component from A and B.
% The second part is the "overlap" or dot-product between the two right
% singular values.
% The third part is the "overlap" or dot-product between the two left
% singular values.
% 
% The above expression is applicable to a *single* singular component from
% each matrix. To include the effect of all singular components we must add
% up each of those terms for every component i from A and j from B.
% Something like:
% \sum_ij  SA_i * SB_j * (VA_i' VB_j) * (UA_i' UB_j)
% 
% We can show that, if A and B were both the same image, this expression
% would reduce to:
% \sum_i SA_i^2
% We can use this to "normalise" the result. That is, we can divide through
% by this to make the maximum value of the comparison equal to one.

%% Implementation
% The implementation of the image comparison is given in compareImages.m
% 
% This implementation requires the images to be of the same size. We have
% done this by resizing all of our test images beforehand.

%% Comparison between like images
% In order to test the compareImages function, we can apply it to two cases
% where we expect to have a high degree of similarity.
% a) The same image.
% b) An image showing the same type of object.
% Here we choose the apple image. We expect a) to return 1, due to the
% normalisation and b) to return a large value.

imA = imread('apple-9.png');

disp('The similarity value for the same image is:')
compareImages(imA,imA)

imB = imread('apple-13.png');

disp('The similarity value for the two apple images is')
compareImages(imA,imB)

%% Mass comparison
% We can now compare all of the test images to one another.

% We first obtain the full file list
filelist = dir('*.png')
N = length(filelist)

% filenames = {}
 for i = 1:length(filelist)
     filenames{i} = filelist(i).name;
 end

% We will store each comparison into a matrix, so that each row/column
% corresponds to a particular image.
mat = zeros(N);

% We now loop over all of the files twice, so that we go through each
% element of the matrix.
for iA = 1:N
    fileA = filelist(iA);
    imA = imread(fileA.name);

    for iB = iA:N
        fileB = filelist(iB);

        imB = imread(fileB.name);
        
        val = compareImages(imA,imB);
        
        mat(iA,iB) = val;
        mat(iB,iA) = mat(iA,iB);
    end
end

%% Mass comparison results
% After obtaining the comparison between all of the images, we can plot
% this.

imagesc(mat)
colorbar

% We also output all of the filenames and their corresponding number.
for i = 1:N
    disp(['i = ', num2str(i), ' is ', filelist(i).name])
end

% You should be able to see a couple of things:
% 1) All of the diagonal elements are 1. This is because we normalised the
% comparison to be maximal when the images are the same.
% 2) The next-largest elements are the two clusters for images 3,4 and 5,6.
% The filename list shows that these correspond to the pair of apples and
% the pair of hammers. This makes sense - the same shaped objects compare
% well.
% 3) There is some vague similarity between the apple images and the hat,
% teddy images. These images have similar shapes so it is not surprising
% that some level of comparison can be made.

%% Further use
% Please keep a hold of this similarity comparison. You will need this to
% complete the project in week 4.
