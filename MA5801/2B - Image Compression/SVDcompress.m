%% Assessment 2B - Image Compression Code Workbook
% 13848336 Nikki Fitzherbert
%
%% Introduction
%  This code workbook demonstrates how a function can be created in Matlab
%  to present a reconstructed version of an image using Singular Value
%  Decomposition. The code was based off the worked example in the script
%  "compression.m", which used SVD to compress and then reconstruct a
%  greyscale image.
%
%  The function contained in this code extends the example to construct a
%  set of colour images.

%% Defining the function
%
% This function "SVDcompress" takes two arguments: a filename and an array
% of singular values. Filename is the image that the function will be
% applied to, and must be pre-loaded using a command such as "imread". 
% Nretain is the specified array of singular values, which can be a single
% number or multiple numbers.
function outimage = SVDcompress(filename, Nretain)
    
    %% Converting the image into the required format
    %
    % The loaded image is renamed to "pic" to shorten the subsequent code
    % and converted from an integer format to a floating point format so
    % the SVD can be performed.
    pic = filename;
    pic = double(pic);
    
    % The range of data in the loaded image is also changed to 0-1.
    pic = pic / (max(pic,[],'all') - min(pic,[],'all'));
    
    %% Creating the RGB base colour components
    %
    % A colour image is made up of three different base colours. This
    % information is contained in the third dimension of the loaded matrix
    % and take a value of 1, 2, or 3 corresponding to the base colours
    % "red", "green" and "blue" respectively.
    colour_ind = [1, 2, 3];
        
    % The image can be separated into three 2D arrays, one for each of the
    % three base colours using the third dimension of the matrix
    R = pic(:,:,1);
    G = pic(:,:,2);
    B = pic(:,:,3);
    
    % The results can also be displayed graphically, together with the
    % original image as a reference point. The use of "figure" and
    % "subplot" means that all the images will be displayed at the same
    % time in a single figure.
    %  
    % figure
    %
    % subplot(2,2,1)
    % image(pic)
    % axis equal tight off
    % title('Original')
    %
    % subplot_i = 2;
    % for colour_ind = colour_ind
    %     subplot(2,2,subplot_i)
    %     imagesc(pic(:,:,colour_ind))
    %     axis equal tight off
    %     title(['Base colour #', num2str(colour_ind)])
    %     
    %     subplot_i = subplot_i + 1
    % end
    
    %% Performing the Singular Value Decomposition (SVD) for each base colour
    %
    % It is a relatively simple process to perform a SVD for each of the
    % three base colours. This can either be done using the three 2D arrays
    % defined in lines 45-47 or directly via a for loop (below).
    for colour_ind = colour_ind
        [U(:,:,colour_ind), S(:,:,colour_ind), V(:,:,colour_ind)] = svd(pic(:,:,colour_ind));
    end
    
    %% Reconstructing the image using a subset of singular values
    %
    % The final matrix is "outimage", which is a be a matrix representation
    % of the truncated reconstructed image. Only part of the information
    % available will be used to reconstruct the image. How much will be
    % used will be determined by the values of Nretain, which indicates the
    % number of singular values retained by the function.
    %
    % The reconstructed image(s) (depending on whether the value for
    % Nretain is a single value or an array of values) can be displayed
    % graphically. The use of "figure" in conjunction with "subplot" allows
    % multiple plots to be displayed on the same image file output in
    % Matlab.   
    figure
    
    subplot_i = 1;
    % Each of the matrices that were the result of the SVD (that is, U, S
    % and V matrices for each of the base colours) are then truncated
    % according to the value(s) of Nretain input into the SVDcompress
    % function.
    for Nretain = Nretain
        Uretr = U(:,1:Nretain,1);
        Sretr = S(1:Nretain,1:Nretain,1);
        Vretr = V(:,1:Nretain,1);
        
        Uretg = U(:,1:Nretain,2);
        Sretg = S(1:Nretain,1:Nretain,2);
        Vretg = V(:,1:Nretain,2);
        
        Uretb = U(:,1:Nretain,3);
        Sretb = S(1:Nretain,1:Nretain,3);
        Vretb = V(:,1:Nretain,3);
        
        % These nine matrices then need to be combined to form three
        % matrices; one each for the U, S and V variables that will then
        % multiply together to form the reconstructed images.
        Uret = cat(3, Uretr, Uretg, Uretb);
        Sret = cat(3, Sretr, Sretg, Sretb);
        Vret = cat(3, Vretr, Vretg, Vretb);
        
        for colour_ind = 1:max(colour_ind)
            outimage(:,:,colour_ind) = Uret(:,:,colour_ind) * Sret(:,:,colour_ind) * Vret(:,:,colour_ind)';
        end
        
        % The reconstructed images can be viewed in the Matlab console in a
        % single file.
        subplot(4,1,subplot_i);
        imagesc(outimage)
        axis equal tight off
        title(['Outimage, N = ', num2str(Nretain)])
        
        subplot_i = subplot_i + 1;
        
        % Or they can be saved as individual files back onto the desktop
        % for later use in other documents such as a research report.
        imwrite(outimage, strcat('Outimage, N = ', num2str(Nretain), '.png'))
        
    end
  
    
end