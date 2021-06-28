function rating = compareImages(imA,imB)

% Convert images to floating point for SVD
imA = double(imA);
imB = double(imB);

% Helps to subtract out average component:
imA = imA - mean(imA(:));
imB = imB - mean(imB(:));

% First do the SVD of the two images.
[U1,S1,V1] = svd(imA);
[U2,S2,V2] = svd(imB);

% Grab the number of singular values, which is always the smaller of the
% two dimensions of the matrix.
n = min(size(imA));

% These "norms" are a way of making the maximal value of the comparison
% equal to 1. 
norm1 = sqrt(sum(diag(S1).^2));
norm2 = sqrt(sum(diag(S2).^2));

S1 = S1 / norm1;
S2 = S2 / norm2;

tot = 0;
for i = 1:n
    for j = 1:n
        %print("i,j", i,j, "V:", dot(V1[i,:], V2[j,:]), "U:", dot(U1[:,i], U2[:,j]), "S:", S1[i] * S2[j])

        dotprods = (V1(:,i)' * V2(:,j)) * (U1(:,i)' * U2(:,j));
        val = dotprods * S1(i,i) * S2(j,j);
        
        %print("val:", val)
        tot = tot + val;
    end
end

rating = tot;
    
end