%% Assessment 2E - Characterising relations (Task 2)
% 13848336 Nikki Fitzherbert
%
% This code demonstrates how an 'istransitive' function can be created to
% help determine if a matrix is an equivalance relation.

% Transitivity is determined by checking if all non-zero elements in the
% square of the matrix are accompanied by non-zero (true) elements at the
% same location in the non-squared version of the matrix.
function istransitive = istransitive(matrix)
    matrix = matrix;
    
    Rsq_nonzeros = find(matrix^2 > 0);
    R_nonzeros = find(matrix > 0);
    
    if length(Rsq_nonzeros) == length(R_nonzeros) & all((Rsq_nonzeros - R_nonzeros) == 0)
        disp 'true'
    else disp 'false'
    end
end