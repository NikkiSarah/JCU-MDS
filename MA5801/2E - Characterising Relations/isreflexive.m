%% Assessment 2E - Characterising relations (Task 1)
% 13848336 Nikki Fitzherbert
%
% This code demonstrates how an 'isreflexive' function can be created to
% help determine if a matrix is an equivalance relation.

% Reflexivity is determined by checking if all the diagonal elements of a
% matrix are true. That is, this is equivalent to checking that there are
% no zeros on the main diagonal.
function isreflexive = isreflexive(matrix)
    matrix = matrix;
    
    if any(diag(matrix)) == 0
        disp 'false'
    else disp 'true'
    end
end