%% Assessment 2E - Characterising relations (Task 3)
% 13848336 Nikki Fitzherbert
%
% This code uses Matlab's 'issymmetric' function along with the
% user-defined 'isreflexive' and 'istransitive' functions to determine
% which of three input matrices are an equivalence relation, and how many
% equivalence classes that relation has.

load R.mat

issymmetric(R1)
isreflexive(R1)
istransitive(R1)
% All functions return 'true', so R1 is an equivalence relation.

plot(digraph(R1))
% This shows a graph comprised of five disjoint graphs, so R1 is an
% equivalence relation with five equivalence classes.

issymmetric(R2)
isreflexive(R2)
istransitive(R2)
% The first two functions return 'true', but the third returns 'false',
% indicating that R2 is symmetric and reflexive, but not transitive.
% Therefore, R2 is not an equivalence relation and plotting the digraph is
% not required (although interesting if one wishes to view how such a
% matrix is displayed graphically).
plot(digraph(R2))

issymmetric(R3)
isreflexive(R3)
istransitive(R3)
% Only the second function returns 'true', indicating that R3 is reflexive,
% but not symmetric nor transitive. Therefore, R3 is not an equivalence
% relation and plotting the digraph is not required (although interesting
% if one wishes to view how such a matrix is displayed graphically).
plot(digraph(R3))