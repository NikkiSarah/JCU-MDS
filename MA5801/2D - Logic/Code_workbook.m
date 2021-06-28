%% Assessment 2D - Matlab Logic Exercise Code Workbook
% 13848336 Nikki Fitzherbert

% This code demonstrates how Matlab can be used to assess the truth value
% of a proposition. That is, it will determine if the proposition '(p
% implies q) and not-q if and only if p or q' is a tautology, a
% contradiction or neither

% Step 1: write an if-then function that defines if-then in terms of NOT
% and OR
% this uses the implication law: 'p implies q is logically equivalent to
% not-p or q'
ifthen = @(p,q) ~p|q

% Step 2: write and if-and-only-if connective function in terms of 'ifthen'
% this uses the equivalence law: 'p if and only if q is logically
% equivalent to p implies q and q implies p'
iff = @(p,q) ifthen(p,q) & ifthen(q,p)

% Step 3: define the proposition function using the two anonymous functions
% previously defined in Steps 1 and 2
proposition = @(p,q) iff(ifthen(p,q) & ~q, p|q)

% Step 4: use the function to determine if the proposition is a tautology,
% contradiction or neither
proposition(true, true)
proposition(true, false)
proposition(false, true)
proposition(false, false)

% Matlab outputs zero for every combination of p and q. Therefore, this
% expression is a contradiction.