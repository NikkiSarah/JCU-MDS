%% Assessment 3: Microchip Voltage
% Nikki Fitzherbert 13848336
%
% The overarching goal of this code is to track a power surge in a
% microchip that is modelled by a linear system of equations.
%
% This code is accompanied by a word document that contains
% responses and relevant Matlab output for a number of goals.
%
% Voltage in the microchip circuit changes with time according to the
% following equation: Vn+1 = A*Vn
%
% where Vn is a vector of voltages at the nth millisecond and the elements
% of Vn correspond to different important locations in the microchip.

%% Step 1
%
% The matrix A is stored in the file "voltage_evolve.csv" and can be
% loaded easily into Matlab using the function "csvread":

A = csvread('voltage_evolve.csv');

%% Step 2
%
% The initial condition for microchip is: V0 = [1,0,0,0,...]T. It describes
% the situation in which the voltage is one at the input and zero
% everywhere else when time = 0 (that is, the '0th' millisecond).

V0 = [1; zeros(20,1)];

% The matrix A, initial condition V0 and the relationship described in the
% introduction can be used to calculate the voltages of the microchip
% circuit at all locations for time = 1 to time = 120 (that is, up to and
% including the 120th millisecond). Whilst this can be accomplished using a
% set of sequential equations, it is far more practical and efficient to
% use a loop for a larger number of iterations as is the case here.

% Start with an empty matrix
V = zeros(21,121);

n = 120;

for i = 0
    % Set the first colummn to equal V0
    V(:,i + 1) = V0(:,i + 1);
    % Iterate to get voltages at all locations up to time period n
    for j = 1:n
        V(:,j + 1) = A*V(:,j);
    end
end

V;

%% Step 3 / GOAL 1
%
% The results contained within V can now be used to plot the values of the
% voltage for the first five locations in the microchip for 0 < n < 120

V_sub = V(1:5, 2:120);
n = 1:n - 1;

plot(n, V_sub)
title('Voltage at first five locations up to 120th millisecond')
xlabel('time (milliseconds)')
ylabel('voltage')
legend('Input', 'Junction 1', 'Junction 2', 'Junction 3', 'Junction 4')
legend('boxoff')

% It can also be saved for use in other applications like Microsoft Word.
% saveas(gcf, 'C:\Users\Nikki\Documents\Training & Study\JCU\2019_Essential Mathematics\Week 2 - Vectors etc\Assessment 3\Goal_1.png')

%% Step 4 / GOAL 2
%
% It is also very easy to export the contents of the voltage matrix. For
% example, the code below exports the 21 voltages for the 120th millisecond
% as a text file.

V120 = V(:,121);
% fpath = 'C:\Users\Nikki\Documents\Training & Study\JCU\2019_Essential Mathematics\Week 2 - Vectors etc\Assessment 3\Goal_2.txt';
% save(fpath, 'V120', '-ascii')


%% Steps 5 and 6
%
% Eigenvalues (matrix D) and eigenvectors (matrix U) for the matrix A can
% be calculated using the commmand "eigs"; however, unless the second
% option is specified, "eigs" will only display the six largest
% eigenvalues.

[U,D] = eigs(A,21);
evals = diag(D);

% eigs doesn't always return the eigenvalues and eigenvectors in order of
% magnitude, so they need to be sorted (or equivalent) to determine the
% largest eigenvalue.

[dummy,ind] = sort(abs(evals), 'descend');
evals = evals(ind);
evals(1,1)

% The largest eigenvalue has a value of 1.0012.

%% Step 7 / Goal 3
%
% Since the eigenvectors in matrix U correspond to the eigenvalues in
% matrix D, the columns of U also need to be sorted by the indices used to
% order the eigenvalues.
%
% U is also multiplied by -1 so for easier comparison in later stages of
% the code. This is a purely cosmetic operation and does not affects the
% results.

U = U(:,ind);
U(:,[1,21]) = -U(:,[1,21]);

U;

% If this has been coded correctly, then the evaluation of U*D - U*D (the
% eigenvalue composition of A) will approach zero in Matlab due to rounding
% error.

if (A*U - U*D) < 0.1
    disp("well done")
else disp("check code")
end

% The eigenvector corresponding to the largest eigenvalue is therefore
% the first column of U:

maxevec = U(:,1)

% This eigenvector can be plotted against the result from the time
% evolution (V120) for easy comparison.

loc = 1:21;
plot(loc, V120)
title('Comparison of eigenvector and V120 results')
xlabel('Microchip location')
ylabel('voltage')
legend('V120')
legend('boxoff')

hold on
plot(loc, maxevec)
legend('V120','eigenvector')

diff = V120 - maxevec;

% As the chart shows, the pattern of voltage levels across the 21 microchip
% locations are the same i.e. it drops dramatically after the input
% location and is the roughly the same level across the 20 junctions. On
% the other hand, the voltage levels are consistently much smaller for the
% eigenvector compared to the result from the time evolution.

% saveas(gcf, 'C:\Users\Nikki\Documents\Training & Study\JCU\2019_Essential Mathematics\Week 2 - Vectors etc\Assessment 3\Goal_3.png');

%% Step 8
%
% There has been a voltage spike in a circuit and the task now is to use
% the measurements from the 120th millisecond to trace the voltages back in
% time. This should identify the location of any potential issues with the
% circuit.
%
% In order to accomplish this, the equation at the start of the code has to
% be inverted: Vtn = Ainverse*Vtn-1
%
% where Vt is Vtilda.

Ainv = inv(A);

%% Step 9
%
% In this scenario, the initial condition is Vt120; V120 from the previous
% scenario.
% 
Vt120 = V120;
%
% Vt120 and Ainv can be used together to work backwards and calculate the
% previous 22 voltages; that is, from V120 to V98.
%
% As before, start with an empty matrix
Vt = zeros(21,121);

n = 120;

for i = 120
    % Set the final colummn to equal Vt120
        Vt(:,i + 1) = Vt120(:,i - (i - 1));
    % Iterate back to get voltages at all locations from time period 120 to
    % time period 98
    for j = n:-1:98
        Vt(:,j) = Ainv*Vt(:,j + 1);
    end
end

Vt;

%% Step 10 / Goal 4
%
% The results contained within Vt can now be used to plot the values of the
% voltage for all microchips for n = 98 to n = 120.

V_sub2 = V(:, 99:121);
Vt_sub = Vt(:, 99:121);
n = 98:120;

plot(n, V_sub2)
title('Foward and backtracked voltages at all locations for the last 22 milliseconds')
xlabel('time (milliseconds)')
ylabel('voltage')

hold on
plot(n, Vt_sub, 'o')

% saveas(gcf, 'C:\Users\Nikki\Documents\Training & Study\JCU\2019_Essential Mathematics\Week 2 - Vectors etc\Assessment 3\Goal_4.png');

%% Step 11 / Goal 5
%
% The voltages produced by the two different approaches (forward and
% backtracked) are nearly indistinguishable for all locations on the
% microchip until n = 101. For this time point and the previous one
% (n = 10), the voltages produced by backtracking are increasingly smaller
% in magnitude (on average) than the voltages produced by the forward
% approach. However, for time points n = 98 and n = 99, this abruptly
% changes. At time point n = 99, most of the backtracked voltages are
% larger than those produced by the forward approach. At n = 98, all are
% much larger by a relatively high order of magnitude.
%
% This sudden divergence is a possible indication of an issue with the
% backtracking approach of calculating voltages for different locations in
% the microchip. The reason for this, as stated in step 12, is that the
% problem was ill-posed.

%% Step 13 / Goal 6
%
% The condition number for matrix A is:
cond(A)

% The condition number for the inverse of matrix A is:
cond(Ainv)

%% Step 14 / Goal 7
%
% Eigenvalues for matrix A were calculated in the section of the code
% marked as 'Steps 5 and 6'. The value of the largest eigenvalue in
% absolute terms was 1.0012.

evals(1,1)

% The largest eigenvalue for the Ainv matrix can be calculated exactly the
% same way as previously in this code:

[Uinv,Dinv] = eigs(Ainv,21);
evalsInv = diag(Dinv);

[dummyInv,indInv] = sort(abs(evalsInv), 'descend');
evalsInv = evalsInv(indInv);

evalsInv;
evalsInv(1,1)

% The largest eigenvalue for the inverse of matrix A has a value of 
% 1.4263 + 4.8038i.
%
% The value of the largest eigenvalue for matrix A is a real number, but
% the largest eigenvalue for the inverse of matrix A has both a real and
% complex part.

%% Step 15 / Goal 8
%
% In order to save the values of Vn and Vtn to a file that can be exported
% (in this case a csv file called 'Voltages.csv'). The matrix array
% 'Voltages' is coded such that it includes columns filled with zeros for
% the voltages of Vtn for n = 0 to n = 97. This is done for completeness
% as voltages for these time points were not calculated in this code.

Voltages = [V Vt];
fpath = 'C:\Users\Nikki\Documents\Training & Study\JCU\2019_Essential Mathematics\Week 2 - Vectors etc\Assessment 3\Voltages.csv';
save(fpath, 'Voltages', '-ascii')




