%X = [1 2; 3 4; 5 6];
%y = [3; 4; 5;];
%theta = [1; 2; 3];

%disp(size(X))
%disp(size(theta))
%disp(theta' * X)

data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

disp(size(X))
disp(size(theta))


computeCost(X, y, theta)