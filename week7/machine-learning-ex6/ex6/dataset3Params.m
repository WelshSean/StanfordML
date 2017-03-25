function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
sigma_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% Initialise counters to use to store values of C and Sigma that minimise
% the error.
min_err = Inf;
best_c = Inf;
best_sigma = Inf;

%c_vec = [0 0.001 ]';
%sigma_vec = [0 0.001 ]';
for i = 1:length(c_vec)
    for j = 1:length(sigma_vec)
        predictions = zeros(length(Xval),1);
        c_test=c_vec(i);
        sigma_test=sigma_vec(j);
        model= svmTrain(X, y, c_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        predictions(:) = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        formatSpec = 'sigma_test: %4.2f, c_test: %4.2f, error: %4.2f \n';
        fprintf(formatSpec,sigma_test, c_test, error)
        
        % Check of this is the smallest error that we've seen so far
        if error < min_err
            min_err = error
            best_c = c_test
            best_sigma = sigma_test
            formatSpec = 'New sigma_best: %4.2f \n';
            fprintf(formatSpec,best_sigma)
        end
    end
end

C=best_c
sigma=best_sigma

formatSpec = 'OPTIMAL PARAMETERS:: sigma: %4.2f, C: %4.2f \n';
fprintf(formatSpec,sigma, C)






% =========================================================================

end
