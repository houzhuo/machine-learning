function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.


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
C_mat = [ 0.1; 0.3; 1 ]
sigma_mat= [0.1; 0.3; 1]
min_error = 1000000;
C = 0.0;
sigma = 0.0;
for i=1:length(C_mat)
  
  c = C_mat(i)
  for j=1:length(sigma_mat)
    sig = sigma_mat(j)
    model= svmTrain(X, y, c, @(X, y) gaussianKernel(X, y, sig)); 
    pred = svmPredict(model,Xval)
    error = mean(double(pred ~= yval))
    if (min_error > error)
      min_error = error
      C = c
      sigma = sig
    end
  endfor
endfor






% =========================================================================

end