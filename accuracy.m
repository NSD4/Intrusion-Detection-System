% Predicts the classes of training data given weights of neural network after optimisation
% Outputs accuracy of NN model
function acc = accuracy (Theta1, Theta2, X, y)

  m = size(X, 1);
  num_classes = size(Theta2, 1);

  % Each column is a training example's feature values
  a1 = [ones(m, 1) X];
  a2 = sigmoid(a1 * Theta1');

  % Adding bias units
  a2 = [ones(m, 1) a2];

  % Output layer predictions
  a3 = sigmoid(a2 * Theta2');

  % Max returns the index of each column that contains the largest value and stores them a row vector in predictions
  % The index stores in predictions relates to a specific class
  [dummy, predictions] = max(a3, [], 2);

  % Predictions == y will produce a vector of 1's=Correct Prediction and 0's being the converse.
  accuracy = mean(double(predictions == y)) * 100;
  disp("The accuracy of the trained Neural Network:"), disp(accuracy);
endfunction
