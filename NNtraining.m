disp("Setting things up..."); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))

RawData = [];

% Benign training examples
for i = 0:10000,
  step = i * 76;
  ithExample = dlmread("C:/YourFilePath/Day-00-00-2018_TrafficForML_CICFlowMeter.csv", "", [step,0,step,78]);
  RawData = [ithExample; RawData];

  if (i == 0),
    disp("Reading benign data ... line 0"); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
  elseif (i == 76000),
    disp("Reading benign data ... line 1000"); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
  end;
end;

% Malicious Bot training examples
BotStartRow = 762385;
for i = 0:10000,
  step = BotStartRow + (i * 28);
  ithExample = dlmread("C:/Users/namjo/OneDrive/Documents/Namjote.S.Dulay/ML/DataSets/IDS/Training Data/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv", "", [step,0,step,78]);
  RawData = [ithExample; RawData];
  if (i == 0),
    disp("Reading Malicious data ... line 0"); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
  elseif (i == 790385),
    disp("Reading Malicious data ... line 1000"); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
  end;
end;

disp("Raw data loaded..."); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
X = RawData(:,1:end-1);
y = RawData(:,end);
disp("X and y extracted..."); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
lambda = 1;
input_layer_size = size(X,2);
hidden_layer_size = 50;
num_labels = 2;

% Function to be minimised, parameter for fmincg
J = @(p) costFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
disp("Getting ready to optimise weights..."); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
% More parameters for fmincg
initial_Theta1 = randInitialWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitialWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
options = optimset('MaxIter', 50);
disp("Optimising weights..."); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
% Optimising weights of NN and return as unrolled vector in nn_params
[nn_params, ~] = fmincg(J, initial_nn_params, options);

% Unroll nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
disp("Weights have been optimised..."); strftime ("%Y-%m-%d %H:%M:%S", localtime (time ()))
% Ouput accuracy of trained NN
accuracy (Theta1, Theta2, X, y);

% Write weights/thetas to txt file
SaveThetas = input("Do you wish to store these weights in optimisedNNweights.txt?: (Y/N)");
if (SaveThetas == "Y"),
  save("-append", "-ascii", "optimisedNNweights.txt", "Theta1", "Theta2");
end;
