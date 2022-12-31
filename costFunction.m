% This function calculates cost of NN with given weights and derivatives of Thetas.
% This function is passed into fmincg and is used to optimise weights/Thetas.
function [J grad] = costFunction (nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    m = size(X, 1);
    X = [ones(m,1) X];

    % Variables to return
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));


    % FULLY VECTORISED IMPLEMENTATION - FOR LOOP IS TOO SLOW

    % -----------------------------------------------------------------------------------
    %      Part 1 - Carry out Foward propogation, then find cost using Cost Function
    % -----------------------------------------------------------------------------------

    % each column is a training example's features -- (n+1) x m
    a1Vec = X';

    % -- r2 x m, where r2 is number of units in layer 2
    a2Vec = sigmoid(Theta1*a1Vec);

    %Adds bias unit to layer 2 activation units for every iteration of a training example -- (r2+1) x m
    a2Vec = [ones(1,m); a2Vec];

    % Output layer predictions -- r3 x m, where r3 is number of units in layer 3
    a3Vec = sigmoid(Theta2*a2Vec);

    yVec=[1:num_labels] == y; % [1:num_labels] produces a row vector with elements with range(1:10)
    yVec = yVec';

    % Each column shows cost of output layer after FP for a single training example -- n x m
    JVec = (yVec.*log(a3Vec)) + ((1-yVec).*log(1-a3Vec));

    % Unregularised Cost
    J = J + sum(JVec(:));
    J = J * -1/m;

    % Regularised Cost
    theta1Reg = Theta1(:,2:end).^2;
    theta2Reg = Theta2(:,2:end).^2;
    J = J + lambda/2/m*(sum(theta1Reg(:)) + sum(theta2Reg(:)));

    % ----------------------------------------------------------------------------------------------------
    %     Part 2 - Carry out Back propogation to compute deltas for each node, then calculate derivatives
    % ----------------------------------------------------------------------------------------------------

    UpperCasedelta1 = 0;
    UpperCasedelta2 = 0;

    % Back propogation
    LowerCasedelta3 = a3Vec - yVec;
    LowerCasedelta2 = (Theta2'*LowerCasedelta3).*a2Vec.*(1-a2Vec);
    LowerCasedelta2 = LowerCasedelta2(2:end,:); % removing 0th unit in delta 2 after every training example as this is a bias unit
    UpperCasedelta1 = UpperCasedelta1 + (LowerCasedelta2*a1Vec');
    UpperCasedelta2 = UpperCasedelta2 + (LowerCasedelta3*a2Vec');

    % Derivative computations
    Theta1Reg = lambda*[zeros(size(Theta1,1),1) Theta1(:, 2:end)];
    Theta2Reg = lambda*[zeros(size(Theta2,1),1) Theta2(:, 2:end)];
    Theta1_grad = 1/m*(UpperCasedelta1 + Theta1Reg);
    Theta2_grad = 1/m*(UpperCasedelta2 + Theta2Reg);

    % Unroll the gradients - fmincg optimisation function requires parameters to be scalars/vectors
    grad = [Theta1_grad(:) ; Theta2_grad(:)];

endfunction
