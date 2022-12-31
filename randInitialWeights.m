%L_in = S(j+1) where 1 is subscript. L_out = Sj
% This function acts as symmetry breaking
function Weights = randInitialWeights (L_out, L_in)
  % Epsilon equation:
  epsilon = sqrt(6) / sqrt(L_out + L_in);
  Weights = (rand(L_in, L_out + 1) * 2 * epsilon) - epsilon;
endfunction
