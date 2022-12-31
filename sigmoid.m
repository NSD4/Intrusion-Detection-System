function g = sigmoid (z)
  % Works for z being a scalar, vector or matrix.
  g = (exp(-z)+1).^-1;
endfunction
