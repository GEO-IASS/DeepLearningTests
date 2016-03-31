function [ spares_Y ] = softmaxifyY(Y)
%SOFTMAXIFYY Turns a n dimesional integer column vector into a n by diff_in_Y
%spares matrix
assert(size(Y,2) == 1); 
n = size(Y,1);
min_Y = min(Y);
max_Y = max(Y);
diff_in_Y = max_Y - min_Y + 1;
spares_Y = zeros(n,diff_in_Y);
for i = 1:n
    spares_Y(i,Y(i) - min_Y + 1) = 1;
end

end