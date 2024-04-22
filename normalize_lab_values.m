function [l_norm, a_norm, b_norm] = normalize_lab_values(lab_color)
%NORMALIZE_LAB_VALUES Maps CIELAB values to be within 0 and 1
% The default range of lightness is 0-100, and a* and b* are -128 to 128
l_norm = lab_color(1)/100;
a_norm = (lab_color(2)/128 + 1)/2;
b_norm = (lab_color(3)/128 + 1)/2;
end

