function [ R ] = SSIM_n( I,F )
%SSIM_N Summary of this function goes here
%   Detailed explanation goes here
% written in 2017.12.31 by qilei
[r,c,N]=size(I);
I=double(I)/255;
F=double(F)/255;
for i=1:N
    ssim_n(i)=ssim(I(:,:,i),F);
end
R=sum(ssim_n)/N;
end
