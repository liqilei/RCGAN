function  Result = Metric(I,V,X)
I=double(I);
V=double(V);
X=double(X);
Result.Total = [];

temp(:,:,1)=I;
temp(:,:,2)=V;

Result.SSIM=ssim_n_matlab(temp,X);
Result.Total = [Result.Total; Result.SSIM];






grey_level=256;
