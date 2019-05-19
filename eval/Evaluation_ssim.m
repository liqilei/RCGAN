clc,clear,close all;
addpath(genpath('./fusionmetrics'))
dbstop if error;
ir_path = '../Demo_dataset/test/TNO/ir/';
vi_path = '../Demo_dataset/test/TNO/vi/';
fused_path = '../release/results/';

ir_list = dir([ir_path, '*.bmp']);
vi_list = dir([vi_path, '*.bmp']);
fused_list = dir([fused_path, '*.bmp']);
ssim_score = [];
for ii = 1 : size(ir_list, 1)
    disp(ii)
    if ((ir_list(ii).name == vi_list(ii).name) & (ir_list(ii).name == fused_list(ii).name))
        a = imread(fullfile(ir_path, ir_list(ii).name));
        b = imread(fullfile(vi_path, vi_list(ii).name));
        c = imread(fullfile(fused_path, fused_list(ii).name));
    else
        disp('wrong length')
    end
    if size(a, 3) == 3
        a = rgb2gray(a);
    end
    if size(b, 3) == 3
        b = rgb2gray(b);
    end
    A=double(a);
    B=double(b);
    C=double(c);
    grey_level=256;
    Criteria = Metric(A,B,C);
    ssim_score = [ssim_score, Criteria.Total];
end
