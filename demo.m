%% Multi-label segmentation
clear;
clc;
close all;

image = imread('RedFlower.bmp');
[m,n,~] = size(image);

labelImg = imread('RedFlower-label.png'); % label image

load('RedFlower-superpixel.mat'); % superpixel

[seg_mask,seg_probabilities] = GMRSeg(image,labelImg,superpixels);
figure;imshow(image,[]);title('Source input image');
figure;imshow(labelImg,[]);title('Human label image');
figure;imshow(im2double(image) .* repmat(seg_mask,[1 1 3]),[]);title('Segmented object');
