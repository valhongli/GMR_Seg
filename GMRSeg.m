function [seg_mask,seg_probabilities] = GMRSeg(srcImg,labelImg,superpixels)
%GMRSEG Summary of this function goes here
  %GMRSEG Summary of this function goes here
%   The is the main function of our graph-based manifold ranking
%   segmentation
%
% Inputs: srcImg   - source image
%         labelImg - 1 stands for labeled fore-ground; 2 stands for
%                    labeled back-ground; 0 stands for no labels
%
%
% Outputs: seg_mask - binary segmentation results
%          seg_probabilities - segmentation results with probabilities,
%                              not binary segmentation

% Li Hong wrote this code in 03-31-2015, part of this code comes from "saliency detection
% via graph-based manifold ranking - CVPR 2013"

    alpha = 0.99;% control the balance of two items in manifold ranking cost function
    [mLabel,nLabel,~] = size(labelImg);
 %% ------------ generate background and foreground superpixels -----------   
    spnum = max(superpixels(:)); % the number of superpixels
    
    TempFG = zeros(size(labelImg)); % fore-ground indication
    TempFG(labelImg == 1) = 1;
    TempBG = zeros(size(labelImg)); % back-ground indication
    TempBG(labelImg == 2) = 1;
    
    [m,n,k] = size(srcImg);
    
    [mSup,nSup,~] = size(superpixels);
    
    if ((mLabel ~= mSup) || (nLabel ~= nSup))
        error('Size of superpixel and lableimage does not match!!!');
    end
    
    fgSup = TempFG .* superpixels; %the non-zeros are labeled fore-ground superpixels
    bgSup = TempBG .* superpixels; %the non-zeros are labeled back-ground superpixels
    fgSup = setdiff(unique(fgSup),0);% all the labeled fore-ground superpixels
    bgSup = setdiff(unique(bgSup),0);% all the labeled back-ground superpixels  
%% -----------------------------
    input_vals = reshape(srcImg, m * n, k);
    rgb_vals = zeros(spnum,1,3);
    inds = cell(spnum,1);
    for i = 1:spnum
        inds{i} = find(superpixels == i);
        rgb_vals(i,1,:) = mean(input_vals(inds{i},:),1);
    end  
    lab_vals = colorspace('Lab<-', rgb_vals); 
    seg_vals = reshape(lab_vals,spnum,3);% feature for each superpixel
%% ----------------------design the graph model--------------------------%%
% get edges(the connection between two nodes) of the graph model
    adjloop = AdjcProcloop(superpixels,spnum); % get adjacent matrix
    edges = [];
    
    % get the superpixel's neighbor by walk through each row(each row 
    % represent a superpixel and its all connected node in the graph(0 
    % stand for no edge and 1 stand for there is a edge between them))
    for i = 1:spnum 
        indext = [];
        
        % search in the i-th row, find all the nodes have edges between 
        % them and superpixel i, inner circle
        ind = find(adjloop(i,:) == 1); 
        
        % but also connected to the nodes sharing common boundaries 
        % with its neighboring node),outer circle
        for j = 1:length(ind) 
            indj = find(adjloop(ind(j),:) == 1);
            indext = [indext,indj];
        end
        
        indext = [indext,ind];%all the connected nodes of current node (neighbors and neighbors' neighbors)
        indext = indext((indext > i));
        indext = unique(indext); % exclude those repeated connection
        
        % get the current node's all neighbors, establish a edge between
        % each of them        
        if(~isempty(indext))
            ed = ones(length(indext),2);
            ed(:,2) = i * ed(:,2);
            ed(:,1) = indext;
            edges = [edges;ed];
        end
    end
 
%% --------- must link and must-not link construction ------------------
%%{   
% must link construction    
    %fore-ground must link
     fgSup = sort(fgSup,'descend'); 
     edge_new = [];
     num = length(fgSup);
     for ii = num:-1:1
         ind = fgSup(fgSup > fgSup(ii));
         for jj = 1:length(ind)
             edge = [ind(jj) fgSup(ii)];
             edge_new = [edge_new;edge];
         end
     end
     all_edges = union(edges,edge_new,'rows');
     [B,IX] = sort(all_edges);
     edges_final_fore_ground = [all_edges(IX(:,2),1) all_edges(IX(:,2),2)];
    
     %back-ground must link
     bgSup = sort(bgSup,'descend'); 
     edge_new = [];
     num = length(bgSup);
     for ii = num:-1:1
         ind = bgSup(bgSup > bgSup(ii));
         for jj = 1:length(ind)
             edge = [ind(jj) bgSup(ii)];
             edge_new = [edge_new;edge];
         end
     end
     all_edges = union(edges,edge_new,'rows');
     [B,IX] = sort(all_edges);
     edges_final_back_ground = [all_edges(IX(:,2),1) all_edges(IX(:,2),2)];
     
     all_edges_final = union(edges_final_fore_ground,edges_final_back_ground,'rows');
     [B,IX] = sort(all_edges_final);
     edges = [all_edges_final(IX(:,2),1) all_edges_final(IX(:,2),2)];
%%}     
%%{
% must-not link construction
     bgSup = setdiff(bgSup,intersect(fgSup,bgSup)); 
     fgSup = sort(fgSup,'descend');
     bgSup = sort(bgSup,'descend'); 
     edges_temp = [];
     for ii = 1:length(fgSup)               
        for jj = 1:length(bgSup)           
           edge = [fgSup(ii) bgSup(jj)];
           edges_temp = [edges_temp;edge];            
        end         
     end 
     edges_temp = unique(edges_temp,'rows');
     [B,IX] = sort(edges_temp);
     edges_new = [edges_temp(IX(:,2),1) edges_temp(IX(:,2),2)]; 
     temp =  edges_new(~(edges_new(:,1) > edges_new(:,2)),1);
     edges_new(~(edges_new(:,1) > edges_new(:,2)),1) =  edges_new(~(edges_new(:,1) > edges_new(:,2)),2);
     edges_new(~(edges_new(:,1) > edges_new(:,2)),2) = temp; 
     edges_new = unique(edges_new,'rows');
     [B,IX] = sort(edges_new);
     edges_mustnot_link = [edges_new(IX(:,2),1) edges_new(IX(:,2),2)];
     
     % exclude the edges_mustnot_link from original edges
     edges = setdiff(edges,edges_mustnot_link,'rows');
 %% find the optimal sigma
    % adaptive weight
    distMat = getDistMat(edges,seg_vals,spnum);
    [~,W,~] = median_local_dist(distMat);    
    dd = sum(W); 
    D = sparse(1:spnum,1:spnum,dd); 
    clear dd;
    optAff =(D - alpha * W) \ eye(spnum);

 % get the indication vector Y    
    Y_fg = zeros(spnum,1);
    Y_bg = zeros(spnum,1);
    Y_fg(fgSup) = 1;
    Y_bg(bgSup) = 1;

 %%------ first stage: fore-ground inference
    fgInf = optAff * Y_fg;
    fgInf = (fgInf - min(fgInf(:)))/(max(fgInf(:))-min(fgInf(:)));

 %%------ second stage: back-ground inference
    bgInf = optAff * Y_bg;
    bgInf = (bgInf - min(bgInf(:)))/(max(bgInf(:))-min(bgInf(:)));       

    bgInf = 1 - bgInf;   
 
    objInf = fgInf .* bgInf; 

    objInf = (objInf - min(objInf(:)))/(max(objInf(:))-min(objInf(:)));
    
 %% ----- assign the segmentation label to each pixel
    
    segMapSoft = zeros(m,n);
    for ii = 1:spnum
        segMapSoft(inds{ii}) = objInf(ii);
    end
    segMapSoft = (segMapSoft - min(segMapSoft(:)))/(max(segMapSoft(:))-min(segMapSoft(:)));

    segMap = zeros(m,n);
    segMap(segMapSoft > mean(segMapSoft(:))) = 2;
    segMap(segMapSoft <= mean(segMapSoft(:))) = 1; 
 
    seg_mask = mat2gray(segMap);
    seg_probabilities = segMapSoft;
end
