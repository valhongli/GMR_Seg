function distMat = getDistMat(edges,seg_vals,spnum)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
        
    valDistances = sqrt(sum((seg_vals(edges(:,1),:) - seg_vals(edges(:,2),:)).^2,2));
    
    distMat = sparse([edges(:,1);edges(:,2)],[edges(:,2);edges(:,1)],[valDistances;valDistances],spnum,spnum);
    
    %dist = normalize(valDistances); %Normalize to [0,1]

end

