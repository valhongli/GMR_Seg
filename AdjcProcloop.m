function adjcMerge = AdjcProcloop(M,N)
% $Description:
%    -compute the adjacent matrix
% $Agruments
% Input;
%    -M: superpixel label matrix
%    -N: superpixel number 
% Output:
%    -adjcMerge: adjacent matrix, has no weight on the edges, just the
%    connection representation

adjcMerge = zeros(N,N); % N * N adjacent matrix
[m,n] = size(M);

% connected to those nodes neighboring it
for i = 1:m-1
    for j = 1:n-1
        
        if(M(i,j) ~= M(i,j + 1))%right neighbor
            adjcMerge(M(i,j),M(i,j+1)) = 1;% symmetric matrix
            adjcMerge(M(i,j+1),M(i,j)) = 1;
        end;
        
        if(M(i,j) ~= M(i + 1,j))%bottom neighbor
            adjcMerge(M(i,j),M(i + 1,j)) = 1;
            adjcMerge(M(i + 1,j),M(i,j)) = 1;
        end;
        
        if(M(i,j) ~= M(i + 1,j + 1))%right bottom neighbor
            adjcMerge(M(i,j),M(i + 1,j + 1)) = 1;
            adjcMerge(M(i + 1,j + 1),M(i,j)) = 1;
        end;
        
        if(M(i+1,j) ~= M(i,j+1))% bottom and right neighbor
            adjcMerge(M(i + 1,j),M(i,j + 1)) = 1;
            adjcMerge(M(i,j + 1),M(i + 1,j)) = 1;
        end;
        
    end;
end;
%figure;imshow(adjcMerge,[]);
%{
bd = unique([M(1,:),M(m,:),M(:,1)',M(:,n)']); %four boundary superpixel
for i = 1:length(bd)
    for j = i + 1:length(bd)
        adjcMerge(bd(i),bd(j)) = 1;
        adjcMerge(bd(j),bd(i)) = 1;
    end
end
%}
%figure;imshow(adjcMerge,[]);