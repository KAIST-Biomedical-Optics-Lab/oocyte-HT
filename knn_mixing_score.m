function [mixVec, mixMean] = knn_mixing_score(X, grp, k)
% X   : n x p standardized feature matrix
% grp : n x 1 numeric labels
% k   : number of nearest neighbors
%
% mixVec(i) = fraction of k nearest neighbors of sample i
%             that belong to a different group
% mixMean   = mean(mixVec)
%
% Lower is better (less local overlap)

    X = double(X);
    grp = grp(:);

    n = size(X,1);

    if n < 3
        mixVec = nan(n,1);
        mixMean = NaN;
        return
    end

    k = min(k, n-1);
    if k < 1
        mixVec = nan(n,1);
        mixMean = NaN;
        return
    end

    % pairwise Euclidean distance in raw standardized feature space
    D = pdist2(X, X, 'euclidean');

    % exclude self
    D(1:n+1:end) = inf;

    mixVec = nan(n,1);

    for i = 1:n
        [~, idxSort] = sort(D(i,:), 'ascend');
        nnIdx = idxSort(1:k);
        mixVec(i) = mean(grp(nnIdx) ~= grp(i));
    end

    mixMean = mean(mixVec, 'omitnan');
end