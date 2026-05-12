function [dbiVal, chVal] = cluster_validity_scores(X, grp)
% X   : n x p standardized feature matrix
% grp : n x 1 numeric group labels
%
% DBI: lower is better
% CH : higher is better

    X = double(X);
    grp = grp(:);

    n = size(X,1);
    groups = unique(grp);
    k = numel(groups);

    if n < 3 || k < 2
        dbiVal = NaN;
        chVal  = NaN;
        return
    end

    % ----- centroids and within-cluster scatter -----
    centroids = nan(k, size(X,2));
    S = nan(k,1);   % average distance to centroid

    for i = 1:k
        idx = grp == groups(i);
        Xi = X(idx,:);
        if isempty(Xi)
            dbiVal = NaN;
            chVal  = NaN;
            return
        end

        ci = mean(Xi, 1, 'omitnan');
        centroids(i,:) = ci;

        d = sqrt(sum((Xi - ci).^2, 2));
        S(i) = mean(d, 'omitnan');
    end

    % ----- DBI -----
    M = squareform(pdist(centroids, 'euclidean'));  % centroid distances
    M(1:k+1:end) = NaN;

    Rij = nan(k,k);
    for i = 1:k
        for j = 1:k
            if i ~= j
                if M(i,j) == 0
                    Rij(i,j) = Inf;
                else
                    Rij(i,j) = (S(i) + S(j)) / M(i,j);
                end
            end
        end
    end

    Ri = max(Rij, [], 2, 'omitnan');
    dbiVal = mean(Ri, 'omitnan');

    % ----- CH -----
    grandMean = mean(X, 1, 'omitnan');

    SS_between = 0;
    SS_within  = 0;

    for i = 1:k
        idx = grp == groups(i);
        Xi = X(idx,:);
        ni = size(Xi,1);
        ci = centroids(i,:);

        SS_between = SS_between + ni * sum((ci - grandMean).^2);
        SS_within  = SS_within  + sum(sum((Xi - ci).^2));
    end

    if k == 1 || n == k || SS_within == 0
        chVal = NaN;
    else
        chVal = (SS_between / (k - 1)) / (SS_within / (n - k));
    end
end