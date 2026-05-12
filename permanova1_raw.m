%% ============================================================
% LOCAL FUNCTION: one-way PERMANOVA on raw feature matrix
%% ============================================================
function [pVal, R2, Fstat] = permanova1_raw(X, grp, nPerm)
% X   : n x p feature matrix
% grp : n x 1 numeric group labels
% nPerm : number of permutations

    X = double(X);
    grp = grp(:);

    n = size(X,1);
    groupList = unique(grp);
    k = numel(groupList);

    if n < 3 || k < 2
        pVal = NaN;
        R2   = NaN;
        Fstat = NaN;
        return
    end

    grandMean = mean(X,1,'omitnan');

    SS_total = sum(sum((X - grandMean).^2));

    SS_between = 0;
    for i = 1:k
        idx = grp == groupList(i);
        Xi = X(idx,:);
        ni = size(Xi,1);
        if ni == 0
            continue
        end
        mui = mean(Xi,1,'omitnan');
        SS_between = SS_between + ni * sum((mui - grandMean).^2);
    end

    SS_within = SS_total - SS_between;

    df_between = k - 1;
    df_within  = n - k;

    if df_between <= 0 || df_within <= 0 || SS_within < 0
        pVal = NaN;
        R2   = NaN;
        Fstat = NaN;
        return
    end

    MS_between = SS_between / df_between;
    MS_within  = SS_within  / df_within;

    if MS_within == 0
        Fstat = Inf;
    else
        Fstat = MS_between / MS_within;
    end

    R2 = SS_between / SS_total;

    permF = nan(nPerm,1);
    for pi = 1:nPerm
        grpPerm = grp(randperm(n));

        SSb_perm = 0;
        for i = 1:k
            idx = grpPerm == groupList(i);
            Xi = X(idx,:);
            ni = size(Xi,1);
            if ni == 0
                continue
            end
            mui = mean(Xi,1,'omitnan');
            SSb_perm = SSb_perm + ni * sum((mui - grandMean).^2);
        end

        SSw_perm = SS_total - SSb_perm;

        MSb_perm = SSb_perm / df_between;
        MSw_perm = SSw_perm / df_within;

        if MSw_perm == 0
            permF(pi) = Inf;
        else
            permF(pi) = MSb_perm / MSw_perm;
        end
    end

    pVal = (1 + sum(permF >= Fstat)) / (nPerm + 1);
end