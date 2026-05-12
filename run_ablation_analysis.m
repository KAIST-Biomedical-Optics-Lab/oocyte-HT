function [TabAbl_2DBF, TabAbl_25DBF, TabAbl_HT] = run_ablation_analysis( ...
    XrawByMod, grpByMod, featureNamesByMod, modNames, ...
    morphology_2D, intensity_2D, texture_2D, ...
    morphology_3D, intensity_3D, mass_3D, texture_3D, ...
    nPerm, knn_k, ARI_CLUSTER_REPS)

TabAbl_2DBF  = table();
TabAbl_25DBF = table();
TabAbl_HT    = table();

nMod = numel(modNames);

for mi = 1:nMod

    modName   = modNames{mi};
    Xfull     = XrawByMod{mi};
    grpNum    = grpByMod{mi};
    featNames = featureNamesByMod{mi};

    if isempty(Xfull) || isempty(grpNum) || isempty(featNames)
        warning('Skipping %s because input data is empty.', modName);
        continue
    end

    nFeat = size(Xfull,2);
    if nFeat < 2
        warning('Skipping %s because number of features is < 2.', modName);
        continue
    end

    [~, r2Base, ~]    = permanova1_raw(Xfull, grpNum, nPerm);
    [dbiBase, chBase] = cluster_validity_scores(Xfull, grpNum);
    [~, mixBase]      = knn_mixing_score(Xfull, grpNum, knn_k);
    silBase           = mean(silhouette(Xfull, grpNum), 'omitnan');

    nClust = numel(unique(grpNum));
    cluBase = kmeans(Xfull, nClust, ...
        'Replicates', ARI_CLUSTER_REPS, ...
        'Distance', 'sqeuclidean', ...
        'Display', 'off');
    ariBase = adjusted_rand_index(grpNum, cluBase);

    deltaR2  = nan(nFeat,1);
    deltaARI = nan(nFeat,1);
    deltaSil = nan(nFeat,1);
    deltaDBI = nan(nFeat,1);
    deltaCH  = nan(nFeat,1);
    deltaMix = nan(nFeat,1);

    for fj = 1:nFeat
        keepIdx = true(1,nFeat);
        keepIdx(fj) = false;

        Xabl = Xfull(:, keepIdx);

        [~, r2Abl, ~]   = permanova1_raw(Xabl, grpNum, nPerm);
        [dbiAbl, chAbl] = cluster_validity_scores(Xabl, grpNum);
        [~, mixAbl]     = knn_mixing_score(Xabl, grpNum, knn_k);
        silAbl          = mean(silhouette(Xabl, grpNum), 'omitnan');

        cluAbl = kmeans(Xabl, nClust, ...
            'Replicates', ARI_CLUSTER_REPS, ...
            'Distance', 'sqeuclidean', ...
            'Display', 'off');
        ariAbl = adjusted_rand_index(grpNum, cluAbl);

        deltaR2(fj)  = r2Base  - r2Abl;
        deltaARI(fj) = ariBase - ariAbl;
        deltaSil(fj) = silBase - silAbl;
        deltaDBI(fj) = dbiAbl - dbiBase;
        deltaCH(fj)  = chBase - chAbl;
        deltaMix(fj) = mixAbl - mixBase;
    end

    featFamily = strings(nFeat,1);

    for fj = 1:nFeat
        feat = string(featNames{fj});

        if strcmp(modName,'2DBF') || strcmp(modName,'2.5DBF')
            if ismember(feat, morphology_2D)
                featFamily(fj) = "Morphology";
            elseif ismember(feat, intensity_2D)
                featFamily(fj) = "Intensity / RI statistics";
            elseif ismember(feat, texture_2D)
                featFamily(fj) = "Texture";
            else
                featFamily(fj) = "Other";
            end

        elseif strcmp(modName,'HT')
            if ismember(feat, morphology_3D)
                featFamily(fj) = "Morphology";
            elseif ismember(feat, intensity_3D)
                featFamily(fj) = "Intensity / RI statistics";
            elseif ismember(feat, mass_3D)
                featFamily(fj) = "Mass";
            elseif ismember(feat, texture_3D)
                featFamily(fj) = "Texture";
            else
                featFamily(fj) = "Other";
            end

        else
            error('Unknown modality: %s', modName);
        end
    end

    Tab = table( ...
        string(featNames(:)), featFamily, ...
        deltaR2, deltaARI, deltaSil, deltaDBI, deltaCH, deltaMix, ...
        'VariableNames', {'Feature','FeatureFamily','DeltaR2','DeltaARI','DeltaSil','DeltaDBI','DeltaCH','DeltaMix'});

    if strcmp(modName, '2DBF')
        TabAbl_2DBF = Tab;
    elseif strcmp(modName, '2.5DBF')
        TabAbl_25DBF = Tab;
    elseif strcmp(modName, 'HT')
        TabAbl_HT = Tab;
    end

    fprintf('\n===== ABLATION | %s =====\n', modName);
    disp(sortrows(Tab, 'DeltaR2', 'descend'));
end
end