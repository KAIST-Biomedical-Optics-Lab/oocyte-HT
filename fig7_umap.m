% 2DBF 2.5DBF HT UMAP
clear; clc; close all;

scriptFullPath = mfilename('fullpath');

if ~isempty(scriptFullPath)
    scriptDir = fileparts(scriptFullPath);
    cd(scriptDir);
    addpath(scriptDir);
else
    warning('Running outside script file → using current folder');
    scriptDir = pwd;
end

assert(exist('run_umap','file')==2, 'run_umap.m not found on MATLAB path');
rng(0,'twister');


%% User settings: UMAP, PCA, ablation test, silhouette and PERMANOVA score ON OFF 선택
DRAW_UMAP_PANEL   = true;
CALC_SILHOUETTE   = true;
CALC_CH           = true;
CALC_DBI          = true;
CALC_KNN_MIX      = true;
CALC_PERMANOVA    = true; 
CALC_ARI          = true;
CALC_AC           = true;
ARI_CLUSTER_REPS  = 20;
RUN_ABLATION      = true;
umap_n_neighbors = 15; 
umap_min_dist    = 0.10;
nPerm = 999;
knn_k = 10;

if CALC_PERMANOVA || RUN_ABLATION
    assert(exist('permanova1_raw','file')==2, 'permanova1_raw.m not found on MATLAB path');
end
if CALC_DBI || CALC_CH
    assert(exist('cluster_validity_scores','file')==2, 'cluster_validity_scores.m not found on MATLAB path');
end
if CALC_KNN_MIX || RUN_ABLATION
    assert(exist('knn_mixing_score','file')==2, 'knn_mixing_score.m not found on MATLAB path');
end
if RUN_ABLATION
    assert(exist('run_ablation_analysis','file')==2, 'run_ablation_analysis.m not found on MATLAB path');
end


%% File
fp = 'G:\RealData\excels\all_features_20260419_192742.xlsx';
sheetName = 'all_features';
assert(isfile(fp), 'Excel file not found');

%% Groups / Modalities
groupOrder = {'D0','D1','D2'};
modNames   = {'2DBF','2.5DBF','HT'};


%% Colors/ Markers=========================================================

colD0   = [1.00 0.30 0.60];
colD1   = [0.30 1.00 0.50];
colD2   = [0.30 0.60 1.00];
colGray = [0.82 0.82 0.82];
colEdge = [0 0 0];
mkMain = 30;
mkGray = 30;


%% Read Data===============================================================
T = readtable(fp, 'Sheet', sheetName, 'VariableNamingRule', 'preserve');
T.modalityLabel = string(strtrim(T.modality));
Gall = upper(strtrim(string(T.group)));
Gall(~ismember(Gall, string(groupOrder))) = missing;
T.group = Gall;


%% Feature Definition
morphology_2D = [
    "area_um2"
    "major_axis_2d_um"
    "minor_axis_2d_um"
    "aspect_ratio_2d"
    "fourier_power_c1"
    "fourier_power_c2"
    "ellipticity_2d"
    "perimeter_um"
    "perimeter_over_area"
    ];

intensity_2D = [
    "intensity_mean"
    "intensity_std"
    "intensity_cv"
    ];

texture_2D = [
    "asm"
    "contrast"
    "correlation"
    "variance"
    "homogeneity"
    "sum_average"
    "sum_variance"
    "sum_entropy"
    "entropy"
    "diff_variance"
    "diff_entropy"
    "imc1"
    "imc2"
    "dissimilarity"
    "cluster_shade"
    "cluster_tendency"
    ];

morphology_3D = [
    "area_um2"
    "major_axis_2d_um"
    "minor_axis_2d_um"
    "aspect_ratio_2d"
    "fourier_power_c1"
    "fourier_power_c2"
    "ellipticity_2d"
    "perimeter_um"
    "perimeter_over_area"
    "volume_pL"
    "major_axis_3d_um"
    "minor_axis_3d_um"
    "aspect_ratio_3d"
    "sph_power_l0"
    "sph_power_l2"
    "ellipticity_3d"
    "surface_um2"
    "surface_over_volume"
    ];

intensity_3D = [
    "RI_mean"
    "RI_std"
    "RI_cv"
    ];

mass_3D = [
    "dry_mass_pg"
    "drymass_density_mg_per_ml"
    ];

texture_3D = texture_2D;

feat_2D = [morphology_2D; intensity_2D; texture_2D];
feat_3D = [morphology_3D; intensity_3D; mass_3D; texture_3D];

%% Check Feature Existence=================================================
needCols = unique([feat_2D; feat_3D]);
missingCols = needCols(~ismember(needCols, string(T.Properties.VariableNames)));
assert(isempty(missingCols), ['Missing columns: ' strjoin(cellstr(missingCols), ', ')]);

% Storage
nMod = numel(modNames);
sampleIDByMod = cell(nMod,1);
umapByMod     = cell(nMod,1);
groupStrByMod = cell(nMod,1);

silScore = nan(nMod,1);
chScore  = nan(nMod,1);
dbiScore = nan(nMod,1);
mixScore = nan(nMod,1);
permR2   = nan(nMod,1);
permP    = nan(nMod,1);
permF    = nan(nMod,1);
ariScore = nan(nMod,1);
acScore  = nan(nMod,1);

featureNamesByMod = cell(nMod,1);
XrawByMod         = cell(nMod,1);
grpByMod          = cell(nMod,1);

TabAbl_2DBF  = table();
TabAbl_25DBF = table();
TabAbl_HT    = table();

%% Main Loop===============================================================

for mi = 1:nMod

    modName = modNames{mi};

    fprintf('\n====================\n');
    fprintf('Modality: %s\n', modName);

    idxMod = strcmp(cellstr(T.modalityLabel), modName);
    Tm = T(idxMod,:);
    % sample ID column 자동 탐색
    candidateIDVars = ["sample","sample_id","sampleID","filename","file","name"];
    idVar = "";
    
    for vv = 1:numel(candidateIDVars)
        if ismember(candidateIDVars(vv), string(Tm.Properties.VariableNames))
            idVar = candidateIDVars(vv);
            break
        end
    end
    
    if idVar == ""
        % 없으면 row 번호라도 임시로 부여
        sampleID = "row_" + string((1:height(Tm)).');
    else
        sampleID = string(Tm.(idVar));
    end

    fprintf('Rows in modality = %d\n', height(Tm));
    assert(height(Tm) > 0, 'No rows found for modality: %s', modName);

    if strcmp(modName,'2DBF') || strcmp(modName,'2.5DBF')
        allFeat = feat_2D;
    elseif strcmp(modName,'HT')
        allFeat = feat_3D;
    else
        error('Unknown modality: %s', modName);
    end

    Xs = double(table2array(Tm(:, cellstr(allFeat))));
    Gs = string(Tm.group);

    % 1) remove features mostly invalid
    badMask = isnan(Xs) | isinf(Xs);
    badFrac = mean(badMask, 1);
    keepFeat = badFrac <= 0.30;

    fprintf('\nFeature validity check for %s\n', modName);
    for kk = 1:numel(allFeat)
        fprintf('  %-30s | bad = %3d / %3d (%.1f%%) | %s\n', ...
            allFeat(kk), sum(badMask(:,kk)), size(Xs,1), 100*badFrac(kk), string(keepFeat(kk)));
    end

    removedFeat = allFeat(~keepFeat);
    if ~isempty(removedFeat)
        fprintf('Removed features in %s due to too many NaN/Inf:\n', modName);
        disp(removedFeat);
    end

    Xs = Xs(:, keepFeat);
    allFeat = allFeat(keepFeat);

    % 2) remove rows still invalid
    rowBad = ismissing(Gs) | any(isnan(Xs) | isinf(Xs), 2);
    Xs(rowBad,:) = [];
    Gs(rowBad)   = [];
    sampleID(rowBad) = [];

    fprintf('Remaining valid rows for %s = %d\n', modName, size(Xs,1));
    assert(size(Xs,1) >= 5, 'Too few valid rows for modality: %s', modName);
    assert(size(Xs,2) >= 2, 'Too few valid features for modality: %s', modName);

    % 3) remove zero-variance features
    featStd = std(Xs, 0, 1, 'omitnan');
    keepVar = featStd > 0 & ~isnan(featStd);

    removedZeroVar = allFeat(~keepVar);
    if ~isempty(removedZeroVar)
        fprintf('Removed zero-variance features in %s:\n', modName);
        disp(removedZeroVar);
    end

    Xs = Xs(:, keepVar);
    allFeat = allFeat(keepVar);

    assert(size(Xs,2) >= 2, 'Too few non-constant features for modality: %s', modName);

    % z-score
    mu = mean(Xs, 1, 'omitnan');
    sg = std(Xs, 0, 1, 'omitnan');
    Xs = (Xs - mu) ./ sg;

    [~,~,grpNum] = unique(Gs);

    featureNamesByMod{mi} = cellstr(allFeat);
    XrawByMod{mi}         = Xs;
    grpByMod{mi}          = grpNum;
    sampleIDByMod{mi}     = sampleID;
    groupStrByMod{mi}     = Gs;

    % quantitative scores
    if CALC_ARI || CALC_AC
        nClust = numel(unique(grpNum));
        cluPred = kmeans(Xs, nClust, ...
            'Replicates', ARI_CLUSTER_REPS, ...
            'Distance', 'sqeuclidean', ...
            'Display', 'off');
    end

    if CALC_ARI
        ariScore(mi) = adjusted_rand_index(grpNum, cluPred);
    end

    if CALC_AC
        uTrue = unique(grpNum);
        uPred = unique(cluPred);
        nTrue = numel(uTrue);
        nPred = numel(uPred);
        C = zeros(nTrue, nPred);

        for ii = 1:nTrue
            for jj = 1:nPred
                C(ii,jj) = sum((grpNum == uTrue(ii)) & (cluPred == uPred(jj)));
            end
        end

        if nTrue ~= nPred
            error('AC calculation requires same number of true groups and predicted clusters.');
        end

        permIdx = perms(1:nPred);
        bestCorrect = -inf;

        for pp = 1:size(permIdx,1)
            thisCorrect = 0;
            for rr = 1:nTrue
                thisCorrect = thisCorrect + C(rr, permIdx(pp,rr));
            end
            if thisCorrect > bestCorrect
                bestCorrect = thisCorrect;
            end
        end

        acScore(mi) = bestCorrect / numel(grpNum);
    end

    if CALC_PERMANOVA
        [pVal, R2, Fstat] = permanova1_raw(Xs, grpNum, nPerm);
        permR2(mi) = R2;
        permP(mi)  = pVal;
        permF(mi)  = Fstat;
    end

    if CALC_DBI || CALC_CH
        [dbiVal, chVal] = cluster_validity_scores(Xs, grpNum);
        dbiScore(mi) = dbiVal;
        chScore(mi)  = chVal;
    end

    if CALC_KNN_MIX
        [~, mixMean] = knn_mixing_score(Xs, grpNum, knn_k);
        mixScore(mi) = mixMean;
    end

    if CALC_SILHOUETTE
        silVec = silhouette(Xs, grpNum);
        silScore(mi) = mean(silVec, 'omitnan');
    end

    fprintf('%s | PERMANOVA R2=%.4f | p=%.4g | F=%.4f | ARI=%.4f | DBI=%.4f | CH=%.4f | mix=%.4f | sil=%.4f\n', ...
        modName, permR2(mi), permP(mi), permF(mi), ariScore(mi), dbiScore(mi), chScore(mi), mixScore(mi), silScore(mi));
end
    
%% PLOTS=================================================================== 
% UMAP Panel

if DRAW_UMAP_PANEL

    figure('Color','w','Units','centimeters','Position',[2 2 34 24]);
    tiledlayout(3,4,'Padding','compact','TileSpacing','compact');

    for mi = 1:nMod

        modName = modNames{mi};
        Xs = XrawByMod{mi};
        grpNum = grpByMod{mi};

        Gs = strings(size(grpNum));
        Gs(grpNum==1) = "D0";
        Gs(grpNum==2) = "D1";
        Gs(grpNum==3) = "D2";

        try
            Ys = run_umap(Xs, ...
                'n_components', 2, ...
                'n_neighbors', umap_n_neighbors, ...
                'min_dist', umap_min_dist, ...
                'metric', 'euclidean', ...
                'randomize', false, ...
                'verbose', 'none');
        catch
            fprintf('Custom UMAP options failed for %s. Falling back to run_umap(Xs).\n', modName);
            Ys = run_umap(Xs);
        end
        umapByMod{mi} = Ys;
        U1 = Ys(:,1);
        U2 = Ys(:,2);

        idx0 = strcmp(cellstr(Gs), 'D0');
        idx1 = strcmp(cellstr(Gs), 'D1');
        idx2 = strcmp(cellstr(Gs), 'D2');

        padFrac = 0.08;

        xmin = min(U1); xmax = max(U1);
        ymin = min(U2); ymax = max(U2);

        xr = xmax - xmin;
        yr = ymax - ymin;
        rangeXY = max(xr, yr);
        if rangeXY == 0
            rangeXY = 1;
        end

        xc = (xmin + xmax)/2;
        yc = (ymin + ymax)/2;

        xL = [xc - rangeXY/2*(1+padFrac), xc + rangeXY/2*(1+padFrac)];
        yL = [yc - rangeXY/2*(1+padFrac), yc + rangeXY/2*(1+padFrac)];

        nexttile((mi-1)*4 + 1); hold on
        scatter(U1(idx0),U2(idx0),mkMain,colD0,'o','filled','MarkerEdgeColor',colEdge,'LineWidth',0.8);
        scatter(U1(idx1),U2(idx1),mkMain,colD1,'o','filled','MarkerEdgeColor',colEdge,'LineWidth',0.8);
        scatter(U1(idx2),U2(idx2),mkMain,colD2,'o','filled','MarkerEdgeColor',colEdge,'LineWidth',0.8);
        axis equal; xlim(xL); ylim(yL); box on
        set(gca, 'XTick', [], 'YTick', [], 'XTickLabel', [], 'YTickLabel', [])

        nexttile((mi-1)*4 + 2); hold on
        scatter(U1(~idx0),U2(~idx0),mkGray,colGray,'o','filled','MarkerEdgeColor','none');
        scatter(U1(idx0),U2(idx0),mkMain,colD0,'o','filled','MarkerEdgeColor','none');
        axis equal; xlim(xL); ylim(yL); box on
        set(gca, 'XTick', [], 'YTick', [], 'XTickLabel', [], 'YTickLabel', [])

        nexttile((mi-1)*4 + 3); hold on
        scatter(U1(~idx1),U2(~idx1),mkGray,colGray,'o','filled','MarkerEdgeColor','none');
        scatter(U1(idx1),U2(idx1),mkMain,colD1,'o','filled','MarkerEdgeColor','none');
        axis equal; xlim(xL); ylim(yL); box on
        set(gca, 'XTick', [], 'YTick', [], 'XTickLabel', [], 'YTickLabel', [])

        nexttile((mi-1)*4 + 4); hold on
        scatter(U1(~idx2),U2(~idx2),mkGray,colGray,'o','filled','MarkerEdgeColor','none');
        scatter(U1(idx2),U2(idx2),mkMain,colD2,'o','filled','MarkerEdgeColor','none');
        axis equal; xlim(xL); ylim(yL); box on
        set(gca, 'XTick', [], 'YTick', [], 'XTickLabel', [], 'YTickLabel', [])
    end
end

% QUANTITATIVE BAR PLOTS


xlab = categorical(modNames);
xlab = reordercats(xlab, modNames);

if CALC_PERMANOVA
    figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
    bar(xlab, permR2);
    ylabel('PERMANOVA R^2');
    ylim([0 0.6]); box on
    title(sprintf('PERMANOVA R^2 (%d perms)', nPerm));
end

if CALC_DBI
    figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
    bar(xlab, dbiScore);
    ylabel('Davies-Bouldin Index');
    box on
    title('DBI (lower is better)');
end

if CALC_CH
    figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
    bar(xlab, chScore);
    ylabel('Calinski-Harabasz Index');
    box on
    title('CH (higher is better)');
end

if CALC_KNN_MIX
    figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
    bar(xlab, mixScore);
    ylabel(sprintf('kNN mixing (k=%d)', knn_k));
    ylim([0 1]); box on
    title('kNN mixing (lower is better)');
end

if CALC_SILHOUETTE
    figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
    bar(xlab, silScore);
    ylabel('Mean silhouette');
    ylim([0 0.5]); box on
    title('Silhouette (raw feature space)');
end

if CALC_ARI
    figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
    bar(xlab, ariScore);
    ylabel('Adjusted Rand Index (ARI)');
    ylim([-0.2 1]); box on
    title('ARI: label-cluster agreement');
end
if CALC_AC
    figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
    bar(xlab, acScore);
    ylabel('Clustering Accuracy (AC)');
    ylim([0 1]);
    box on
    title('AC: best label-cluster matching');
end

% Summary
fprintf('\n===== SUMMARY =====\n');
for mi = 1:nMod
    fprintf('%s : PERMANOVA R2=%.4f, p=%.4g, F=%.4f, AC=%.4f, ARI=%.4f, DBI=%.4f, CH=%.4f, mix=%.4f, sil=%.4f\n', ...
        modNames{mi}, permR2(mi), permP(mi), permF(mi), acScore(mi), ariScore(mi), dbiScore(mi), chScore(mi), mixScore(mi), silScore(mi));
end

%% Ablation study==========================================================

if RUN_ABLATION

    [TabAbl_2DBF, TabAbl_25DBF, TabAbl_HT] = run_ablation_analysis( ...
        XrawByMod, grpByMod, featureNamesByMod, modNames, ...
        morphology_2D, intensity_2D, texture_2D, ...
        morphology_3D, intensity_3D, mass_3D, texture_3D, ...
        nPerm, knn_k, ARI_CLUSTER_REPS);

    hasAbl1 = ~isempty(TabAbl_2DBF)  && ismember('Feature', TabAbl_2DBF.Properties.VariableNames);
    hasAbl2 = ~isempty(TabAbl_25DBF) && ismember('Feature', TabAbl_25DBF.Properties.VariableNames);
    hasAbl3 = ~isempty(TabAbl_HT)    && ismember('Feature', TabAbl_HT.Properties.VariableNames);

    fprintf('\nAblation table check:\n');
    fprintf('  2DBF   : %d\n', hasAbl1);
    fprintf('  2.5DBF : %d\n', hasAbl2);
    fprintf('  HT     : %d\n', hasAbl3);

        %% ============================================================
        % HT ONLY: leave-one-feature-out effect on PERMANOVA / silhouette
        %% ============================================================

        RUN_HT_FEATURE_DROP_ANALYSIS = true;

        if RUN_HT_FEATURE_DROP_ANALYSIS

            miHT = find(strcmp(modNames, 'HT'), 1);
            assert(~isempty(miHT), 'HT modality not found.');

            Xfull     = XrawByMod{miHT};
            grpNum    = grpByMod{miHT};
            featNames = string(featureNamesByMod{miHT});

            assert(~isempty(Xfull), 'HT feature matrix is empty.');
            assert(size(Xfull,2) >= 2, 'HT needs at least 2 features.');

            nFeat = size(Xfull, 2);

            [~, r2Base, Fbase] = permanova1_raw(Xfull, grpNum, nPerm);
            silBaseVec = silhouette(Xfull, grpNum);
            silBase    = mean(silBaseVec, 'omitnan');

            R2_without  = nan(nFeat,1);
            F_without   = nan(nFeat,1);
            Sil_without = nan(nFeat,1);

            DeltaR2_drop  = nan(nFeat,1);
            DeltaSil_drop = nan(nFeat,1);

            for fj = 1:nFeat
                keepIdx = true(1, nFeat);
                keepIdx(fj) = false;

                Xabl = Xfull(:, keepIdx);

                [~, r2Abl, Fabl] = permanova1_raw(Xabl, grpNum, nPerm);
                silAblVec = silhouette(Xabl, grpNum);
                silAbl    = mean(silAblVec, 'omitnan');

                R2_without(fj)  = r2Abl;
                F_without(fj)   = Fabl;
                Sil_without(fj) = silAbl;

                DeltaR2_drop(fj)  = r2Base - r2Abl;
                DeltaSil_drop(fj) = silBase - silAbl;
            end

            featFamily = strings(nFeat,1);
            for fj = 1:nFeat
                feat = featNames(fj);

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
            end

            TabHTdrop = table( ...
                featNames(:), featFamily, ...
                repmat(r2Base, nFeat, 1), R2_without, DeltaR2_drop, ...
                repmat(silBase, nFeat, 1), Sil_without, DeltaSil_drop, ...
                'VariableNames', { ...
                'Feature', 'FeatureFamily', ...
                'R2_full', 'R2_without_feature', 'DeltaR2_drop', ...
                'Sil_full', 'Sil_without_feature', 'DeltaSil_drop'});

            TabHTdrop_R2  = sortrows(TabHTdrop, 'DeltaR2_drop', 'descend');
            TabHTdrop_Sil = sortrows(TabHTdrop, 'DeltaSil_drop', 'descend');

            fprintf('\n============================================================\n');
            fprintf('HT feature-drop analysis\n');
            fprintf('Baseline PERMANOVA R2 = %.6f\n', r2Base);
            fprintf('Baseline PERMANOVA F  = %.6f\n', Fbase);
            fprintf('Baseline silhouette   = %.6f\n', silBase);
            fprintf('============================================================\n');

            fprintf('\nTop HT features by PERMANOVA R2 drop:\n');
            disp(TabHTdrop_R2(:, {'Feature','FeatureFamily','R2_without_feature','DeltaR2_drop'}));

            fprintf('\nTop HT features by silhouette drop:\n');
            disp(TabHTdrop_Sil(:, {'Feature','FeatureFamily','Sil_without_feature','DeltaSil_drop'}));

            TabHT_feature_drop = TabHTdrop;

        end

                tabsPosForLim = {TabAbl_2DBF, TabAbl_25DBF, TabAbl_HT};

        allValsR2_pos = [];
        allValsSil_pos = [];

        for miLim = 1:3
            TabLim = tabsPosForLim{miLim};

            if isempty(TabLim)
                continue
            end

            if ismember('DeltaR2', TabLim.Properties.VariableNames)
                valsR2 = TabLim.DeltaR2(TabLim.DeltaR2 > 0);
                valsR2 = valsR2(~isnan(valsR2));
                allValsR2_pos = [allValsR2_pos; valsR2];
            end

            if ismember('DeltaSil', TabLim.Properties.VariableNames)
                valsSil = TabLim.DeltaSil(TabLim.DeltaSil > 0);
                valsSil = valsSil(~isnan(valsSil));
                allValsSil_pos = [allValsSil_pos; valsSil];
            end
        end

        if isempty(allValsR2_pos)
            yLimR2_pos = [0 0.01];
        else
            lo = min(allValsR2_pos);
            hi = max(allValsR2_pos);
            pad = 0.15 * (hi - lo + eps);
            yLimR2_pos = [max(0, lo - pad), hi + pad];
        end

        if isempty(allValsSil_pos)
            yLimSil_pos = [0 0.01];
        else
            lo = min(allValsSil_pos);
            hi = max(allValsSil_pos);
            pad = 0.15 * (hi - lo + eps);
            yLimSil_pos = [max(0, lo - pad), hi + pad];
        end
        
        
        % FAMILY-LEVEL SUMMARY
        tabs = {TabAbl_2DBF, TabAbl_25DBF, TabAbl_HT};
        modLabels = {'2DBF','2.5DBF','HT'};

        familyOrder = [
            "Morphology", ...
            "Intensity / RI statistics", ...
            "Texture", ...
            "Mass", ...
            "Other" ...
            ];

        nFam = numel(familyOrder);

        famMatRawSil  = zeros(nMod, nFam);
        famMatNormSil = zeros(nMod, nFam);

        for mi = 1:nMod
            Tab = tabs{mi};

            if isempty(Tab) || ~ismember('FeatureFamily', Tab.Properties.VariableNames)
                continue
            end

            contrib = Tab.DeltaSil;
            contrib(contrib < 0) = 0;

            for fi = 1:nFam
                idx = Tab.FeatureFamily == familyOrder(fi);
                famMatRawSil(mi, fi) = sum(contrib(idx), 'omitnan');
            end

            rowSum = sum(famMatRawSil(mi,:), 2);
            if rowSum > 0
                famMatNormSil(mi,:) = 100 * famMatRawSil(mi,:) / rowSum;
            else
                famMatNormSil(mi,:) = zeros(1,nFam);
            end
        end

        figure('Color','w','Units','centimeters','Position',[2 2 14 8]);
        bar(famMatNormSil, 'stacked', 'LineWidth', 0.8);
        box on
        ylabel('Contribution composition (%)');
        xticks(1:nMod);
        xticklabels(modLabels);
        ylim([0 100]);
        legend(cellstr(familyOrder), 'Location','eastoutside', 'Box','off');
        title('Feature-family composition of positive \DeltaSil', 'Interpreter','tex');

        for mi = 1:nMod
            y0 = 0;
            for fi = 1:nFam
                val = famMatNormSil(mi,fi);
                if val > 1
                    text(mi, y0 + val/2, sprintf('%.0f%%', val), ...
                        'HorizontalAlignment','center', ...
                        'VerticalAlignment','middle', ...
                        'FontSize',8);
                end
                y0 = y0 + val;
            end
        end

        famMatRawR2  = zeros(nMod, nFam);
        famMatNormR2 = zeros(nMod, nFam);

        for mi = 1:nMod
            Tab = tabs{mi};

            if isempty(Tab) || ~ismember('FeatureFamily', Tab.Properties.VariableNames)
                continue
            end

            contrib = Tab.DeltaR2;
            contrib(contrib < 0) = 0;

            for fi = 1:nFam
                idx = Tab.FeatureFamily == familyOrder(fi);
                famMatRawR2(mi, fi) = sum(contrib(idx), 'omitnan');
            end

            rowSum = sum(famMatRawR2(mi,:), 2);
            if rowSum > 0
                famMatNormR2(mi,:) = 100 * famMatRawR2(mi,:) / rowSum;
            else
                famMatNormR2(mi,:) = zeros(1,nFam);
            end
        end

        figure('Color','w','Units','centimeters','Position',[2 2 14 8]);
        bar(famMatNormR2, 'stacked', 'LineWidth', 0.8);
        box on
        ylabel('Contribution composition (%)');
        xticks(1:nMod);
        xticklabels(modLabels);
        ylim([0 100]);
        legend(cellstr(familyOrder), 'Location','eastoutside', 'Box','off');
        title('Feature-family composition of positive \DeltaR^2', 'Interpreter','tex');

        for mi = 1:nMod
            y0 = 0;
            for fi = 1:nFam
                val = famMatNormR2(mi,fi);
                if val > 1
                    text(mi, y0 + val/2, sprintf('%.0f%%', val), ...
                        'HorizontalAlignment','center', ...
                        'VerticalAlignment','middle', ...
                        'FontSize',8);
                end
                y0 = y0 + val;
            end
        end

        tabsPos = {TabAbl_2DBF, TabAbl_25DBF, TabAbl_HT};
        modLabelsPos = {'2D BF', '2.5D BF', 'HT'};

        colMorph    = [0.80 0.80 0.80];
        colNonMorph = [0.85 0.15 0.15];

        figure('Color','w','Units','centimeters','Position',[2 2 24 8]);
        tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

        for mi = 1:3
            Tab = tabsPos{mi};

            if isempty(Tab) || ~ismember('DeltaSil', Tab.Properties.VariableNames)
                nexttile;
                axis off
                title(modLabelsPos{mi});
                text(0.5, 0.5, 'No valid ablation table', ...
                    'HorizontalAlignment','center', ...
                    'VerticalAlignment','middle');
                continue
            end

            TabPos = Tab(Tab.DeltaSil > 0, :);
            TabPos = sortrows(TabPos, 'DeltaSil', 'descend');

            nexttile;
            hold on;

            if isempty(TabPos)
                title(modLabelsPos{mi});
                text(0.5, 0.5, 'No positive \DeltaSil', ...
                    'HorizontalAlignment','center', ...
                    'VerticalAlignment','middle');
                axis off
                continue
            end

            barColors = zeros(height(TabPos), 3);

            for ii = 1:height(TabPos)
                fam = string(TabPos.FeatureFamily(ii));
                if fam == "Morphology"
                    barColors(ii,:) = colMorph;
                else
                    barColors(ii,:) = colNonMorph;
                end
            end

            b = bar(TabPos.DeltaSil, 'FaceColor','flat', 'EdgeColor','k', 'LineWidth',0.6);
            b.CData = barColors;

            xticks(1:height(TabPos));
            xticklabels(TabPos.Feature);
            xtickangle(90);

            ylabel('\DeltaSil');
            title(modLabelsPos{mi});
            box on
            ylim(yLimSil_pos);
        end

        title(tl, 'Features with positive \DeltaSil', 'Interpreter','tex');

        p1 = patch(nan, nan, colMorph, 'EdgeColor','k');
        p2 = patch(nan, nan, colNonMorph, 'EdgeColor','k');

        lg1 = legend([p1 p2], {'Morphology','Non-morphology'}, ...
            'Orientation','vertical', ...
            'Box','off');
        lg1.Layout.Tile = 'east';

        figure('Color','w','Units','centimeters','Position',[2 2 24 8]);
        tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

        for mi = 1:3
            Tab = tabsPos{mi};

            if isempty(Tab) || ~ismember('DeltaR2', Tab.Properties.VariableNames)
                nexttile;
                axis off
                title(modLabelsPos{mi});
                text(0.5, 0.5, 'No valid ablation table', ...
                    'HorizontalAlignment','center', ...
                    'VerticalAlignment','middle');
                continue
            end

            TabPos = Tab(Tab.DeltaR2 > 0, :);
            TabPos = sortrows(TabPos, 'DeltaR2', 'descend');

            nexttile;
            hold on;

            if isempty(TabPos)
                title(modLabelsPos{mi});
                text(0.5, 0.5, 'No positive \DeltaR^2', ...
                    'HorizontalAlignment','center', ...
                    'VerticalAlignment','middle');
                axis off
                continue
            end

            barColors = zeros(height(TabPos), 3);

            for ii = 1:height(TabPos)
                fam = string(TabPos.FeatureFamily(ii));
                if fam == "Morphology"
                    barColors(ii,:) = colMorph;
                else
                    barColors(ii,:) = colNonMorph;
                end
            end

            b = bar(TabPos.DeltaR2, 'FaceColor','flat', 'EdgeColor','k', 'LineWidth',0.6);
            b.CData = barColors;

            xticks(1:height(TabPos));
            xticklabels(TabPos.Feature);
            xtickangle(90);

            ylabel('\DeltaR^2');
            title(modLabelsPos{mi});
            box on
            ylim(yLimR2_pos);
        end

        title(tl, 'Features with positive \DeltaR^2', 'Interpreter','tex');

        p1 = patch(nan, nan, colMorph, 'EdgeColor','k');
        p2 = patch(nan, nan, colNonMorph, 'EdgeColor','k');

        lg2 = legend([p1 p2], {'Morphology','Non-morphology'}, ...
            'Orientation','vertical', ...
            'Box','off');
        lg2.Layout.Tile = 'east';

    else
        warning('No valid ablation tables found. Skipping heatmap/family summary plots.');
    end


