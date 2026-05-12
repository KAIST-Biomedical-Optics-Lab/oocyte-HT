%% 3D Dice + SSIM by compartment label
clear; close all; clc;

%% File pairs
sampleID = ["D0_oc10"; "D0_oc24"; "D0_oc35"; ...
            "D2_oc58"; "D2_oc65"; "D2_oc95"];

groupName = ["D0"; "D0"; "D0"; "D2"; "D2"; "D2"];

manualFiles = [
    "F:\010_Data\manual_seg\D0\oc10\manual_mask.tiff"
    "F:\010_Data\manual_seg\D0\oc24\manual_mask.tiff"
    "F:\010_Data\manual_seg\D0\oc35\manual_mask.tiff"
    "F:\010_Data\manual_seg\D2\oc58\manual_mask.tiff"
    "F:\010_Data\manual_seg\D2\oc65\manual_mask.tiff"
    "F:\010_Data\manual_seg\D2\oc95\manual_mask.tiff"
];

predFiles = [
    "F:\010_Data\manual_seg\D0\oc10\predicted_mask.tif"
    "F:\010_Data\manual_seg\D0\oc24\predicted.tif"
    "F:\010_Data\manual_seg\D0\oc35\predicted.tif"
    "F:\010_Data\manual_seg\D2\oc58\predicted.tif"
    "F:\010_Data\manual_seg\D2\oc65\predicted.tif"
    "F:\010_Data\manual_seg\D2\oc95\predicted.tif"
];

labelID = [1 2 3 4];
compName = ["ooplasm", "PVS", "ZP", "PB"];

%% Calculate Dice and SSIM
nSample = numel(sampleID);
nComp = numel(labelID);

Dice3D = nan(nSample, nComp);
SSIM3D = nan(nSample, nComp);

for i = 1:nSample

    fprintf('\n[%s]\n', sampleID(i));

    manualVol = tiffreadVolume(manualFiles(i));
    predVol   = tiffreadVolume(predFiles(i));

    if ~isequal(size(manualVol), size(predVol))
        error("[%s] manual and predicted mask sizes are different.", sampleID(i));
    end

    manualVol = double(manualVol);
    predVol   = double(predVol);

    for c = 1:nComp

        labelNow = labelID(c);

        manualMask = manualVol == labelNow;
        predMask   = predVol == labelNow;

        interVol = nnz(manualMask & predMask);
        sumVol   = nnz(manualMask) + nnz(predMask);

        if sumVol == 0
            Dice3D(i,c) = NaN;
        else
            Dice3D(i,c) = 2 * interVol / sumVol;
        end

        manualDouble = double(manualMask);
        predDouble   = double(predMask);

        try
            SSIM3D(i,c) = ssim(predDouble, manualDouble);
        catch
            nZ = size(manualDouble, 3);
            ssimEachZ = nan(nZ,1);

            for z = 1:nZ
                if any(manualDouble(:,:,z), 'all') || any(predDouble(:,:,z), 'all')
                    ssimEachZ(z) = ssim(predDouble(:,:,z), manualDouble(:,:,z));
                end
            end

            SSIM3D(i,c) = mean(ssimEachZ, 'omitnan');
        end

        fprintf('%s | Dice = %.4f | SSIM = %.4f\n', ...
            compName(c), Dice3D(i,c), SSIM3D(i,c));
    end
end

%% Save result table: wide format
ResultTable = table(sampleID, groupName);

for c = 1:nComp
    ResultTable.("Dice_" + compName(c)) = Dice3D(:,c);
    ResultTable.("SSIM_" + compName(c)) = SSIM3D(:,c);
end

disp(ResultTable);


%% -------------------------
% Plot: Dice vs SSIM + scatter
%% -------------------------
order = [1 2 3 4];  % ooplasm, PVS, ZP, PB
compName_ord = ["Ooplasm", "PVS", "ZP", "PB"];

Dice_plot = Dice3D(:, order);
SSIM_plot = SSIM3D(:, order);

nComp = numel(compName_ord);

meanVals = [
    mean(Dice_plot, 1, 'omitnan')
    mean(SSIM_plot, 1, 'omitnan')
]';

sdVals = [
    std(Dice_plot, 0, 1, 'omitnan')
    std(SSIM_plot, 0, 1, 'omitnan')
]';

figure('Color','w','Position',[250 200 650 520]);
hold on;

b = bar(meanVals, 0.65, 'grouped');

% 색
b(1).FaceColor = [0.70 0.68 0.68];   % Dice
b(2).FaceColor = [0.05 0.05 0.05];   % SSIM

b(1).EdgeColor = 'none';
b(2).EdgeColor = 'none';

%% error bar + scatter
for k = 1:2

    xEnd = b(k).XEndPoints;

    % error bar
    errorbar(xEnd, meanVals(:,k), sdVals(:,k), ...
        'k', 'LineStyle','none', ...
        'LineWidth',0.8, 'CapSize',5);

    % scatter
    for c = 1:nComp

        if k == 1
            vals = Dice_plot(:,c);
        else
            vals = SSIM_plot(:,c);
        end

        vals = vals(~isnan(vals));

        jitter = linspace(-0.04, 0.04, numel(vals));

        plot(xEnd(c) + jitter, vals, 'o', ...
            'MarkerSize',5.5, ...
            'MarkerFaceColor',[0.2 0.2 0.2], ...
            'MarkerEdgeColor',[0.2 0.2 0.2], ...
            'LineStyle','none');
    end
end

%% axis
ylim([0 1.0]);
xlim([0.5 nComp+0.5]);

xticks(1:nComp);
xticklabels(compName_ord);

yticks(0:0.2:1.0);

title('Manual vs AI-based segmentation comparison', ...
    'FontSize',15, 'FontWeight','normal');

legend({'Dice coefficient','3D SSIM'}, ...
    'Location','northoutside', ...
    'Orientation','horizontal', ...
    'Box','off');

box off;
grid off;

ax = gca;
ax.FontSize = 13;
ax.LineWidth = 0.8;
ax.TickDir = 'out';

%% save
outFig = "F:\010_Data\manual_seg\Dice_SSIM_compartment_bar_scatter.png";
exportgraphics(gcf, outFig, 'Resolution',300);