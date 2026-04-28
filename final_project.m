%% NEUR 416 Final Project - Orientation Column Simulation
% Simulates an orientation column using a recurrent neural network
% trained with subtractive normalization (Hebbian learning).
clear; close all; clc;



% Read in .mat file containing gratings created by Python script
% Data files are not included in Git repo. Run generate_grating.ipynb
% before final_project.m to generate all data files.
train_data = load('train_gratings.mat');
train_gratings = train_data.images;
train_orientations = train_data.orientations;

test_data = load('test_gratings.mat');
test_gratings = test_data.images;

psych_data = load('psychometric_gratings.mat');
psych_gratings = psych_data.images;

%% ==================== PARAMETERS ====================
imgSz = size(train_gratings, 2);                          % image size (16x16 pixels)
nInputs = size(train_gratings, 2) * size(train_gratings, 3);    % 256 input neurons
nOrientations = numel(unique(train_orientations));        % number of training orientations
nOutputs = 32;                                      % 32 output neurons
nTrials = size(train_gratings, 1);                        % training trials
cyclesPerImage = 3;                                 % spatial frequency
eta = 0.001;                                        % learning rate

% Added seed to make sure the results are reproducible
rngSeed = 416;
if isempty(rngSeed)
    rng('shuffle');
else
    rng(rngSeed);
end

% Input assertions
assert(ndims(train_gratings) == 3, 'train_gratings have dim 3');
assert(all(size(train_gratings, 2:3) == [16, 16]), 'train_gratings must be size [N x 16 x 16]');
assert(size(train_gratings, 1) == 40000, 'train_gratings must contain exactly 40,000 trials.');
assert(numel(train_orientations) == size(train_gratings, 1), ...
       'train_orientations length must match number of training trials.');

% Lateral inhibition parameters
sigma_E = 1;
sigma_I = 4;
g_E     = 0.6;
g_I     = 0.7;

%% ==================== LATERAL INHIBITION MATRIX ====================
% Build distance matrix, then compute M using the rubric formula
dist = zeros(nOutputs);
for a = 1:nOutputs
    for b = 1:nOutputs
        dist(a,b) = min(abs(a - b), nOutputs - abs(a - b));  % circular distance
    end
end
M = g_E * exp(-dist.^2 / (2 * sigma_E^2)) - g_I * exp(-dist.^2 / (2 * sigma_I^2));
% No self-connection
M(logical(eye(nOutputs))) = 0;

%% ==================== INITIALIZE WEIGHTS ====================
W = randn(nOutputs, nInputs) * 0.01;

%% ==================== TRAINING ====================
fprintf('Training for %d trials...\n', nTrials);
for t = 1:nTrials

    % Gather stimulus
    theta = train_orientations(t);
    stim = squeeze(train_gratings(t,:,:));

    % % Optionally display gratings to double-check
    % figure;
    % imshow(stim, []);
    % title(sprintf('Trial %d — %.1f°', t, theta));
    % drawnow;
    % waitfor(gcf);

    u = stim(:);  % flatten to 256x1

    % Feedforward: threshold-linear activation
    v = max(0, W * u);

    % Lateral inhibition (single step) + ReLU
    v = max(0, v + M * v);

    % Subtractive normalization learning rule
    % dW = v*u' - v*(sum(u))/N_u * ones
    sumU = sum(u);
    dW = v * u' - (v * sumU / nInputs) * ones(1, nInputs);
    W = W + eta * dW;

    % Bounding procedure
    % 1. Zero-mean each output neuron's weights
    W = W - mean(W, 2);

    % 2. L2 normalize each output neuron's weights
    norms = sqrt(sum(W.^2, 2)) + eps;
    W = W ./ norms;

    % Progress
    if mod(t, 10000) == 0
        fprintf('  Trial %d / %d complete\n', t, nTrials);
    end
end
fprintf('Training complete.\n\n');

%% ==================== VERIFICATION: PLOT RECEPTIVE FIELDS ====================
figure('Name', 'Learned Receptive Fields');
for a = 1:nOutputs
    subplot(4, 8, a);
    imagesc(reshape(W(a,:), imgSz, imgSz));
    colormap gray; axis image off;
    title(sprintf('N%d', a), 'FontSize', 7);
end
sgtitle('Learned Receptive Fields');

%% ==================== TESTING: TUNING CURVES ====================
fprintf('Computing tuning curves...\n');
nTestAngles = size(test_gratings, 1);
testAngles = 1:nTestAngles;
nTestTrials = size(test_gratings, 2);            % repetitions per angle for noise averaging
meanResponses = zeros(nOutputs, nTestAngles);

for theta = 0:nTestAngles-1
    responses = zeros(nOutputs, nTestTrials);
    for trial = 1:nTestTrials
        stim = squeeze(test_gratings(theta+1, trial, :, :));

        % % Optionally display gratings to double-check
        % if trial == 1
        %     figure;
        %     imshow(stim, []);
        %     title(sprintf('Trial %d — %.1f°', t, theta));
        %     drawnow;
        %     waitfor(gcf);
        % end

        u = stim(:);
        v = max(0, W * u);
        v = max(0, v + M * v);
        responses(:, trial) = v;
    end
    meanResponses(:, theta+1) = mean(responses, 2);
end

% Fit Gaussian to each neuron's tuning curve
% For circular data, find peak and fit around it
prefOrient = zeros(nOutputs, 1);  % preferred orientation (mean)
tuningWidth = zeros(nOutputs, 1); % tuning width (std)

for a = 1:nOutputs
    resp = meanResponses(a, :);

    % Use circular statistics to find preferred orientation
    % Double the angles to handle the 0-180 wraparound
    angles_rad = (pi/180) * (testAngles * 2);  % double for axial data
    sumSin = sum(resp .* sin(angles_rad));
    sumCos = sum(resp .* cos(angles_rad));
    prefAngle2 = atan2(sumSin, sumCos);
    prefOrient(a) = (180/pi) * (prefAngle2) / 2;
    if prefOrient(a) < 0
        prefOrient(a) = prefOrient(a) + 180;
    end

    % Fit Gaussian for tuning width using fminsearch (no toolbox needed)
    % Shift responses so peak is centered
    peakIdx = mod(round(prefOrient(a)), 180) + 1;  % 1-indexed, wraps correctly
    shiftedResp = circshift(resp, [0, 90 - peakIdx + 1]);
    xFit = (-89:90)';
    yFit = shiftedResp(:);  % column vector
    % Gaussian: a1*exp(-((x-b1)/c1)^2)
    gaussFun = @(p) sum((p(1) * exp(-((xFit - p(2)) ./ (abs(p(3)) + eps)).^2) - yFit).^2);
    p0 = [max(yFit), 0, 20];  % initial guess: [amplitude, center, width]
    try
        opts = optimset('Display', 'off', 'MaxFunEvals', 5000, 'MaxIter', 2000);
        pFit = fminsearch(gaussFun, p0, opts);
        tuningWidth(a) = abs(pFit(3) / sqrt(2));  % sigma
    catch
        tuningWidth(a) = NaN;
    end
end

% Print results
fprintf('\nTuning curve results:\n');
fprintf('%-10s %-20s %-20s\n', 'Neuron', 'Pref Orient (deg)', 'Tuning Width (deg)');
for a = 1:nOutputs
    fprintf('%-10d %-20.1f %-20.1f\n', a, prefOrient(a), tuningWidth(a));
end

% Plot tuning curves for a subset of neurons
figure('Name', 'Orientation Tuning Curves');
nPlot = min(9, nOutputs);
for i = 1:nPlot
    subplot(3, 3, i);
    plot(testAngles, meanResponses(i, :), 'k', 'LineWidth', 1);
    xlabel('Orientation (deg)');
    ylabel('Mean response');
    title(sprintf('Neuron %d', i));
    xlim([0 179]);
end
sgtitle('Orientation Tuning Curves');

%% ==================== PSYCHOMETRIC CURVES & JND ====================
fprintf('\nComputing psychometric curves...\n');
refOrients = [0, 30, 60, 90, 120, 150];  % reference orientations
deltaRange = -25:1:25;                    % test offsets in degrees
nPsychTrials = size(psych_gratings, 4);                        % trials per comparison

% Store psychometric data
psychData = zeros(length(refOrients), length(deltaRange));

for ri = 1:length(refOrients)
    refTheta = refOrients(ri);
    fprintf('  Reference orientation: %d deg\n', refTheta);

    for di = 1:length(deltaRange)
        delta = deltaRange(di);
        testTheta = mod(refTheta + delta, 180);

        nHigher = 0;
        for trial = 1:nPsychTrials
            % Generate and decode reference stimulus
            stimRef = squeeze(psych_gratings(ri, di, 1, trial, :, :));
            % stimRef = generate_grating(refTheta, imgSz, K, noiseSD);
            uRef = stimRef(:);
            vRef = max(0, W * uRef);
            vRef = max(0, vRef + M * vRef);

            % Population vector decoding for reference
            thetaDecRef = 0.5 * atan2(sum(vRef .* sin((pi/180) * (2 * prefOrient))), ...
                                      sum(vRef .* cos((pi/180) * (2 * prefOrient))));
            thetaDecRef = (180/pi) * (thetaDecRef);
            if thetaDecRef < 0; thetaDecRef = thetaDecRef + 180; end

            % Generate and decode test stimulus
            % stimTest = generate_grating(testTheta, imgSz, K, noiseSD);
            stimTest = squeeze(psych_gratings(ri, di, 2, trial, :, :));
            uTest = stimTest(:);
            vTest = max(0, W * uTest);
            vTest = max(0, vTest + M * vTest);

            thetaDecTest = 0.5 * atan2(sum(vTest .* sin((pi/180) * (2 * prefOrient))), ...
                                        sum(vTest .* cos((pi/180) * (2 * prefOrient))));
            thetaDecTest = (180/pi) * (thetaDecTest);
            if thetaDecTest < 0; thetaDecTest = thetaDecTest + 180; end

            % Compare: circular difference
            diff_decoded = thetaDecTest - thetaDecRef;
            % Wrap to [-90, 90]
            diff_decoded = mod(diff_decoded + 90, 180) - 90;

            if diff_decoded > 0
                nHigher = nHigher + 1;
            end
        end
        psychData(ri, di) = nHigher / nPsychTrials * 100;
    end
end

% Plot psychometric curves
figure('Name', 'Psychometric Curves');
colors = lines(length(refOrients));
hold on;
for ri = 1:length(refOrients)
    plot(deltaRange, psychData(ri, :), 'Color', colors(ri,:), 'LineWidth', 1.5);
end
yline(50, '--k');
xlabel('\Delta\theta (deg)');
ylabel('% judged stimulus 2 > stimulus 1');
title('Psychometric Curves Across Orientations');
legend(arrayfun(@(x) sprintf('%d', x), refOrients, 'UniformOutput', false), ...
       'Location', 'southeast');
hold off;

%% ==================== COMPUTE JND ====================
JND = zeros(length(refOrients), 1);

for ri = 1:length(refOrients)
    curve = psychData(ri, :) / 100;  % normalize to [0,1]

    % Fit sigmoid: f(x) = 1 / (1 + exp(-a*(x-b)))
    sigmoidFun = @(p, x) 1 ./ (1 + exp(-p(1) * (x - p(2))));
    p0 = [0.5, 0];  % initial guess: [slope, midpoint]
    try
        opts = optimset('Display', 'off');
        pFit = lsqcurvefit(sigmoidFun, p0, deltaRange', curve', ...
                           [0.01, -30], [5, 30], opts);

        % JND = (75th percentile - 25th percentile) / 2
        % 75th: 1/(1+exp(-a*(x-b))) = 0.75 => x = b + ln(3)/a
        % 25th: 1/(1+exp(-a*(x-b))) = 0.25 => x = b - ln(3)/a
        x75 = pFit(2) + log(3) / pFit(1);
        x25 = pFit(2) - log(3) / pFit(1);
        JND(ri) = (x75 - x25) / 2;
    catch
        % Fallback: interpolate
        idx75 = find(curve >= 0.75, 1, 'first');
        idx25 = find(curve >= 0.25, 1, 'first');
        if ~isempty(idx75) && ~isempty(idx25)
            JND(ri) = (deltaRange(idx75) - deltaRange(idx25)) / 2;
        else
            JND(ri) = NaN;
        end
    end
end

% Plot JND bar chart
figure('Name', 'JND Comparison');
bar(refOrients, JND);
xlabel('Center Orientation (deg)');
ylabel('JND (deg)');
title('Estimated JND across Center Orientations');
set(gca, 'XTick', refOrients);

% Print JND results
fprintf('\nJND Results:\n');
fprintf('%-20s %-15s\n', 'Reference (deg)', 'JND (deg)');
for ri = 1:length(refOrients)
    fprintf('%-20d %-15.2f\n', refOrients(ri), JND(ri));
end

fprintf('\nAll done!\n');
