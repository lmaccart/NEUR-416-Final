%% NEUR 416 Final Project - Orientation Column Simulation
% Simulates an orientation column using a recurrent neural network
% trained with subtractive normalization (Hebbian learning).
clear; close all; clc;

%% ==================== PARAMETERS ====================
imgSz = 16;                          % image size (16x16 pixels)
nInputs = imgSz^2;                   % 256 input neurons
nOutputs = 32;                       % 32 output neurons
nOrientations = 16;                  % number of training orientations
nTrials = 40000;                     % training trials
cyclesPerImage = 3;                  % spatial frequency
K = 2 * pi * cyclesPerImage / 2;     % spatial freq for coords in [-1,1] (span=2)
noiseSD = 0.05;                      % noise standard deviation
eta = 0.001;                         % learning rate

% Lateral inhibition parameters
sigma_E = 1;
sigma_I = 4;

% 16 evenly spaced orientations from 0 to 180 (excluding 180)
orientations = linspace(0, 180, nOrientations + 1);
orientations = orientations(1:end-1);  % [0, 11.25, 22.5, ..., 168.75]

%% ==================== LATERAL INHIBITION MATRIX ====================
M = zeros(nOutputs);
for a = 1:nOutputs
    for b = 1:nOutputs
        if a ~= b
            dist = min(abs(a - b), nOutputs - abs(a - b));  % circular distance
            % Mexican hat lateral inhibition (Ricker wavelet)
            % Short-range excitation minus broad inhibition
            M(a,b) = exp(-dist^2 / (2 * sigma_E^2)) - exp(-dist^2 / (2 * sigma_I^2));
        end
    end
end

%% ==================== INITIALIZE WEIGHTS ====================
W = randn(nOutputs, nInputs) * 0.01;

%% ==================== TRAINING ====================
fprintf('Training for %d trials...\n', nTrials);
for t = 1:nTrials
    % Pick a random orientation
    idx = randi(nOrientations);
    theta = orientations(idx);

    % Generate stimulus
    stim = generate_grating(theta, imgSz, K, noiseSD);
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
testAngles = 0:179;          % 1-degree steps
nTestTrials = 30;            % repetitions per angle for noise averaging
meanResponses = zeros(nOutputs, length(testAngles));

for ai = 1:length(testAngles)
    theta = testAngles(ai);
    responses = zeros(nOutputs, nTestTrials);
    for trial = 1:nTestTrials
        stim = generate_grating(theta, imgSz, K, noiseSD);
        u = stim(:);
        v = max(0, W * u);
        v = max(0, v + M * v);
        responses(:, trial) = v;
    end
    meanResponses(:, ai) = mean(responses, 2);
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
nPsychTrials = 30;                        % trials per comparison

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
            stimRef = generate_grating(refTheta, imgSz, K, noiseSD);
            uRef = stimRef(:);
            vRef = max(0, W * uRef);
            vRef = max(0, vRef + M * vRef);

            % Population vector decoding for reference
            thetaDecRef = 0.5 * atan2(sum(vRef .* sin((pi/180) * (2 * prefOrient))), ...
                                       sum(vRef .* cos((pi/180) * (2 * prefOrient))));
            thetaDecRef = (180/pi) * (thetaDecRef);
            if thetaDecRef < 0; thetaDecRef = thetaDecRef + 180; end

            % Generate and decode test stimulus
            stimTest = generate_grating(testTheta, imgSz, K, noiseSD);
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
