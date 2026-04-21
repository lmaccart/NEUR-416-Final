function stim = generate_grating(theta_deg, imgSz, K, noiseSD)
% GENERATE_GRATING Generate a contrast grating stimulus with random phase and noise.
%   stim = generate_grating(theta_deg, imgSz, K, noiseSD)
%
%   theta_deg - orientation in degrees
%   imgSz     - image size (e.g., 16 for 16x16)
%   K         - spatial frequency (radians per pixel)
%   noiseSD   - standard deviation of additive Gaussian noise

    theta = theta_deg * pi / 180;
    phi = rand * 2 * pi;  % random phase

    % Pixel coordinates from -1 to 1
    coords = linspace(-1, 1, imgSz);
    [x, y] = meshgrid(coords, coords);
    grating = cos(K * x * cos(theta) + K * y * sin(theta) - phi);

    % Add noise
    stim = grating + noiseSD * randn(imgSz);
end
