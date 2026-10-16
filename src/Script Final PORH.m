clc
clear all
close all

%% Importation des Données
% Load file
data = readtable('BARIV030-M12.xlsx'); 

% Extract time and PORH signal from dataset
time = data{27:end, 3};  % Assume time is in the 3rd column
porh = data{27:end, 4};  % Assume PORH signal is in the 4th column

% Filter out non-numerical data and ensure compatibility
if iscell(time)
    time = cellfun(@str2double, time, 'UniformOutput', false);
    time = cell2mat(time(~cellfun(@isnan, time))); % Convert and filter NaNs
else
    time = time(~isnan(time)); % Remove NaN
end

if iscell(porh)
    porh = cellfun(@str2double, porh, 'UniformOutput', false);
    porh = cell2mat(porh(~cellfun(@isnan, porh))); % Convert and filter NaNs
else
    porh = porh(~isnan(porh)); % Remove NaN
end

% Ensure time and porh are vectors of the same length
min_len = min(length(time), length(porh));
time = time(1:min_len);
porh = porh(1:min_len);

% Plot the original PORH signal
figure;
%subplot(2, 1, 1);
plot(time, porh, 'b');
grid on;
xlabel('Time (s)');
ylabel('PORH Signal');
title('Original PORH Signal');

%% Sampling Frequency
fs = 32; % Sampling frequency in Hz (given)

%% Frequency-Domain Low-Pass Filtering
% Low-pass filter parameters
cutoff_freq = 2.0;        % Cutoff frequency (Hz)
n = length(porh);         % Signal length
f = (0:n-1)*(fs/n);       % Frequency vector
fft_signal = fft(porh);   % FFT of the signal

% Create a low-pass filter in the frequency domain
low_pass_filter = (f <= cutoff_freq) | (f >= (fs - cutoff_freq));
fft_signal(~low_pass_filter) = 0; % Zero out frequencies above the cutoff

% Inverse FFT to get the low-passed signal in the time domain
filtered_signal_time = real(ifft(fft_signal)); % Ensure a real-valued signal

% % Plot the time-domain filtered signal
% subplot(2, 1, 2);
% plot(time, filtered_signal_time, 'b');
% grid on;
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Low-Pass Filtered Signal (Time Domain)');

%% Remove Outliers Using a Moving Median Filter
% Define window size for the moving median filter (adjustable)
window_size = 200; % Choose based on the sampling rate and desired smoothness

% Compute the moving median of the signal
moving_median = movmedian(filtered_signal_time, window_size);

% Compute the absolute deviation from the moving median
deviation = abs(filtered_signal_time - moving_median);

% Define a threshold for outlier detection (e.g., 3 times the moving MAD)
threshold = 5 * movmad(filtered_signal_time, window_size);

% Identify outliers (deviation greater than threshold)
outliers = deviation > threshold;

% Replace outliers with the local moving median
filtered_signal_no_outliers = filtered_signal_time;
filtered_signal_no_outliers(outliers) = moving_median(outliers);

%% Moving Average Filter
window_sizes = [50, 100, 150, 250, 500]; % Ajuster la taille de la fenêtre pour le lissage

filtered_signals = struct();

for i = 1:length(window_sizes)
    ws = window_sizes(i);
    % Moving Average
    filtered_signals.movavg{i} = movmean(filtered_signal_no_outliers, ws);
    
    % Plot each filtered signal compared to the original signal in a new figure
    figure;
    plot(time, filtered_signal_no_outliers, 'b', 'DisplayName', 'Original Signal');
    hold on;
    plot(time, filtered_signals.movavg{i}, 'm', 'DisplayName', ['Moving Average Filtered (Window Size = ' num2str(ws) ')']);
    xlabel('Time');
    ylabel('Signal');
    legend;
    title(['Moving Average Filter with Window Size = ' num2str(ws)]);
    hold off;
end

%% Evaluate Filtering Technique
rmse = @(x, y) sqrt(mean((x - y).^2));

for i = 1:length(window_sizes)
    ws = window_sizes(i);
    filtered_signal = filtered_signals.movavg{i};
    rmse_value = rmse(porh, filtered_signal);
    fprintf('RMSE for window size %d: %.4f\n', ws, rmse_value);
end
%% Plot Results
figure;

% Original filtered signal
subplot(2, 1, 1);
plot(time, filtered_signal_time, 'b');
hold on;
plot(time(outliers), filtered_signal_time(outliers), 'ro');
grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title('Signal (With Outliers)');
legend('Signal', 'Outliers');

% Filtered signal after removing outliers
subplot(2, 1, 2);
plot(time, filtered_signal_no_outliers, 'g');
grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title('Signal (After Outlier Removal)');

%% Derivative to Detect Key Points
filtered_signal = filtered_signals.movavg{3}; % Example: use the third window size
% Compute the derivative of the filtered signal
signal_derivative = diff(filtered_signal) ./ diff(time);

% Smoothing the derivative to reduce noise
smooth_derivative = movmean(signal_derivative, 5); % Moving average with a window of 5

% Dynamic thresholds based on signal behavior
mean_derivative = mean(smooth_derivative);
std_derivative = std(smooth_derivative);

% Adjust thresholds for sensitivity
k = 1 ;  % Sensitivity factor
threshold_start = mean_derivative - k * std_derivative;  % Negative slope (start)

% Detect Occlusion Start (large negative slope)
occlusion_start_idx = find(smooth_derivative < threshold_start, 1, 'first');
if ~isempty(occlusion_start_idx)
    occlusion_start_time = time(occlusion_start_idx);
else
    error('Occlusion start not detected');
end

% Compute Rolling Averages and Slope
% Parameters for rolling averages
short_window = 20; % Short-term rolling average window
long_window = 400; % Long-term rolling average window

% Compute rolling averages
short_avg = movmean(filtered_signal, short_window);
long_avg = movmean(filtered_signal, long_window);

% Compute slope (ratio of short-term to long-term averages)
slope = short_avg ./ long_avg;

% Dynamic threshold for significant slope changes
threshold_end = 0.5; % Adjust as needed for sensitivity

% Detect the end of occlusion (positive slope after the start)
occlusion_end_idx = find(slope < threshold_end & time > occlusion_start_time, 1, 'first');
if ~isempty(occlusion_end_idx)
    occlusion_end_time = time(occlusion_end_idx);
else
    error('Occlusion end not detected');
end

% Plot Results
figure;

% Plot filtered signal with occlusion points
subplot(3, 1, 1);
plot(time, filtered_signal, 'b'); hold on;
xline(occlusion_start_time, '--r', 'Occlusion Start');
xline(occlusion_end_time, '--g', 'Occlusion End');
grid on;
title('Filtered Signal with Occlusion Points');
xlabel('Time (s)');
ylabel('Signal');
legend('Filtered Signal', 'Occlusion Start');

% Plot derivative with threshold
subplot(3, 1, 2);
plot(time(1:end-1), smooth_derivative, 'k'); hold on;
yline(threshold_start, '--r', 'Threshold Start');
scatter(time(occlusion_start_idx), smooth_derivative(occlusion_start_idx), 'r', 'filled', 'DisplayName', 'Occlusion Start');
grid on;
title('Derivative of Signal with Thresholds');
xlabel('Time (s)');
ylabel('Derivative');
legend('Signal Derivative', 'Start Threshold', 'Occlusion Start');

% Visualize slope
subplot(3, 1, 3);
plot(time, slope, 'k'); hold on;
ylim([0, 1.5]); % Set y-axis limits to [0, 1.5]
yline(threshold_end, '--g', 'Threshold End');
scatter(time(occlusion_end_idx), slope(occlusion_end_idx), 'g', 'filled', 'DisplayName', 'Occlusion End');
grid on;
xlabel('Time');
ylabel('Slope');
title('Slope (Ratio of Short-Term to Long-Term Averages)');
legend('Signal Slope Ratio', 'End Threshold', 'Occlusion End');

%% Biological Zero, Rest Flow, and Peak Flow
% Biological Zero
biological_zero = mean(filtered_signal(occlusion_start_idx:occlusion_end_idx)); % Average during occlusion

% Rest Flow
rest_flow = mean(filtered_signal(1:occlusion_start_idx - 1)); % Average before occlusion

% Peak Flow
peak_flow = max(filtered_signal(occlusion_end_idx:end)); % Maximum signal during hyperemia

%% Area Calculations
% Occlusion Area (OA)
occlusion_area = trapz(time(occlusion_start_idx:occlusion_end_idx), filtered_signal(occlusion_start_idx:occlusion_end_idx));

% Define range for hyperemia (after occlusion end)
hyperemia_range = occlusion_end_idx:length(filtered_signal);

% Find the maximum signal value in the hyperemia range and its index
[peak_flow, relative_max_idx] = max(filtered_signal(hyperemia_range));
max_idx = hyperemia_range(1) + relative_max_idx - 1; % Adjust to absolute index

% Time to Max (TM): Time from occlusion end to the peak value
time_to_max = time(max_idx) - occlusion_end_time;
% Define the range for hyperemia (after occlusion end)
hyperemia_range = occlusion_end_idx:length(filtered_signal);

% Hyperemia Area (HA)
hyperemia_area = trapz(time(occlusion_end_idx:end), filtered_signal(occlusion_end_idx:end));

%% Validation: Visualization of Areas
figure;
plot(time, filtered_signal, 'b', 'DisplayName', 'Filtered Signal');
hold on;

% Occlusion area (shaded)
fill([time(occlusion_start_idx:occlusion_end_idx); flip(time(occlusion_start_idx:occlusion_end_idx))], ...
     [filtered_signal(occlusion_start_idx:occlusion_end_idx); zeros(length(occlusion_start_idx:occlusion_end_idx), 1)], ...
     'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Occlusion Area (OA)');

% Extract the valid range of the signal
time_range = time(occlusion_end_idx:length(time)); 
signal_range = filtered_signal(occlusion_end_idx:length(filtered_signal));

% Fill hyperemia area until the end of the signal
fill([time_range; flip(time_range)], ...
     [signal_range; zeros(size(signal_range))], ...
     'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Hyperemia Area (HA)');

% Add key points
xline(occlusion_start_time, '--r', 'LineWidth', 1.5, 'DisplayName', 'Occlusion Start');
xline(occlusion_end_time, '--g', 'LineWidth', 1.5, 'DisplayName', 'Occlusion End');
xline(time(max_idx), '--k', 'LineWidth', 1.5, 'DisplayName', 'Time to Max');
scatter(time(max_idx), peak_flow, 'r', 'filled', 'DisplayName', 'Peak Flow');


legend;
xlabel('Time (s)');
ylabel('Filtered PORH Signal');
title('Key Metrics with Areas');
grid on;
hold off;

%% Save Computed Metrics
% Store the metrics in a table
metrics_table = table(occlusion_start_time, occlusion_end_time, time_to_max, ...
    occlusion_area, hyperemia_area, biological_zero, rest_flow, peak_flow, ...
    'VariableNames', {'OcclusionStartTime', 'OcclusionEndTime', 'TimeToMax', ...
    'OcclusionArea', 'HyperemiaArea', 'BiologicalZero', 'RestFlow', 'PeakFlow'});

% Save metrics to an Excel file
writetable(metrics_table, 'C:\Users\ELTEBOOK\Documents\MScT SCE Bio\M1\S7\internship\30-m12\computed_metrics.xlsx');

% Save metrics to a MAT file
%save('C:\Users\ELTEBOOK\Documents\MScT SCE Bio\M1\S7\internship\30-v1\computed_metrics.mat', 'metrics_table');

% Display metrics for validation
disp('Validation of Computed Metrics:');
disp(metrics_table);

