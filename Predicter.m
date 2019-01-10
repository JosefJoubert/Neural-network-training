transform = 'FFT';

filepath1 = 'C:\NSYNTH\piano_test_data\keyboard_acoustic_004-057-075.wav';
%filepath1 = 'C:\NSYNTH\vocal_train_data\vocal_acoustic_002-062-075.wav';
filepath2 = 'C:\NSYNTH\piano_test_data\keyboard_acoustic_004-089-127.wav';

delay1 = 0; %ms
delay2 = 0; %ms

[file1,fs] = audioread(filepath1);
[file2,fs] = audioread(filepath2);


fftarr = [];

loadFile = load('autoenc.mat');
winSize = loadFile.winSize;
freqarr = linspace(0,fs/4,winSize/4);



overlap = loadFile.overlap;
frameLength = loadFile.frameLength;
autoenc = loadFile.C_autoenc;

%add delay

file1 = file1(1+fs*delay1/1000:1+fs*delay1/1000+fs*frameLength/1000+winSize-overlap);
file2 = file2(1+fs*delay2/1000:1+fs*delay2/1000+fs*frameLength/1000+winSize-overlap);
combinedfile = (file1+file2);

sound(file1,fs)

for v = 1:floor((fs*frameLength/1000)/(winSize-overlap))
    buffer = file1(((v-1)*(winSize-overlap))+1:(((v-1)*(winSize-overlap))+winSize));
    targetbuffer = abs(fft(buffer'));
    fftarr=[fftarr;targetbuffer(1:winSize/4)];
end

timearr = linspace(0,frameLength,floor((fs*frameLength/1000)/(winSize-overlap)));

%use autoencoder to predict outcome
reconstructed = gather(GPU_PREDICTOR(fftarr,0));

figure('NumberTitle', 'off', 'Name', filepath1);
subplot(3,1,1)
surf(freqarr,timearr,fftarr,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Original Spectrum');
subplot(3,1,2)
surf(freqarr,timearr,reconstructed,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Autoencoder Reconstructed Spectrum');
subplot(3,1,3)
separated = real(sqrt(reconstructed.*fftarr));
surf(freqarr,timearr,separated,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Separated Spectrum');

fftarr = [];
sound(file2,fs)

for v = 1:floor((fs*frameLength/1000)/(winSize-overlap))
    buffer = file2(((v-1)*(winSize-overlap))+1:(((v-1)*(winSize-overlap))+winSize));
    targetbuffer = abs(fft(buffer'));
    fftarr=[fftarr;targetbuffer(1:winSize/4)];
end

timearr = linspace(0,frameLength,floor((fs*frameLength/1000)/(winSize-overlap)));

%use autoencoder to predict outcome
reconstructed = GPU_PREDICTOR(fftarr,0);

figure('NumberTitle', 'off', 'Name', filepath2);
subplot(3,1,1)
surf(freqarr,timearr,fftarr,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Original Spectrum');
subplot(3,1,2)
surf(freqarr,timearr,reconstructed,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Autoencoder Reconstructed Spectrum');
subplot(3,1,3)
separated = real(sqrt(reconstructed.*fftarr));
surf(freqarr,timearr,separated,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Separated Spectrum');

fftarr = [];
sound(combinedfile,fs);

for v = 1:floor((fs*frameLength/1000)/(winSize-overlap))
    buffer = combinedfile(((v-1)*(winSize-overlap))+1:(((v-1)*(winSize-overlap))+winSize));
    targetbuffer = abs(fft(buffer'));
    fftarr=[fftarr;targetbuffer(1:winSize/4)];
end

timearr = linspace(0,frameLength,floor((fs*frameLength/1000)/(winSize-overlap)));

%use autoencoder to predict outcome
reconstructed = GPU_PREDICTOR(fftarr,verbose);

figure('NumberTitle', 'off', 'Name', 'Combined Spectrum');
subplot(3,1,1)
surf(freqarr,timearr,fftarr,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Original Spectrum');
subplot(3,1,2)
surf(freqarr,timearr,reconstructed,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Autoencoder Reconstructed Spectrum');
subplot(3,1,3)
separated = real(sqrt(reconstructed.*fftarr));
surf(freqarr,timearr,separated,'edgecolor','none');
view(0, 90);
xlabel('Frequency Bin (C3 -> E8)');
ylabel('Time (ms)');
title('Separated Spectrum');



