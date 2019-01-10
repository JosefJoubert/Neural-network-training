trainFolder = 'C:\NSYNTH\piano_train_data\';
wavFiles = dir(fullfile(trainFolder, '*.wav'));
winSize=2048;
frameLength = 400; %ms
overlap=1024;
hiddenSize = 100;
gpu_train = false;
reLoad = true;

transform = 'FFT';
nr_files = 100;%size(wavFiles,1);
nr_mixed = 0;
train_data = cell(1,nr_files+nr_mixed);

[x,fs] = audioread(strcat(trainFolder,wavFiles(1).name));
[sin_array,cos_array] = calc_Target_arrays(winSize,fs);

tic
%transform files
if(reLoad)
    for file=1:nr_files
        file
        [x,fs] = audioread(strcat(trainFolder,wavFiles(file).name));
        fftarr = [];

        for v = 1:floor((fs*frameLength/1000)/(winSize-overlap))
            buffer = x(((v-1)*(winSize-overlap))+1:(((v-1)*(winSize-overlap))+winSize));
            targetbuffer = abs(fft(buffer));
            fftarr =[fftarr;targetbuffer(1:winSize/4)'];
        end
        train_data{file} = fftarr;
    end
end

fprintf('Time to read files: %d minutes and %f seconds\n', floor(toc/60), rem(toc,60));
%mix files
randarr = round(rand(nr_mixed,2)*nr_files+0.5);
strFiles = [];
for file = 1:nr_mixed
    file
    train_data{nr_files+file} = train_data{randarr(file,1)}+train_data{randarr(file,2)};
    strFiles = [strFiles,string(strcat(wavFiles(randarr(file,1)).name,' +  ',wavFiles(randarr(file,2)).name))];
end
fprintf('Time to mix files: %d minutes and %f seconds\n', floor(toc/60), rem(toc,60));



%train autoencoder
C_autoenc = GPU_TRAINER(train_data,10,10000,0,[1,512],0.00055);

save('autoenc','C_autoenc','winSize','overlap','frameLength');

timearr = linspace(0,frameLength,floor((fs*frameLength/1000)/(winSize-overlap)));
freqarr = linspace(0,fs/4,winSize/4);


nr_plots = 1;
step = nr_mixed/nr_plots;
for plt=1:nr_plots
    reconstructed = GPU_PREDICTOR(train_data{nr_files+plt*step},0);
    figure('NumberTitle', 'off', 'Name', strFiles(plt*step));
    subplot(2,1,1)
    surf(freqarr,timearr,train_data{nr_files+plt*step},'edgecolor','none');
    view(0, 90);
    xlabel('Frequency Bin (C3 -> E8)');
    ylabel('Time (ms)');
    title('Original Spectrum');
    subplot(2,1,2)
    surf(freqarr,timearr,reconstructed,'edgecolor','none');
    view(0, 90);
    xlabel('Frequency Bin (C3 -> E8)');
    ylabel('Time (ms)');
    title('Autoencoder Reconstructed Spectrum');
end


