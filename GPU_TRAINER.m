function C_autoenc = GPU_TRAINER(data,batch_size,epochs,verbose,hidden_size,learning_rate)

%reset GPU device
reset(gpuDevice(1))

learning_rate_lower = learning_rate;
input_size = size(data{1});
input_scale = input_size(1)*input_size(2);
batches_per_epoch = floor(size(data,2)/batch_size);

tic
%calculate mean and covariance
mu = zeros(input_size(1),input_size(2));
sigma_squared = zeros(input_size(1),input_size(2));
for cell=1:size(data,2)
    sample = data{cell};
    for pixelw=1:input_size(1)
        for pixelh=1:input_size(2)
            mu(pixelw,pixelh) = mu(pixelw,pixelh) + sample(pixelw,pixelh);
            sigma_squared(pixelw,pixelh) = sigma_squared(pixelw,pixelh) + sample(pixelw,pixelh)^2;
        end
    end
end
mu = mu/size(data,2);
sigma_squared = sigma_squared/size(data,2) + 0.0000001;
for cell=1:size(data,2)
   data{cell} = data{cell}-mu;
   data{cell} = data{cell}./sigma_squared;
end

fprintf('Time to normalize: %d minutes and %f seconds\n', floor(toc/60), rem(toc,60));

gpuDevice();

C_autoenc = struct;
C_autoenc.wHidden = gpuArray(rand(input_size(1),input_size(2),hidden_size(1),hidden_size(2))*0.0001);
C_autoenc.bHidden = gpuArray(zeros(hidden_size(1),hidden_size(2)));
C_autoenc.wOut = gpuArray(rand(input_size(1),input_size(2),hidden_size(1),hidden_size(2))*0.0001);
C_autoenc.bOut = gpuArray(zeros(input_size(1),input_size(2)));
C_autoenc.params = [batch_size,input_size(2),input_size(1),hidden_size(1),hidden_size(2),verbose,learning_rate_lower];
C_autoenc.mu = mu;
C_autoenc.sigma_squared = sigma_squared;
Error_Over_Time = zeros(1,epochs)-1;
Error_data = 0;

figure();
err_plot = semilogy(Error_Over_Time);
xlim([0 epochs])
tic

for epoch = 1:epochs
    epoch_error = 0;
    for batch_nr = 1:batches_per_epoch
        batch = [];
        for x=1:batch_size
           batch = [batch,data{(batch_nr-1)*batch_size+x}];
        end
        batch = gpuArray(batch);
        [C_autoenc.wHidden,C_autoenc.bHidden,C_autoenc.wOut,C_autoenc.bOut,Error_data] = Trainer(batch,C_autoenc.params,C_autoenc.wHidden,C_autoenc.bHidden,C_autoenc.wOut,C_autoenc.bOut);
        last_error = sum(sum(sum(abs(gather(Error_data)))))/input_scale;
        epoch_error = epoch_error + last_error;
    end
    epoch_error = epoch_error/batches_per_epoch
    if(isnan(epoch_error) || isinf(epoch_error))
       fprintf("Error out of bounds.");
       return;
    end
    if(epoch_error < 10^-4)
        break;
    end
    Error_Over_Time(epoch) = epoch_error;
    set(err_plot,'YData',Error_Over_Time);
    title(strcat('Epoch: ',int2str(epoch)));
    drawnow limitrate
end
fprintf('Time to train: %d minutes and %f seconds\n', floor(toc/60), rem(toc,60));


C_autoenc.params(6) = 3;
Trainer(batch,C_autoenc.params,C_autoenc.wHidden,C_autoenc.bHidden,C_autoenc.wOut,C_autoenc.bOut);
clear Error_data  Error_Over_Time A B C batch

end