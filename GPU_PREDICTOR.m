function out_data = GPU_PREDICTOR(in_data,verbose)
    loadFile = load('autoenc.mat');
    C_autoenc = loadFile.C_autoenc;
    in_data = gpuArray(in_data);
    in_data = in_data - C_autoenc.mu;
    in_data = in_data./C_autoenc.sigma_squared;
    
    params = [1,C_autoenc.params(2),C_autoenc.params(3),C_autoenc.params(4),C_autoenc.params(5),verbose];
    out_data = Predictor(in_data,params,C_autoenc.wHidden,C_autoenc.bHidden,C_autoenc.wOut,C_autoenc.bOut);
    
    out_data = out_data.*C_autoenc.sigma_squared;
    out_data = out_data + C_autoenc.mu;
end