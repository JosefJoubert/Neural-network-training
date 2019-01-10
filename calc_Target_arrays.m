
function [sinarr,cosarr] = calc_Target_arrays(winSize,fs)
    winSize = 2048;
    fs = 16000;

    checkFreq = [164.81,174.61,184.997,195.998,207.65,220,233.08,246.94,261.63,277.18,296.67,311.13,329.63,349.23,369.994,391.995,415.31,440,466.16,493.88,523.25,554.37,587.33,622.25,659.255,698,739,783,830,880,932,987,1046,1108,1174,1244,1318,1396,1479,1567,1661,1760,1864,1975,2093,2217,2349,2489,2637,2793,2959,3135,3322,3520,3729,3951,4186];
    fractions = [0.98,0.99,1,1.01,1.02];
    sinarr = zeros(winSize,size(checkFreq,2)*size(fractions,2));
    cosarr = zeros(winSize,size(checkFreq,2)*size(fractions,2));

    for x=1:winSize
        for y=1:size(checkFreq,2)
            for frac=1:size(fractions,2)
                sinarr(x,(y-1)*size(fractions,2)+frac) = sin(x*checkFreq(y)*fractions(frac)*2*pi/fs);
                cosarr(x,(y-1)*size(fractions,2)+frac) = cos(x*checkFreq(y)*fractions(frac)*2*pi/fs);
            end
        end
    end

end